import argparse
import copy
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterator

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm

from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder
from clt.training.checkpointing import CheckpointManager
from clt.training.wandb_logger import DummyWandBLogger
from clt.training.data.base_store import BaseActivationStore


# --- Minimal dummy activation store (same behavior as original) ---
class DummyActivationStore(BaseActivationStore):
    def __iter__(self): return self
    def __next__(self): raise StopIteration
    def state_dict(self): return {}
    def load_state_dict(self, _): pass
    def close(self): pass
    def get_batch(self): pass


# --- CLT patching wrapper (kept logic but simplified) ---
class CLTPatchedModel(torch.nn.Module):
    """
    Wrap an existing causal LM and patch identified 'MLP-like' modules so their
    forward uses CLT reconstructions when available. Retains detailed debug info.
    """
    MLP_NAME_PATTERNS = ("mlp", "feed_forward", "ffn")

    def __init__(self, original_model: torch.nn.Module, clt_model: CrossLayerTranscoder):
        super().__init__()
        self.original_model = original_model
        self.clt_model = clt_model
        self.device = original_model.device
        self.dtype = original_model.dtype

        # debugging state
        self.debug_info = {
            "total_mlp_layers": 0,
            "successful_patches": 0,
            "failed_patches": 0,
            "layer_output_differences": [],
            "activation_shapes": {},
        }

        self._patch_mlp_layers()

    def _is_mlp_name(self, name: str) -> bool:
        return any(pat in name.lower() for pat in self.MLP_NAME_PATTERNS)

    def _patch_mlp_layers(self):
        # collect candidate modules by dotted name
        candidates = [
            (name, mod) for name, mod in self.original_model.named_modules()
            if name and self._is_mlp_name(name) and any(isinstance(c, torch.nn.Linear) for c in mod.children())
        ]
        self.debug_info["total_mlp_layers"] = len(candidates)

        for idx, (path, mlp_mod) in enumerate(candidates):
            if idx >= getattr(self.clt_model.config, "num_layers", idx + 1):
                print(f"Warning: CLT has fewer layers than found MLPs; skipping {path}")
                self.debug_info["failed_patches"] += 1
                continue

            original_forward = mlp_mod.forward

            # create a patched forward factory to avoid late-binding loop issues
            def make_forward(orig_fwd, layer_idx, mod_path):
                def patched_forward(*args, **kwargs):
                    # find input tensor (prefer first arg, then hidden_states key)
                    x = args[0] if args else kwargs.get("hidden_states")
                    if x is None:
                        # try to find first tensor-like arg
                        for v in kwargs.values():
                            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                                x = v
                                break
                    if x is None:
                        print(f"Warning: could not locate input for layer {layer_idx} ({mod_path}); falling back")
                        return orig_fwd(*args, **kwargs)

                    # move to CLT device/dtype
                    x_clt = x.to(device=self.clt_model.device, dtype=self.clt_model.dtype)
                    b, s, h = x_clt.shape
                    self.debug_info["activation_shapes"][f"layer_{layer_idx}"] = {"batch_size": b, "seq_len": s, "hidden_size": h}

                    with torch.no_grad():
                        # CLT expects flattened [batch*seq, hidden]
                        clt_input = {layer_idx: x_clt.reshape(-1, h)}
                        clt_outputs = self.clt_model.forward(clt_input)
                        if layer_idx not in clt_outputs:
                            # no reconstruction available for this layer
                            # fall back to original
                            return orig_fwd(*args, **kwargs)

                        clt_recon = clt_outputs[layer_idx].reshape(b, s, h)
                        # Compare against original for debug (but return CLT reconstruction)
                        original_out = orig_fwd(*args, **kwargs)
                        diff = torch.mean(torch.abs(original_out - clt_recon)).item()
                        max_diff = torch.max(torch.abs(original_out - clt_recon)).item()
                        self.debug_info["layer_output_differences"].append({"layer": layer_idx, "mean_diff": diff, "max_diff": max_diff})
                        if diff < 1e-6:
                            print(f"⚠️ Layer {layer_idx} ({mod_path}): CLT output nearly identical to original (diff {diff:.6g})")
                        return clt_recon.to(device=self.device, dtype=self.dtype)
                return patched_forward

            try:
                mlp_mod.forward = make_forward(original_forward, idx, path)
                self.debug_info["successful_patches"] += 1
                # print(f"✅ patched layer {idx} at {path}")
            except Exception as e:
                self.debug_info["failed_patches"] += 1
                print(f"❌ failed to patch {path}: {e}")

    def forward(self, *args, **kwargs):
        if not getattr(self, "_first_forward_done", False):
            self._first_forward_done = True
            print("\n=== FIRST FORWARD PASS (CLT patches active) ===")
        out = self.original_model(*args, **kwargs)
        if getattr(self, "_first_forward_done", False) and not getattr(self, "_debug_summary_printed", False):
            self._debug_summary_printed = True
            # self._print_debug_summary()
        return out

    def _print_debug_summary(self):
        print("\n=== CLT PATCHING SUMMARY ===")
        print(f"Found MLP candidates: {self.debug_info['total_mlp_layers']}")
        print(f"Patched: {self.debug_info['successful_patches']}, Failed: {self.debug_info['failed_patches']}")
        if self.debug_info["layer_output_differences"]:
            avg = sum(d["mean_diff"] for d in self.debug_info["layer_output_differences"]) / len(self.debug_info["layer_output_differences"])
            print(f"Average mean diff: {avg:.6g}")
            if avg < 1e-6:
                print("❌ CRITICAL: CLT outputs nearly identical to original MLP outputs.")
            else:
                print("✅ CLT modifies MLP outputs.")
        # parameter counts & small gradient-flow test (kept from original)
        enc_params = sum(p.numel() for p in self.clt_model.encoder_module.parameters())
        dec_params = sum(p.numel() for p in self.clt_model.decoder_module.parameters())
        print(f"CLT encoder params: {enc_params:,}, decoder params: {dec_params:,}")
        print("--- gradient flow test ---")
        test_in = torch.randn(1, 10, self.clt_model.config.d_model, device=self.clt_model.device)
        test_out = self.clt_model.forward({0: test_in.reshape(-1, self.clt_model.config.d_model)})
        if 0 in test_out and test_out[0].requires_grad:
            print("✅ CLT output requires gradients")
        else:
            print("❌ CLT output does not require gradients or no output for layer 0")
        print("=" * 50)

    def remove_patches(self):
        print("CLT patches are active. To remove, reload the original model.")

    def get_debug_info(self) -> Dict[str, Any]:
        return self.debug_info


# --- Model evaluator with common helpers to reduce duplication ---
class ModelEvaluator:
    def __init__(self, model_name: str, device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.target_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device)
        self.target_model.eval()
        print(f"---> Loaded model {model_name} on {self.device}")

    def load_replacement_model(self, checkpoint_path: str) -> CLTPatchedModel:
        print(f"\nLoading CLT from {checkpoint_path} ...")
        clt_model = load_clt_checkpoint(checkpoint_path, self.device)
        print(f"---> Loaded CLT: {clt_model.config.num_layers} layers, {clt_model.config.num_features} features")
        orig_copy = copy.deepcopy(self.target_model)
        return CLTPatchedModel(orig_copy, clt_model)

    def load_target_dataset(self, dataset_name: str, subset_name: Optional[str] = None, split: str = "train",
                            text_field: Optional[str] = None, max_examples: int = 100) -> Tuple[Any, str]:
        print(f"---> Loading dataset {dataset_name} ...")
        if subset_name is None:
            try:
                configs = get_dataset_config_names(dataset_name)
                subset_name = configs[0] if configs else None
            except Exception:
                subset_name = None

        ds = load_dataset(dataset_name, subset_name) if subset_name else load_dataset(dataset_name)
        # pick a split (preserve original fallbacks)
        if split in ds:
            ds = ds[split]
        elif 0 in ds:
            ds = ds[0]
        else:
            ds = ds[list(ds.keys())[0]]

        if text_field is None:
            text_field = self._detect_text_field(ds)

        ds_small = ds.select(range(min(max_examples, len(ds))))
        print(f"Using {len(ds_small)} examples with text field: '{text_field}'")
        return ds_small, text_field

    @staticmethod
    def _detect_text_field(dataset) -> str:
        candidates = ["text", "question", "content", "sentence", "document", "doc"]
        for f in candidates:
            if f in dataset.features:
                return f
        for field, feature in dataset.features.items():
            if hasattr(feature, "dtype") and feature.dtype in ("string", "str"):
                return field
        raise ValueError(f"Could not detect text field. Available: {list(dataset.features.keys())}")

    # --- helper generator that yields tokenized inputs on device and valid text ---
    def _valid_tokenized_inputs(self, dataset, text_field: str, max_length: int):
        for ex in dataset:
            text = ex.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            if inputs.input_ids.size(1) == 0:
                continue
            yield {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def _forward_pair(self, replacement_model: torch.nn.Module, inputs: Dict[str, torch.Tensor]):
        tgt_logits = self.target_model(**inputs).logits
        repl_logits = replacement_model(**inputs).logits
        return tgt_logits, repl_logits

    @torch.no_grad()
    def evaluate_kl_divergence(self, replacement_model: torch.nn.Module, dataset, text_field: str, max_length: int = 512) -> float:
        total_kl, count = 0.0, 0
        for inputs in tqdm(self._valid_tokenized_inputs(dataset, text_field, max_length), desc="Computing KL divergence"):
            tgt, repl = self._forward_pair(replacement_model, inputs)
            mn = min(tgt.size(1), repl.size(1))
            kl = F.kl_div(F.log_softmax(repl[:, :mn, :], dim=-1),
                          F.softmax(tgt[:, :mn, :], dim=-1),
                          reduction="batchmean")
            total_kl += kl.item(); count += 1
        return total_kl / count if count else 0.0

    @torch.no_grad()
    def evaluate_token_agreement(self, replacement_model: torch.nn.Module, dataset, text_field: str, max_length: int = 512) -> float:
        total_matches, total_tokens = 0, 0
        for inputs in tqdm(self._valid_tokenized_inputs(dataset, text_field, max_length), desc="Computing token agreement"):
            tgt, repl = self._forward_pair(replacement_model, inputs)
            mn = min(tgt.size(1), repl.size(1))
            tgt_preds = torch.argmax(tgt[:, :mn, :], dim=-1)
            repl_preds = torch.argmax(repl[:, :mn, :], dim=-1)
            total_matches += (tgt_preds == repl_preds).sum().item()
            total_tokens += tgt_preds.numel()
        return total_matches / total_tokens if total_tokens else 0.0

    @torch.no_grad()
    def evaluate_global_kl(self, replacement_model: torch.nn.Module, dataset, text_field: str, max_length: int = 512) -> float:
        sum_tgt, sum_repl, count = None, None, 0
        for inputs in tqdm(self._valid_tokenized_inputs(dataset, text_field, max_length), desc="Computing global KL"):
            tgt, repl = self._forward_pair(replacement_model, inputs)
            tgt_probs = F.softmax(tgt, dim=-1)
            repl_probs = F.softmax(repl, dim=-1)
            tgt_mean = tgt_probs.mean(dim=1).sum(dim=0)
            repl_mean = repl_probs.mean(dim=1).sum(dim=0)
            sum_tgt = tgt_mean if sum_tgt is None else sum_tgt + tgt_mean
            sum_repl = repl_mean if sum_repl is None else sum_repl + repl_mean
            count += 1
        if count == 0:
            return 0.0
        tgt_dist = sum_tgt / sum_tgt.sum()
        repl_dist = sum_repl / sum_repl.sum()
        kl = (tgt_dist * (tgt_dist.log() - repl_dist.log())).sum().item()
        return kl

    def evaluate_checkpoint(self, checkpoint_path: str, dataset, text_field: str, max_length: int = 512) -> Dict[str, float]:
        print(f"\nEvaluating checkpoint: {checkpoint_path}")
        replacement_model = self.load_replacement_model(checkpoint_path)
        replacement_model.eval()

        # Debugging (kept from original, simplified)
        print("=== CLT INTEGRATION DEBUG ===")
        print(f"CLT config: {replacement_model.clt_model.config}")
        print(f"CLT device/dtype: {replacement_model.clt_model.device}/{replacement_model.clt_model.dtype}")
        print(f"Registered hooks (approx): {len(getattr(replacement_model, 'hooks', []))}")

        # small layer-by-layer debug using a canonical sample (kept original intent)
        sample_text = "The quick brown fox jumps over the lazy dog"
        inputs = self.tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=32).to(self.device)
        with torch.no_grad():
            # attempt to capture original MLP layer outputs via hooks on the unpatched target model
            original_outputs = {}
            def make_layer_hook(i):
                def hook(_, __, out): original_outputs[f"layer_{i}"] = out.detach().clone()
                return hook

            original_hooks = []
            # best-effort: mirror the original code's assumption about .model.layers
            try:
                for i in range(replacement_model.clt_model.config.num_layers):
                    mlp_module = self.target_model.model.layers[i].mlp
                    original_hooks.append(mlp_module.register_forward_hook(make_layer_hook(i)))
                original_logits = self.target_model(**inputs).logits
            except Exception:
                original_logits = self.target_model(**inputs).logits
            finally:
                for h in original_hooks: h.remove()

            patched_logits = replacement_model(**inputs).logits
            diff = torch.mean(torch.abs(original_logits - patched_logits)).item()
            max_diff = torch.max(torch.abs(original_logits - patched_logits)).item()
            print(f"Average output diff: {diff:.6g}, Max diff: {max_diff:.6g}")
            if diff < 1e-6:
                print("❌ CRITICAL: outputs nearly identical; CLT may not be applied.")
            else:
                print("✅ CLT appears to change outputs.")

        results = {
            "kl_divergence": self.evaluate_kl_divergence(replacement_model, dataset, text_field, max_length),
            "token_agreement": self.evaluate_token_agreement(replacement_model, dataset, text_field, max_length),
            "global_kl": self.evaluate_global_kl(replacement_model, dataset, text_field, max_length),
        }

        print(f"Results for {checkpoint_path}: \n\tKL={results['kl_divergence']:.6f}\n\tTokenAgreement={results['token_agreement']:.4%}\n\tGlobalKL={results['global_kl']:.6f}")
        return results


# --- loading CLT checkpoint (kept same behavior) ---
def load_clt_checkpoint(checkpoint_path: str, device: torch.device, config_path: Optional[str] = None) -> CrossLayerTranscoder:
    checkpoint_path = Path(checkpoint_path)
    is_distributed = checkpoint_path.is_dir()

    if config_path is None:
        candidates = [
            checkpoint_path / "cfg.json" if is_distributed else checkpoint_path.parent / "cfg.json",
            checkpoint_path.parent / "cfg.json",
            checkpoint_path.parent.parent / "cfg.json",
        ]
        for c in candidates:
            if c and c.exists():
                config_path = str(c); break
        if config_path is None:
            raise FileNotFoundError(f"cfg.json not found. Searched: {candidates}")

    with open(config_path, "r") as f:
        cfg = CLTConfig(**json.load(f))

    model = CrossLayerTranscoder(cfg, process_group=None, device=device, profiler=None)
    dummy_store = DummyActivationStore()
    dummy_logger = DummyWandBLogger(None, cfg, "", None)
    cp_mgr = CheckpointManager(model=model, activation_store=dummy_store, wandb_logger=dummy_logger,
                               log_dir="", distributed=False, rank=0, device=device, world_size=1)
    cp_mgr.load_checkpoint(str(checkpoint_path))
    model.eval()
    return model


# --- CLI (kept the same flags and behavior) ---
def main():
    p = argparse.ArgumentParser(description="Evaluate CLT checkpoints")
    p.add_argument("--model", required=True, help="Target model name")
    p.add_argument("--dataset", required=True, help="Dataset: 'name' or 'name:config'")
    p.add_argument("--checkpoint", help="Single checkpoint path to evaluate")
    p.add_argument("--checkpoint-dir", help="Directory containing checkpoints")
    p.add_argument("--text-field", help="Text field name (auto-detect if not provided)")
    p.add_argument("--split", default="train", help="Dataset split to use")
    p.add_argument("--max-examples", type=int, default=100, help="Max examples to evaluate")
    p.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    p.add_argument("--device", default="auto", help="Device to use")
    args = p.parse_args()

    evaluator = ModelEvaluator(args.model, args.device)
    dataset, text_field = evaluator.load_target_dataset(args.dataset, split=args.split, text_field=args.text_field, max_examples=args.max_examples)

    if args.checkpoint:
        evaluator.evaluate_checkpoint(args.checkpoint, dataset, text_field, args.max_length)
    elif args.checkpoint_dir:
        ck_dir = Path(args.checkpoint_dir)
        checkpoints = sorted(d for d in ck_dir.iterdir() if d.is_dir() and d.name.startswith("step_"))
        for ck in checkpoints:
            try:
                evaluator.evaluate_checkpoint(str(ck), dataset, text_field, args.max_length)
            except Exception as e:
                print(f"Error evaluating {ck}: {e}")
                import traceback; traceback.print_exc()
    else:
        p.error("Must specify either --checkpoint or --checkpoint-dir")


if __name__ == "__main__":
    main()


# Results for ../../crosslayer-coding-v2/data/clt/allenai/OLMo-2-0425-1B-Instruct/olmo-mix-1124_train_1000000_float32/100feats_4batchsize/step_3000/model.safetensors:
#   KL Divergence: 349.392850
#   Token Agreement: 0.4372%
#   Global KL: 3.345129
# Results for ../../crosslayer-coding-v2/data/clt/allenai/OLMo-2-0425-1B-Instruct/olmo-mix-1124_train_1000000_float32/100feats_4batchsize/step_3000/model.safetensors: 
# KL=349.392850, TokenAgreement=0.4372%, GlobalKL=3.345129