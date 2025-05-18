from __future__ import annotations
import torch
import os
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_saver import save_state_dict
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader
from typing import Optional, Union, Dict, Any, TYPE_CHECKING
from safetensors.torch import save_file as save_safetensors_file, load_file as load_safetensors_file
import numpy as np  # For saving/loading RNG state
import random  # For saving/loading RNG state

# Import for type hinting, moved outside TYPE_CHECKING for runtime availability
# Adjusted import path for wandb_logger
from clt.logging.wandb_logger import WandBLogger, DummyWandBLogger

# Forward declarations for type hinting to avoid circular imports
if TYPE_CHECKING:  # Use the imported TYPE_CHECKING
    from clt.models.clt import CrossLayerTranscoder
    from clt.training.data.base_store import BaseActivationStore
    from torch.optim import Optimizer
    from torch.cuda.amp.grad_scaler import GradScaler


class CheckpointManager:
    wandb_logger: Union[WandBLogger, DummyWandBLogger]

    def __init__(
        self,
        model: "CrossLayerTranscoder",
        activation_store: "BaseActivationStore",
        wandb_logger: Union[WandBLogger, DummyWandBLogger],
        log_dir: str,
        distributed: bool,
        rank: int,
        device: torch.device,
        world_size: int,
    ):
        self.model = model
        self.activation_store = activation_store
        self.wandb_logger = wandb_logger
        self.log_dir = log_dir
        self.distributed = distributed
        self.rank = rank
        self.device = device
        self.world_size = world_size

    def save_checkpoint(
        self,
        step: int,
        optimizer: "Optimizer",
        scheduler: Optional[Any],
        scaler: Optional["GradScaler"],
        n_forward_passes_since_fired: Optional[torch.Tensor],
    ):
        """Save a distributed checkpoint of the model, activation store, and trainer state.

        Args:
            step: Current training step.
            optimizer: The optimizer instance.
            scheduler: The learning rate scheduler instance (optional).
            scaler: The gradient scaler instance (optional, for mixed precision).
            n_forward_passes_since_fired: Tensor tracking dead neuron firing (optional).
        """
        trainer_state_to_save = {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler and scaler.is_enabled() else None,
            "n_forward_passes_since_fired": (
                n_forward_passes_since_fired.cpu() if n_forward_passes_since_fired is not None else None
            ),
            "wandb_run_id": self.wandb_logger.get_current_wandb_run_id(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        self._save_checkpoint(
            step=step,
            trainer_state_to_save=trainer_state_to_save,
        )

    def _save_checkpoint(
        self,
        step: int,
        trainer_state_to_save: Dict[str, Any],
    ):
        """Save a distributed checkpoint of the model and activation store state.

        Also saves optimizer, scheduler, step, scaler, dead neuron counter, and WandB run ID via trainer_state_to_save.

        Args:
            step: Current training step
            trainer_state_to_save: Dictionary containing all trainer-related states to save.
        """

        if not self.distributed:  # Non-distributed save
            os.makedirs(self.log_dir, exist_ok=True)
            model_checkpoint_path = os.path.join(self.log_dir, f"clt_checkpoint_{step}.safetensors")
            latest_model_path = os.path.join(self.log_dir, "clt_checkpoint_latest.safetensors")
            store_checkpoint_path = os.path.join(self.log_dir, f"activation_store_checkpoint_{step}.pt")
            latest_store_path = os.path.join(self.log_dir, "activation_store_checkpoint_latest.pt")
            trainer_state_path = os.path.join(self.log_dir, f"trainer_state_{step}.pt")
            latest_trainer_state_path = os.path.join(self.log_dir, "trainer_state_latest.pt")

            try:
                save_safetensors_file(self.model.state_dict(), model_checkpoint_path)
                save_safetensors_file(self.model.state_dict(), latest_model_path)
                if hasattr(self.wandb_logger, "log_artifact"):  # Check if real logger
                    self.wandb_logger.log_artifact(
                        artifact_path=model_checkpoint_path, artifact_type="model", name=f"clt_checkpoint_{step}"
                    )
                torch.save(self.activation_store.state_dict(), store_checkpoint_path)
                torch.save(self.activation_store.state_dict(), latest_store_path)
                torch.save(trainer_state_to_save, trainer_state_path)
                torch.save(trainer_state_to_save, latest_trainer_state_path)

            except Exception as e:
                print(f"Warning: Failed to save non-distributed checkpoint at step {step}: {e}")
            return

        checkpoint_dir = os.path.join(self.log_dir, f"step_{step}")
        latest_checkpoint_dir = os.path.join(self.log_dir, "latest")

        model_state_dict_for_dist_save = self.model.state_dict()
        try:
            save_state_dict(
                state_dict=model_state_dict_for_dist_save,
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=False,
            )
            save_state_dict(
                state_dict=model_state_dict_for_dist_save,
                storage_writer=FileSystemWriter(latest_checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=False,
            )
        except Exception as e:
            print(f"Rank {self.rank}: Warning: Failed to save distributed model checkpoint at step {step}: {e}")

        if self.rank == 0:
            store_checkpoint_path = os.path.join(checkpoint_dir, "activation_store.pt")
            latest_store_path = os.path.join(latest_checkpoint_dir, "activation_store.pt")
            try:
                torch.save(self.activation_store.state_dict(), store_checkpoint_path)
                torch.save(self.activation_store.state_dict(), latest_store_path)
            except Exception as e:
                print(f"Rank 0: Warning: Failed to save activation store state at step {step}: {e}")

            model_safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
            latest_model_safetensors_path = os.path.join(latest_checkpoint_dir, "model.safetensors")
            try:
                full_model_state_dict = self.model.state_dict()
                save_safetensors_file(full_model_state_dict, model_safetensors_path)
                save_safetensors_file(full_model_state_dict, latest_model_safetensors_path)
                print(
                    f"Rank 0: Saved consolidated model to {model_safetensors_path} and {latest_model_safetensors_path}"
                )
            except Exception as e:
                print(f"Rank 0: Warning: Failed to save consolidated .safetensors model at step {step}: {e}")

            trainer_state_filepath = os.path.join(checkpoint_dir, "trainer_state.pt")
            latest_trainer_state_filepath = os.path.join(latest_checkpoint_dir, "trainer_state.pt")
            try:
                torch.save(trainer_state_to_save, trainer_state_filepath)
                torch.save(trainer_state_to_save, latest_trainer_state_filepath)
                print(f"Rank 0: Saved trainer state to {trainer_state_filepath} and {latest_trainer_state_filepath}")
            except Exception as e:
                print(f"Rank 0: Warning: Failed to save trainer state at step {step}: {e}")

            if hasattr(self.wandb_logger, "log_artifact"):  # Check if real logger
                self.wandb_logger.log_artifact(
                    artifact_path=checkpoint_dir,
                    artifact_type="model_checkpoint",
                    name=f"dist_checkpoint_{step}",
                )
        if dist.is_initialized():
            dist.barrier()

    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: "Optimizer",
        scheduler: Optional[Any],
        scaler: Optional["GradScaler"],
    ) -> Dict[str, Any]:
        """Load a checkpoint for model, activation store, and apply trainer state.

        Args:
            checkpoint_path: Path to the checkpoint.
                             For non-distributed, this is the path to the model.safetensors file.
                             For distributed, this is the path to the *directory* (e.g., .../step_N).
            optimizer: The optimizer instance to load state into.
            scheduler: The learning rate scheduler instance (optional).
            scaler: The gradient scaler instance (optional, for mixed precision).

        Returns:
            A dictionary containing critical atomic states like 'step',
            'n_forward_passes_since_fired', and 'wandb_run_id'.
        """
        loaded_trainer_state_dict: Dict[str, Any] = {}
        return_atomic_states: Dict[str, Any] = {
            "step": 0,
            "n_forward_passes_since_fired": None,
            "wandb_run_id": None,
        }

        if not self.distributed:
            model_file_path = checkpoint_path
            if not (os.path.isfile(model_file_path) and model_file_path.endswith(".safetensors")):
                print(
                    f"Error: For non-distributed load, checkpoint_path must be a .safetensors model file. Got: {model_file_path}"
                )
                return return_atomic_states

            print(f"Attempting to load non-distributed checkpoint from model file: {model_file_path}")
            try:
                full_state_dict = load_safetensors_file(model_file_path, device=str(self.device))
                self.model.load_state_dict(full_state_dict)
                print(f"Successfully loaded non-distributed model from {model_file_path}")

                base_dir = os.path.dirname(model_file_path)
                base_name = os.path.basename(model_file_path)
                store_checkpoint_fname = ""
                trainer_state_fname = ""
                if base_name == "clt_checkpoint_latest.safetensors":
                    store_checkpoint_fname = "activation_store_checkpoint_latest.pt"
                    trainer_state_fname = "trainer_state_latest.pt"
                elif base_name.startswith("clt_checkpoint_") and base_name.endswith(".safetensors"):
                    step_str = base_name.replace("clt_checkpoint_", "").replace(".safetensors", "")
                    store_checkpoint_fname = f"activation_store_checkpoint_{step_str}.pt"
                    trainer_state_fname = f"trainer_state_{step_str}.pt"

                if store_checkpoint_fname and trainer_state_fname:
                    store_path = os.path.join(base_dir, store_checkpoint_fname)
                    trainer_state_path = os.path.join(base_dir, trainer_state_fname)
                    if os.path.exists(store_path):
                        self.activation_store.load_state_dict(torch.load(store_path, map_location=self.device))
                        print(f"Loaded activation store state from {store_path}")
                    if os.path.exists(trainer_state_path):
                        loaded_trainer_state_dict = torch.load(
                            trainer_state_path, map_location=self.device, weights_only=False
                        )
                        print(f"Loaded trainer state dict from {trainer_state_path}")
                else:
                    print(
                        f"Warning: Could not determine store/trainer state filenames from model path {model_file_path}"
                    )

            except Exception as e:
                print(f"Error loading non-distributed checkpoint from {model_file_path}: {e}")
                return return_atomic_states

        else:  # Distributed Load
            if not os.path.isdir(checkpoint_path):
                print(f"Error: Checkpoint path {checkpoint_path} is not a directory.")
                return return_atomic_states
            try:
                state_dict_to_load = self.model.state_dict()
                load_state_dict(
                    state_dict=state_dict_to_load,
                    storage_reader=FileSystemReader(checkpoint_path),
                    planner=DefaultLoadPlanner(),
                    no_dist=False,
                )
                print(f"Rank {self.rank}: Loaded distributed model checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Rank {self.rank}: Error loading distributed model checkpoint from {checkpoint_path}: {e}")
                consolidated_model_path = os.path.join(checkpoint_path, "model.safetensors")
                if os.path.exists(consolidated_model_path):
                    try:
                        full_model_state = load_safetensors_file(consolidated_model_path, device=str(self.device))
                        self.model.load_state_dict(full_model_state)
                        print(
                            f"Rank {self.rank}: Successfully loaded consolidated model from {consolidated_model_path}"
                        )
                    except Exception as e_consol:
                        print(f"Rank {self.rank}: Failed to load consolidated model: {e_consol}")
                        return return_atomic_states
                else:
                    print(f"Rank {self.rank}: Consolidated model.safetensors not found. Cannot fallback.")
                    return return_atomic_states

            if self.rank == 0:
                store_file_path = os.path.join(checkpoint_path, "activation_store.pt")
                if os.path.exists(store_file_path):
                    try:
                        self.activation_store.load_state_dict(torch.load(store_file_path, map_location=self.device))
                        print(f"Rank 0: Loaded activation store state from {store_file_path}")
                    except Exception as e:
                        print(f"Rank 0: Warning: Failed to load activation store state: {e}")
                trainer_state_file_path = os.path.join(checkpoint_path, "trainer_state.pt")
                if os.path.exists(trainer_state_file_path):
                    try:
                        loaded_trainer_state_dict = torch.load(
                            trainer_state_file_path, map_location="cpu", weights_only=False
                        )
                        print(f"Rank 0: Loaded trainer state dict from {trainer_state_file_path}")
                    except Exception as e:
                        print(f"Rank 0: Warning: Failed to load trainer state dict: {e}")

            if dist.is_initialized():  # Ensure barrier is only called if initialized
                dist.barrier()

            if dist.is_initialized() and self.world_size > 1:
                object_list = [loaded_trainer_state_dict if self.rank == 0 else {}]
                dist.broadcast_object_list(object_list, src=0)
                if self.rank != 0:
                    loaded_trainer_state_dict = object_list[0]
                if loaded_trainer_state_dict is None:  # Should be an empty dict if not rank 0 and broadcast worked
                    loaded_trainer_state_dict = {}
                print(
                    f"Rank {self.rank}: Received broadcasted trainer state dict. Step: {loaded_trainer_state_dict.get('step')}"
                )

        if loaded_trainer_state_dict:
            try:
                optimizer.load_state_dict(loaded_trainer_state_dict["optimizer_state_dict"])
                if scheduler and loaded_trainer_state_dict.get("scheduler_state_dict"):
                    scheduler.load_state_dict(loaded_trainer_state_dict["scheduler_state_dict"])
                if scaler and scaler.is_enabled() and loaded_trainer_state_dict.get("scaler_state_dict"):
                    scaler.load_state_dict(loaded_trainer_state_dict["scaler_state_dict"])

                if "torch_rng_state" in loaded_trainer_state_dict:
                    torch.set_rng_state(loaded_trainer_state_dict["torch_rng_state"].cpu())
                if "numpy_rng_state" in loaded_trainer_state_dict:
                    np.random.set_state(loaded_trainer_state_dict["numpy_rng_state"])
                if "python_rng_state" in loaded_trainer_state_dict:
                    random.setstate(loaded_trainer_state_dict["python_rng_state"])

                return_atomic_states["step"] = loaded_trainer_state_dict.get("step", 0)
                return_atomic_states["n_forward_passes_since_fired"] = loaded_trainer_state_dict.get(
                    "n_forward_passes_since_fired"
                )
                return_atomic_states["wandb_run_id"] = loaded_trainer_state_dict.get("wandb_run_id")

                print(
                    f"Rank {self.rank}: Applied optimizer, scheduler, scaler, and RNG states. Resuming from step {return_atomic_states['step']}"
                )

            except KeyError as e:
                print(
                    f"Rank {self.rank}: KeyError when applying loaded trainer state ({e}). Some states might not be restored."
                )
            except Exception as e:
                print(f"Rank {self.rank}: Error applying loaded trainer state: {e}")
        else:
            print(
                f"Rank {self.rank}: No trainer state dict was loaded or broadcasted. Starting fresh or from model/store defaults."
            )

        return return_atomic_states
