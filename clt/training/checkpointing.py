import torch
import os
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_saver import save_state_dict
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader
from typing import Optional, Union

# Forward declarations for type hinting to avoid circular imports
# These would typically be imported directly if not for potential circular dependencies
# For this refactoring, we assume these will be resolved by passing instances.
if False:  # TYPE_CHECKING
    from clt.models.clt import CrossLayerTranscoder
    from clt.training.data.base_store import BaseActivationStore
    from clt.training.wandb_logger import WandBLogger, DummyWandBLogger


class CheckpointManager:
    def __init__(
        self,
        model: "CrossLayerTranscoder",
        activation_store: "BaseActivationStore",
        wandb_logger: "Union['WandBLogger', 'DummyWandBLogger']",
        log_dir: str,
        distributed: bool,
        rank: int,
        device: torch.device,
    ):
        self.model = model
        self.activation_store = activation_store
        self.wandb_logger = wandb_logger
        self.log_dir = log_dir
        self.distributed = distributed
        self.rank = rank
        self.device = device

    def _save_checkpoint(self, step: int):
        """Save a distributed checkpoint of the model and activation store state.

        Uses torch.distributed.checkpoint to save sharded state directly.

        Args:
            step: Current training step
        """
        if not self.distributed:  # Non-distributed save
            os.makedirs(self.log_dir, exist_ok=True)
            model_checkpoint_path = os.path.join(self.log_dir, f"clt_checkpoint_{step}.pt")
            latest_model_path = os.path.join(self.log_dir, "clt_checkpoint_latest.pt")
            store_checkpoint_path = os.path.join(self.log_dir, f"activation_store_checkpoint_{step}.pt")
            latest_store_path = os.path.join(self.log_dir, "activation_store_checkpoint_latest.pt")

            try:
                # In non-distributed, model state_dict is the full dict
                torch.save(self.model.state_dict(), model_checkpoint_path)
                torch.save(self.model.state_dict(), latest_model_path)
                # Log checkpoint as artifact to WandB
                self.wandb_logger.log_artifact(
                    artifact_path=model_checkpoint_path, artifact_type="model", name=f"clt_checkpoint_{step}"
                )
                torch.save(self.activation_store.state_dict(), store_checkpoint_path)
                torch.save(self.activation_store.state_dict(), latest_store_path)
            except Exception as e:
                print(f"Warning: Failed to save non-distributed checkpoint at step {step}: {e}")
            return

        # --- Distributed Save ---
        # Define checkpoint directory for this step
        checkpoint_dir = os.path.join(self.log_dir, f"step_{step}")
        latest_checkpoint_dir = os.path.join(self.log_dir, "latest")

        # Save model state dict using distributed checkpointing
        # All ranks participate in saving their shard
        try:
            model_state_dict = self.model.state_dict()
            save_state_dict(
                state_dict=model_state_dict,
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=False,  # Ensure distributed save
            )
            # Also save latest (overwrites previous latest) - maybe link instead?
            # For simplicity, save again. Rank 0 can handle linking later if needed.
            save_state_dict(
                state_dict=model_state_dict,
                storage_writer=FileSystemWriter(latest_checkpoint_dir),
                planner=DefaultSavePlanner(),
                no_dist=False,
            )
        except Exception as e:
            print(f"Rank {self.rank}: Warning: Failed to save distributed model checkpoint at step {step}: {e}")

        # Save activation store state (only rank 0)
        if self.rank == 0:
            store_checkpoint_path = os.path.join(checkpoint_dir, "activation_store.pt")  # Save inside step dir
            latest_store_path = os.path.join(latest_checkpoint_dir, "activation_store.pt")
            try:
                torch.save(self.activation_store.state_dict(), store_checkpoint_path)
                torch.save(self.activation_store.state_dict(), latest_store_path)
            except Exception as e:
                print(f"Rank 0: Warning: Failed to save activation store state at step {step}: {e}")

            # Log checkpoint directory as artifact to WandB (only rank 0)
            self.wandb_logger.log_artifact(
                artifact_path=checkpoint_dir,  # Log the directory
                artifact_type="model_checkpoint",
                name=f"dist_checkpoint_{step}",
            )

        # Barrier to ensure all ranks finish saving before proceeding
        dist.barrier()

    def load_checkpoint(self, checkpoint_path: str):
        """Load a distributed checkpoint for model and activation store state.

        Args:
            checkpoint_path: Path to the *directory* containing the sharded checkpoint.
                             This should be the directory saved by _save_checkpoint (e.g., .../step_N).
        """
        if not os.path.isdir(checkpoint_path):
            print(
                f"Error: Checkpoint path {checkpoint_path} is not a directory. Distributed checkpoints are saved as directories."
            )
            # Check if it's a non-distributed .pt file for backward compatibility or single GPU runs
            if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pt") and not self.distributed:
                print("Attempting to load as non-distributed checkpoint file...")
                self._load_non_distributed_checkpoint(checkpoint_path)
                return
            else:
                return

        if not self.distributed:
            print("Attempting to load a distributed checkpoint directory in non-distributed mode.")
            print("Loading only rank 0 state from the directory (if possible)...")
            try:
                self.model.to(self.device)
                state_dict_to_load = self.model.state_dict()
                load_state_dict(
                    state_dict=state_dict_to_load,
                    storage_reader=FileSystemReader(checkpoint_path),
                    planner=DefaultLoadPlanner(),
                    no_dist=True,
                )
                print(
                    f"Successfully loaded and reconstructed full model state from sharded checkpoint {checkpoint_path}"
                )
            except Exception as e:
                print(f"Error loading distributed checkpoint into non-distributed model from {checkpoint_path}: {e}")
                print(
                    "This might indicate that the checkpoint format requires manual reconstruction or saving the full state dict separately on rank 0 during distributed save."
                )
                return

            store_checkpoint_path = os.path.join(checkpoint_path, "activation_store.pt")
            if os.path.exists(store_checkpoint_path):
                try:
                    store_state = torch.load(store_checkpoint_path, map_location=self.device)
                    if hasattr(self, "activation_store") and self.activation_store is not None:
                        self.activation_store.load_state_dict(store_state)
                        print(f"Loaded activation store state from {store_checkpoint_path}")
                    else:
                        print("Warning: Activation store not initialized. Cannot load state.")
                except Exception as e:
                    print(f"Warning: Failed to load activation store state from {store_checkpoint_path}: {e}")
            else:
                print(f"Warning: Activation store checkpoint not found in {checkpoint_path}")
            print("Warning: Loading sharded model parameters into non-distributed model is not fully implemented.")
            return

        # --- Distributed Load ---
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
            return

        if self.rank == 0:
            store_checkpoint_path = os.path.join(checkpoint_path, "activation_store.pt")
            if os.path.exists(store_checkpoint_path):
                try:
                    store_state = torch.load(store_checkpoint_path, map_location=self.device)
                    if hasattr(self, "activation_store") and self.activation_store is not None:
                        self.activation_store.load_state_dict(store_state)
                        print(f"Rank 0: Loaded activation store state from {store_checkpoint_path}")
                    else:
                        print("Rank 0: Warning: Activation store not initialized. Cannot load state.")
                except Exception as e:
                    print(f"Rank 0: Warning: Failed to load activation store state from {store_checkpoint_path}: {e}")
            else:
                print(f"Rank 0: Warning: Activation store checkpoint not found in {checkpoint_path}")

        dist.barrier()

    def _load_non_distributed_checkpoint(self, checkpoint_path: str, store_checkpoint_path: Optional[str] = None):
        """Loads a standard single-file model checkpoint."""
        if self.distributed:
            print("Error: Attempting to load non-distributed checkpoint in distributed mode.")
            return

        if not os.path.exists(checkpoint_path):
            print(f"Error: Model checkpoint not found at {checkpoint_path}")
            return
        try:
            full_state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(full_state_dict)
            print(f"Loaded non-distributed model checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading non-distributed model checkpoint from {checkpoint_path}: {e}")
            return

        if store_checkpoint_path is None:
            dirname = os.path.dirname(checkpoint_path)
            basename = os.path.basename(checkpoint_path)
            if basename.startswith("clt_checkpoint_"):
                store_basename = basename.replace("clt_checkpoint_", "activation_store_checkpoint_")
                store_checkpoint_path = os.path.join(dirname, store_basename)
            elif basename == "clt_checkpoint_latest.pt":
                store_checkpoint_path = os.path.join(dirname, "activation_store_checkpoint_latest.pt")
            else:
                store_checkpoint_path = None

        if store_checkpoint_path and os.path.exists(store_checkpoint_path):
            try:
                store_state = torch.load(store_checkpoint_path, map_location=self.device)
                if hasattr(self, "activation_store") and self.activation_store is not None:
                    self.activation_store.load_state_dict(store_state)
                    print(f"Loaded activation store state from {store_checkpoint_path}")
                else:
                    print("Warning: Activation store not initialized. Cannot load state.")
            except Exception as e:
                print(f"Warning: Failed to load activation store state from {store_checkpoint_path}: {e}")
        else:
            print(
                f"Warning: Activation store checkpoint path not found or specified: {store_checkpoint_path}. Store state not loaded."
            )
