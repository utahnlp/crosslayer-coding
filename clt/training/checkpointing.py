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
import logging

# Import for type hinting, moved outside TYPE_CHECKING for runtime availability
from clt.training.wandb_logger import WandBLogger, DummyWandBLogger

# Forward declarations for type hinting to avoid circular imports
if TYPE_CHECKING:
    from clt.models.clt import CrossLayerTranscoder
    from clt.training.data.base_store import BaseActivationStore

    # from clt.training.wandb_logger import WandBLogger, DummyWandBLogger # No longer needed here

logger = logging.getLogger(__name__)


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
        # Add optimizer, scheduler, scaler to be available for loading if needed
        # For saving, they will be passed to _save_checkpoint
    ):
        self.model = model
        self.activation_store = activation_store
        self.wandb_logger = wandb_logger
        self.log_dir = log_dir
        self.distributed = distributed
        self.rank = rank
        self.device = device
        self.world_size = world_size

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
        # The trainer_state_to_save already contains step, optimizer_state_dict, etc.
        # For example: trainer_state_to_save["step"] should be === step passed to this function.
        # We will save this entire dictionary.

        if not self.distributed:  # Non-distributed save
            os.makedirs(self.log_dir, exist_ok=True)
            model_checkpoint_path = os.path.join(self.log_dir, f"clt_checkpoint_{step}.safetensors")
            latest_model_path = os.path.join(self.log_dir, "clt_checkpoint_latest.safetensors")
            store_checkpoint_path = os.path.join(self.log_dir, f"activation_store_checkpoint_{step}.pt")
            latest_store_path = os.path.join(self.log_dir, "activation_store_checkpoint_latest.pt")
            trainer_state_path = os.path.join(self.log_dir, f"trainer_state_{step}.pt")
            latest_trainer_state_path = os.path.join(self.log_dir, "trainer_state_latest.pt")

            try:
                # Save model
                save_safetensors_file(self.model.state_dict(), model_checkpoint_path)
                save_safetensors_file(self.model.state_dict(), latest_model_path)
                self.wandb_logger.log_artifact(
                    artifact_path=model_checkpoint_path, artifact_type="model", name=f"clt_checkpoint_{step}"
                )
                # Save activation store
                torch.save(self.activation_store.state_dict(), store_checkpoint_path)
                torch.save(self.activation_store.state_dict(), latest_store_path)
                # Save trainer state (now passed as a dictionary)
                torch.save(trainer_state_to_save, trainer_state_path)
                torch.save(trainer_state_to_save, latest_trainer_state_path)

            except Exception as e:
                logger.warning(f"Warning: Failed to save non-distributed checkpoint at step {step}: {e}")
            return

        # --- Distributed Save ---
        checkpoint_dir = os.path.join(self.log_dir, f"step_{step}")
        latest_checkpoint_dir = os.path.join(self.log_dir, "latest")

        # Save model state dict using distributed checkpointing
        model_state_dict_for_dist_save = self.model.state_dict()
        try:
            # Disable tensor deduplication so that identically shaped but **sharded**
            # parameters (e.g. TP slices whose shapes are padded to be uniform across
            # ranks) are still treated as rank-local shards rather than as replicated
            # tensors.  Without this, only rank-0 data would be saved and, on load, every
            # rank would receive the *same* weights, destroying the learned TP sharding.
            planner_no_dedup = DefaultSavePlanner(dedup_replicated_tensors=False)

            save_state_dict(
                state_dict=model_state_dict_for_dist_save,
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=planner_no_dedup,
                no_dist=False,
            )

            save_state_dict(
                state_dict=model_state_dict_for_dist_save,
                storage_writer=FileSystemWriter(latest_checkpoint_dir),
                planner=planner_no_dedup,
                no_dist=False,
            )
        except Exception as e:
            logger.warning(
                f"Rank {self.rank}: Warning: Failed to save distributed model checkpoint at step {step}: {e}"
            )

        if self.rank == 0:
            # Save activation store
            store_checkpoint_path = os.path.join(checkpoint_dir, "activation_store.pt")
            latest_store_path = os.path.join(latest_checkpoint_dir, "activation_store.pt")
            try:
                torch.save(self.activation_store.state_dict(), store_checkpoint_path)
                torch.save(self.activation_store.state_dict(), latest_store_path)
            except Exception as e:
                logger.warning(f"Rank 0: Warning: Failed to save activation store state at step {step}: {e}")

            # Save consolidated model as .safetensors
            model_safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
            latest_model_safetensors_path = os.path.join(latest_checkpoint_dir, "model.safetensors")
            try:
                full_model_state_dict = self.model.state_dict()
                save_safetensors_file(full_model_state_dict, model_safetensors_path)
                save_safetensors_file(full_model_state_dict, latest_model_safetensors_path)
                logger.info(
                    f"Rank 0: Saved consolidated model to {model_safetensors_path} and {latest_model_safetensors_path}"
                )
            except Exception as e:
                logger.warning(f"Rank 0: Warning: Failed to save consolidated .safetensors model at step {step}: {e}")

            # Save trainer state (optimizer, scheduler, etc.)
            trainer_state_filepath = os.path.join(checkpoint_dir, "trainer_state.pt")
            latest_trainer_state_filepath = os.path.join(latest_checkpoint_dir, "trainer_state.pt")
            try:
                torch.save(trainer_state_to_save, trainer_state_filepath)
                torch.save(trainer_state_to_save, latest_trainer_state_filepath)
                logger.info(
                    f"Rank 0: Saved trainer state to {trainer_state_filepath} and {latest_trainer_state_filepath}"
                )
            except Exception as e:
                logger.warning(f"Rank 0: Warning: Failed to save trainer state at step {step}: {e}")

            self.wandb_logger.log_artifact(
                artifact_path=checkpoint_dir,
                artifact_type="model_checkpoint",
                name=f"dist_checkpoint_{step}",
            )

        dist.barrier()

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint for model, activation store, and trainer state.

        Args:
            checkpoint_path: Path to the checkpoint.
                             For non-distributed, this is the path to the model.safetensors file.
                             For distributed, this is the path to the *directory* (e.g., .../step_N).

        Returns:
            A dictionary containing the loaded trainer state (optimizer, scheduler, step, etc.).
        """
        loaded_trainer_state: Dict[str, Any] = {}

        if not self.distributed:  # Non-distributed load
            model_file_path = checkpoint_path  # Expecting path to .safetensors
            if not (os.path.isfile(model_file_path) and model_file_path.endswith(".safetensors")):
                logger.error(
                    f"Error: For non-distributed load, checkpoint_path must be a .safetensors model file. Got: {model_file_path}"
                )
                return loaded_trainer_state  # Return empty if path is not as expected

            logger.info(f"Attempting to load non-distributed checkpoint from model file: {model_file_path}")
            try:
                full_state_dict = load_safetensors_file(model_file_path, device=str(self.device))
                self.model.load_state_dict(full_state_dict)
                logger.info(f"Successfully loaded non-distributed model from {model_file_path}")

                # Infer paths for store and trainer state
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

                if not store_checkpoint_fname or not trainer_state_fname:
                    logger.warning(
                        f"Warning: Could not determine store/trainer state filenames from model path {model_file_path}"
                    )
                else:
                    store_path = os.path.join(base_dir, store_checkpoint_fname)
                    trainer_state_path = os.path.join(base_dir, trainer_state_fname)

                    # Load activation store
                    if os.path.exists(store_path):
                        store_state = torch.load(store_path, map_location=self.device)
                        self.activation_store.load_state_dict(store_state)
                        logger.info(f"Loaded activation store state from {store_path}")
                    else:
                        logger.warning(f"Warning: Activation store checkpoint not found at {store_path}")

                    # Load trainer state
                    if os.path.exists(trainer_state_path):
                        loaded_trainer_state = torch.load(
                            trainer_state_path, map_location=self.device, weights_only=False
                        )
                        logger.info(f"Loaded trainer state from {trainer_state_path}")
                    else:
                        logger.warning(f"Warning: Trainer state checkpoint not found at {trainer_state_path}")

            except Exception as e:
                logger.error(f"Error loading non-distributed checkpoint from {model_file_path}: {e}")
            return loaded_trainer_state

        # --- Distributed Load ---
        # checkpoint_path is a directory for distributed checkpoints
        if not os.path.isdir(checkpoint_path):
            logger.error(
                f"Error: Checkpoint path {checkpoint_path} is not a directory. Distributed checkpoints are saved as directories."
            )
            return loaded_trainer_state

        try:
            state_dict_to_load = self.model.state_dict()  # Get a template
            load_state_dict(
                state_dict=state_dict_to_load,
                storage_reader=FileSystemReader(checkpoint_path),
                planner=DefaultLoadPlanner(),
                no_dist=False,  # Distributed load
            )
            # The model's state_dict is modified in-place by load_state_dict
            # self.model.load_state_dict(state_dict_to_load) # Not needed if modified in-place
            logger.info(f"Rank {self.rank}: Loaded distributed model checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error loading distributed model checkpoint from {checkpoint_path}: {e}")
            # Attempt to load consolidated if sharded load fails (e.g. loading TP model on single GPU)
            # This part is tricky because load_state_dict above might have partially modified the model.
            # A cleaner approach for "load TP sharded on single GPU" would be separate.
            # For now, if distributed load_state_dict fails, we try the consolidated .safetensors
            logger.info(
                f"Rank {self.rank}: Attempting to load consolidated model.safetensors from the directory as fallback."
            )
            consolidated_model_path = os.path.join(checkpoint_path, "model.safetensors")
            if os.path.exists(consolidated_model_path):
                try:
                    # This load is for a single rank, assuming this rank needs the full model
                    full_model_state = load_safetensors_file(consolidated_model_path, device=str(self.device))
                    self.model.load_state_dict(full_model_state)
                    logger.info(
                        f"Rank {self.rank}: Successfully loaded consolidated model from {consolidated_model_path}"
                    )
                except Exception as e_consol:
                    logger.error(
                        f"Rank {self.rank}: Failed to load consolidated model from {consolidated_model_path}: {e_consol}"
                    )
                    return loaded_trainer_state  # Failed both sharded and consolidated
            else:
                logger.info(
                    f"Rank {self.rank}: Consolidated model.safetensors not found in {checkpoint_path}. Cannot fallback."
                )
                return loaded_trainer_state  # Failed sharded, no consolidated to fallback to

        # Load activation store and trainer state (rank 0 handles files, then potentially broadcasts or uses)
        # For now, only rank 0 loads these files. Other ranks will get empty trainer_state.
        # CLTTrainer will need to handle broadcasting/synchronizing the step number, etc.
        if self.rank == 0:
            store_file_path = os.path.join(checkpoint_path, "activation_store.pt")
            if os.path.exists(store_file_path):
                try:
                    store_state = torch.load(store_file_path, map_location=self.device)
                    self.activation_store.load_state_dict(store_state)
                    logger.info(f"Rank 0: Loaded activation store state from {store_file_path}")
                except Exception as e:
                    logger.warning(
                        f"Rank 0: Warning: Failed to load activation store state from {store_file_path}: {e}"
                    )
            else:
                logger.warning(f"Rank 0: Warning: Activation store checkpoint not found in {checkpoint_path}")

            trainer_state_file_path = os.path.join(checkpoint_path, "trainer_state.pt")
            if os.path.exists(trainer_state_file_path):
                try:
                    # map_location CPU for items that might be on CUDA but not needed there by all ranks yet
                    loaded_trainer_state = torch.load(trainer_state_file_path, map_location="cpu", weights_only=False)
                    logger.info(f"Rank 0: Loaded trainer state from {trainer_state_file_path}")
                except Exception as e:
                    logger.warning(f"Rank 0: Warning: Failed to load trainer state from {trainer_state_file_path}: {e}")
            else:
                logger.warning(f"Rank 0: Warning: Trainer state file not found in {checkpoint_path}")

        # Barrier to ensure all ranks have attempted model loading before proceeding.
        # And rank 0 has loaded other states.
        if dist.is_initialized():  # Check if distributed context is active
            dist.barrier()

        # Broadcast trainer_state from rank 0 to all other ranks
        # This is important because all ranks need the step number, and potentially other states.
        if dist.is_initialized() and self.world_size > 1:
            # Create a list of one element to use with broadcast_object_list
            object_list = [loaded_trainer_state if self.rank == 0 else {}]
            dist.broadcast_object_list(object_list, src=0)
            if self.rank != 0:
                loaded_trainer_state = object_list[0]
            # Ensure loaded_trainer_state is a dict even if broadcast failed or returned None (defensive)
            if loaded_trainer_state is None:
                loaded_trainer_state = {}
            logger.info(
                f"Rank {self.rank}: Received broadcasted trainer state. Step: {loaded_trainer_state.get('step')}"
            )

        return loaded_trainer_state

    def _load_non_distributed_checkpoint(self, checkpoint_path: str, store_checkpoint_path: Optional[str] = None):
        """Loads a standard single-file model checkpoint (.pt or .safetensors)."""
        if self.distributed:
            logger.error("Error: Attempting to load non-distributed checkpoint in distributed mode.")
            return

        if not os.path.exists(checkpoint_path):
            logger.error(f"Error: Model checkpoint not found at {checkpoint_path}")
            return
        try:
            if checkpoint_path.endswith(".safetensors"):
                # For safetensors, device mapping is usually handled by loading directly to device or moving tensors after.
                # load_file takes device string like "cuda:0"
                full_state_dict = load_safetensors_file(checkpoint_path, device=str(self.device))
            elif checkpoint_path.endswith(".pt"):
                full_state_dict = torch.load(checkpoint_path, map_location=self.device)
            else:
                logger.error(
                    f"Error: Unknown checkpoint file extension for {checkpoint_path}. Must be .pt or .safetensors."
                )
                return
            self.model.load_state_dict(full_state_dict)
            logger.info(f"Loaded non-distributed model checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading non-distributed model checkpoint from {checkpoint_path}: {e}")
            return

        if store_checkpoint_path is None:
            dirname = os.path.dirname(checkpoint_path)
            basename = os.path.basename(checkpoint_path)
            # Adjust base name detection for new extension
            if basename.startswith("clt_checkpoint_"):
                store_basename_prefix = basename.split("clt_checkpoint_")[1]
                if store_basename_prefix.endswith(".safetensors"):
                    store_basename_prefix = store_basename_prefix.replace(".safetensors", ".pt")
                elif store_basename_prefix.endswith(".pt"):
                    store_basename_prefix = store_basename_prefix.replace(".pt", ".pt")  # no change if already .pt
                else:  # for older checkpoints that might not have extension in prefix string
                    store_basename_prefix = store_basename_prefix + ".pt"

                # Ensure it correctly forms activation_store_checkpoint_{step}.pt
                if "latest" in basename:
                    store_basename = "activation_store_checkpoint_latest.pt"
                else:
                    # Extract step from basename like clt_checkpoint_100.safetensors -> 100
                    step_str = basename.split("_")[-1].split(".")[0]
                    store_basename = f"activation_store_checkpoint_{step_str}.pt"
                store_checkpoint_path = os.path.join(dirname, store_basename)
            # No change for clt_checkpoint_latest.pt because it's specific enough
            elif basename == "clt_checkpoint_latest.pt" or basename == "clt_checkpoint_latest.safetensors":
                store_checkpoint_path = os.path.join(dirname, "activation_store_checkpoint_latest.pt")
            else:
                store_checkpoint_path = None

        if store_checkpoint_path and os.path.exists(store_checkpoint_path):
            try:
                store_state = torch.load(store_checkpoint_path, map_location=self.device)
                if hasattr(self, "activation_store") and self.activation_store is not None:
                    self.activation_store.load_state_dict(store_state)
                    logger.info(f"Loaded activation store state from {store_checkpoint_path}")
                else:
                    logger.warning("Warning: Activation store not initialized. Cannot load state.")
            except Exception as e:
                logger.warning(f"Warning: Failed to load activation store state from {store_checkpoint_path}: {e}")
        else:
            logger.warning(
                f"Warning: Activation store checkpoint path not found or specified: {store_checkpoint_path}. Store state not loaded."
            )
