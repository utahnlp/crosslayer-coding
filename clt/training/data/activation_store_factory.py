import logging
import torch

from clt.config import TrainingConfig, CLTConfig, ActivationConfig
from clt.nnsight.extractor import ActivationExtractorCLT
from clt.training.data.base_store import BaseActivationStore
from clt.training.data.local_activation_store import LocalActivationStore
from clt.training.data.remote_activation_store import RemoteActivationStore
from clt.training.data.streaming_activation_store import StreamingActivationStore

logger = logging.getLogger(__name__)


def create_activation_store(
    training_config: TrainingConfig,
    clt_config: CLTConfig,
    device: torch.device,
    rank: int,
    world_size: int,
    start_time: float,
    activation_config: ActivationConfig = None,
    shard_data: bool = True,
) -> BaseActivationStore:
    """Create the appropriate activation store based on training config.

    Valid activation_source values:
    - "local_manifest": Use LocalActivationStore with local manifest/chunks.
    - "remote": Use RemoteActivationStore with remote server.

    Args:
        training_config: The training configuration object.
        clt_config: The CLT model configuration (currently unused here after removing generate).
        device: The torch device to use.
        rank: The distributed rank.
        world_size: The distributed world size.
        start_time: The training start time for elapsed time logging (unused if generate is gone).
        shard_data: Whether to include shard data in the store.

    Returns:
        Configured instance of a BaseActivationStore subclass.
    """
    activation_source = training_config.activation_source
    sampling_strategy = training_config.sampling_strategy

    store: BaseActivationStore

    if activation_source == "local_manifest":
        logger.info(f"Rank {rank}: Using LocalActivationStore (reading local manifest/chunks).")
        if not training_config.activation_path:
            raise ValueError(
                "activation_path must be set in TrainingConfig when activation_source is 'local_manifest'."
            )
        store = LocalActivationStore(
            dataset_path=training_config.activation_path,
            train_batch_size_tokens=training_config.train_batch_size_tokens,
            device=device,
            dtype=training_config.activation_dtype,
            rank=rank,
            world=world_size,
            seed=training_config.seed,
            sampling_strategy=sampling_strategy,
            normalization_method=training_config.normalization_method,
            shard_data=shard_data,
        )
        if isinstance(store, LocalActivationStore):
            logger.info(f"Rank {rank}: Initialized LocalActivationStore from path: {store.dataset_path}")
            if store.apply_normalization:
                logger.info(f"Rank {rank}:   Normalization ENABLED using loaded norm_stats.json.")
            else:
                logger.warning(f"Rank {rank}:   Normalization DISABLED (processing failed or file incomplete/invalid).")
    elif activation_source == "remote":
        logger.info(f"Rank {rank}: Using RemoteActivationStore (remote slice server).")
        remote_cfg = training_config.remote_config
        if remote_cfg is None:
            raise ValueError("remote_config dict must be set in TrainingConfig when activation_source is 'remote'.")
        server_url = remote_cfg.get("server_url")
        dataset_id = remote_cfg.get("dataset_id")
        if not server_url or not dataset_id:
            raise ValueError("remote_config must contain 'server_url' and 'dataset_id'.")

        store = RemoteActivationStore(
            server_url=server_url,
            dataset_id=dataset_id,
            train_batch_size_tokens=training_config.train_batch_size_tokens,
            device=device,
            dtype=training_config.activation_dtype,
            rank=rank,
            world=world_size,
            seed=training_config.seed,
            timeout=remote_cfg.get("timeout", 60),
            sampling_strategy=sampling_strategy,
            normalization_method=training_config.normalization_method,
            shard_data=shard_data,
        )
        if isinstance(store, RemoteActivationStore):
            logger.info(f"Rank {rank}: Initialized RemoteActivationStore for dataset: {store.did_raw}")
            if store.apply_normalization:
                logger.info(f"Rank {rank}:   Normalization ENABLED using fetched norm_stats.json.")
            else:
                logger.warning(
                    f"Rank {rank}:   Normalization DISABLED (norm_stats.json not found on server or failed to load)."
                )
    elif activation_source == "streaming":
        logger.info(f"Rank {rank}: Using StreamingActivationStore.")

        cfg = activation_config
        extractor = None
        if rank == 0:
            extractor = ActivationExtractorCLT(
                model_name=cfg.model_name,
                mlp_input_module_path_template=cfg.mlp_input_module_path_template,
                mlp_output_module_path_template=cfg.mlp_output_module_path_template,
                device=device,
                model_dtype=cfg.model_dtype,
                context_size=cfg.context_size,
                inference_batch_size=cfg.inference_batch_size,
                exclude_special_tokens=cfg.exclude_special_tokens,
                prepend_bos=cfg.prepend_bos,
                nnsight_tracer_kwargs=cfg.nnsight_tracer_kwargs,
                nnsight_invoker_args=cfg.nnsight_invoker_args
            )
        store = StreamingActivationStore(
            activation_cfg=activation_config,
            activation_extractor=extractor,
            train_batch_size_tokens=training_config.train_batch_size_tokens,
            device=device,
            dtype=training_config.activation_dtype,
            rank=rank,
            world=world_size,
            seed=training_config.seed,
            sampling_strategy=sampling_strategy,
            normalization_method=training_config.normalization_method,
            shard_data=shard_data,
        )
        
    else:
        raise ValueError(f"Unknown activation_source: {activation_source}. Valid options: 'local_manifest', 'remote'.")
    return store
