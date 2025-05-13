import os
import torch
import torch.distributed as dist
from clt.config import CLTConfig
from clt.models.clt import CrossLayerTranscoder

# Helper to scatter full parameter to local shard (simplified)
import math
import torch.nn.functional as F


def _scatter(full_param: torch.Tensor, model_param: torch.nn.Parameter, partition_dim: int):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    full_dim_size = full_param.size(partition_dim)
    local_dim_padded = math.ceil(full_dim_size / world_size)
    start_index = rank * local_dim_padded
    local_dim_actual = model_param.size(partition_dim)
    end_index = min(start_index + local_dim_actual, full_dim_size)

    if end_index <= start_index:
        return  # nothing to copy for this rank

    indices = [slice(None)] * full_param.dim()
    indices[partition_dim] = slice(start_index, end_index)
    shard = full_param[tuple(indices)].clone()
    pad_amount = local_dim_actual - shard.size(partition_dim)
    if pad_amount > 0:
        pad_dims = [0, 0] * model_param.dim()
        pad_idx = model_param.dim() - 1 - partition_dim
        pad_dims[2 * pad_idx + 1] = pad_amount
        shard = F.pad(shard, tuple(pad_dims))
    model_param.data.copy_(shard.to(model_param.device, model_param.dtype))


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device("cpu")
    cfg = CLTConfig(d_model=32, num_features=64, num_layers=4, activation_fn="jumprelu", jumprelu_threshold=0.01)
    # Single model (no process_group)
    single_model = CrossLayerTranscoder(cfg, process_group=None, device=device)
    # Multi model with TP group
    pg = dist.group.WORLD
    multi_model = CrossLayerTranscoder(cfg, process_group=pg, device=device)
    # copy weights
    single_params = dict(single_model.named_parameters())
    # Copy parameters respecting sharding
    for name, p in multi_model.named_parameters():
        if name not in single_params:
            continue
        sp = single_params[name]
        if "encoders." in name and ".weight" in name:
            _scatter(sp.data, p, partition_dim=0)
        elif "encoders." in name and ".bias" in name:
            _scatter(sp.data, p, partition_dim=0)
        elif "decoders." in name and ".weight" in name:
            _scatter(sp.data, p, partition_dim=1)
        elif "decoders." in name and ".bias" in name:
            p.data.copy_(sp.data)
        elif name == "log_threshold":
            p.data.copy_(sp.data)
    dist.barrier()
    torch.manual_seed(0)
    input_tensor = torch.randn(64, 32)
    # broadcast
    dist.broadcast(input_tensor, src=0)

    for layer in range(4):
        with torch.no_grad():
            sp = single_model.get_preactivations(input_tensor.clone(), layer)
            mp = multi_model.get_preactivations(input_tensor.clone(), layer)
            if rank == 0:
                print("pre", layer, torch.allclose(sp, mp, atol=1e-6))
            s = single_model.encode(input_tensor.clone(), layer)
            m = multi_model.encode(input_tensor.clone(), layer)
            if rank == 0:
                print("enc", layer, torch.allclose(s, m, atol=1e-6))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size, join=True)
