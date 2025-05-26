import datetime

import numpy as np
import torch


# Helper function to format elapsed time
def _format_elapsed_time(seconds: float) -> str:
    """Formats elapsed seconds into HH:MM:SS or MM:SS."""
    td = datetime.timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if td.days > 0 or hours > 0:
        return f"{td.days * 24 + hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def torch_bfloat16_to_numpy_uint16(x: torch.Tensor) -> np.ndarray:
    return np.frombuffer(x.float().numpy().tobytes(), dtype=np.uint16)[1::2].reshape(x.shape)
