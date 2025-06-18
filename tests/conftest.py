import pytest
import torch


def get_available_devices():
    """Returns available devices, including cpu, mps, and cuda if available."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = get_available_devices()


@pytest.fixture(params=DEVICES)
def device(request):
    """Fixture to iterate over all available devices."""
    return torch.device(request.param)
