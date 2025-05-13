import torch


def mark_replicated(param: torch.nn.Parameter):
    """Tag a Parameter that is fully replicated across TP ranks.

    Args:
        param: The torch.nn.Parameter to mark.
    """
    # setattr(param, "_is_replicated", True) # Alternative
    param.__dict__["_is_replicated"] = True


def is_replicated(param: torch.nn.Parameter) -> bool:
    """Check if a Parameter is marked as replicated.

    Args:
        param: The torch.nn.Parameter to check.

    Returns:
        True if the parameter is marked as replicated, False otherwise.
    """
    return getattr(param, "_is_replicated", False)
