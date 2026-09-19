import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    # Write code here
    n = scores.shape[-1]
    result = np.array(scores)
    rows,cols = np.triu_indices(n=n, k=1)
    result[...,rows,cols]=mask_value
    return result