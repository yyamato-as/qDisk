import numpy as np

def measure_noise_level(data, mask, axis=None, rms_or_std="rms"):
    
    data *= mask

    if rms_or_std == "rms":
        return np.sqrt(np.nanmean(data**2, axis=axis, dtype=np.float64))
    else:
        return np.nanstd(data, axis=axis, dtype=np.float64)

