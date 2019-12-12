import numpy as np

# __all__ = ['rle_decode', 'rle_encode', 'rle_to_string']


def rle_encode(mask):
    """
    Convert mask to EncodedPixels in run-length encoding
    from https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines
    Args: 
        mask(np.array): mask to be converted
    """
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    """Array to str"""
    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape, fill_value=1, dtype=int, relative=False):
    """
    Args:
        rle_str (str): rle string
        shape (Tuple[int, int]): shape of the output mask
        relative: if True, rle_str is relative encoded string
    """
    s = rle_str.strip().split(" ")
    starts, lengths = np.array([np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])])
    mask = np.zeros(np.prod(shape), dtype=dtype)
    if relative:
        start = 0
        for index, length in zip(starts, lengths):
            start = start + index
            end = start + length
            mask[start:end] = fill_value
            start = end
        return mask.reshape(shape[::-1]).T
    else:
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = fill_value
        return mask.reshape(shape[::-1]).T
