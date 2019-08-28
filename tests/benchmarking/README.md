# Speed comparison of different models and their memory usage

TODO: add

## Methodology

Check `benchmarking_example.py` for an example of how to measure speed and memory usage. NOTE: Memory usage reported by pytorch doesn't include memory used by internal CUDA libraries which also has to be loaded to device. It means that more memory is actually used than reported by this tests. But the number of additional memory is always constant for the same device.
