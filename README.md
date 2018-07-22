
# Parameter search for TensorFlow

A prototype for parameter search using a TensorFlow Estimator. 

The current code only supports non-distributed training. If multiple GPUs are detected in the local 
machine the search is run in parallel by training a model on each GPU. Otherwise it falls back to 
sequential training on CPU.

The current code depends on Sklearn for generating parameter sets and for parallel execution.

To run the code execute:

	python main.py
