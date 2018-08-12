import os
import time
import random
import socket
import logging
from collections import namedtuple
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
import tensorflow as tf
from utils import ts_rand, current_time_ms
from gpu_info import get_gpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s (%(threadName)s-%(process)d) %(message)s")
tf.logging.set_verbosity(tf.logging.INFO)

ParamSearchResult = namedtuple('ParamSearchResult', ['eval_score', 'eval_results', 'train_time', 'eval_time', 'params'])

def _run_search(model_fn, train_input_fn, eval_input_fn, train_hooks, eval_hooks):
    """Wraps the `_train_and_eval` function in a Spark mapPartitions function.
    
    Args:
        model_fn: Estimator model function.
        train_input_fn: A function that provides input data for training. The function must
            return a 'tf.data.Dataset' object.
        eval_input_fn: A function that provides input data for evaluation. The function must
            return a 'tf.data.Dataset' object.
        train_hooks: List of `SessionRunHook` subclass instances used during training.
        eval_hooks: List of `SessionRunHook` subclass instances used during evaluation.

    Returns:
        A nodeRDD.mapPartitions() function.
    """
    def _wrapper_fn(param_iter):
        return [_train_and_eval(model_fn, train_input_fn, eval_input_fn, params, train_hooks, eval_hooks) for params in param_iter]
    return _wrapper_fn

def _train_and_eval(model_fn, train_input_fn, eval_input_fn, params, train_hooks, eval_hooks):
    """
    Train and evaluate a model on a set of parameters.

    Args:
        model_fn: Estimator model function.
        train_input_fn: A function that provides input data for training. The function must
            return a 'tf.data.Dataset' object.
        eval_input_fn: A function that provides input data for evaluation. The function must
            return a 'tf.data.Dataset' object.
        params: A dict with fit and model parameters.
        train_hooks: List of `SessionRunHook` subclass instances used during training.
        eval_hooks: List of `SessionRunHook` subclass instances used during evaluation.

    Return:
        eval_score: The evaluation score
        eval_results: The return value of `Estimator.evaluate`
        train_time: Train time in milliseconds
        eval_time: Evaluation time in milliseconds
        params: The original parameters the model has been trained on
    """
    model_dir = os.path.join('/tmp/models', ts_rand())
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params)

    def wrapped_train_input_fn():
        # If `params` contains a batch size parameter we apply it to the 
        # dataset. We simply assume that is has not been batched yet.
        if 'batch_size' in params:
            return train_input_fn().batch(params['batch_size'])
        else:
            return train_input_fn()

    logging.info('Train model with: %s' % params)
    start = current_time_ms()
    estimator.train(input_fn=wrapped_train_input_fn, hooks=train_hooks)
    train_time = current_time_ms() - start

    logging.info('Evaluate model with: %s' % params)
    start = current_time_ms()
    eval_results = estimator.evaluate(input_fn=eval_input_fn, hooks=eval_hooks)
    eval_time = current_time_ms() - start
    eval_score = eval_results['loss']
    return (eval_score, eval_results, train_time, eval_time, params)

class BaseParamSearch(object):
    """Base class for parameter search.

    Args:
        model_fn: Estimator model function.
        train_input_fn: A function that provides input data for training. The function must
            return a 'tf.data.Dataset' object.
        eval_input_fn: A function that provides input data for evaluation. The function must
            return a 'tf.data.Dataset' object.
        train_hooks: List of `SessionRunHook` subclass instances used during training.
        eval_hooks: List of `SessionRunHook` subclass instances used during evaluation.
    """    

    def __init__(self, model_fn, train_input_fn, eval_input_fn, train_hooks, eval_hooks):
        self.model_fn = model_fn
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn
        self.train_hooks = train_hooks
        self.eval_hooks = eval_hooks
        self.search_results = None
  

    def search(self, sc):
        """
        Run a parameter search.
        """
        
        num_executors = int(sc._conf.get('spark.executor.instances'))
        executor_idx_rdd = sc.parallelize(list(range(num_executors)), num_executors)
        def get_gpu_info(executor_idx):
            return [get_gpus()]
        gpu_infos = executor_idx_rdd.mapPartitions(get_gpu_info).collect()
        print('GPU INFOS:', gpu_infos)


        param_sets = list(self._get_param_iterator())
        num_partitions = len(param_sets)

        # Put each parameter set in a separate partition
        param_rdd = sc.parallelize(param_sets, num_partitions)
        results = param_rdd.mapPartitions(_run_search(self.model_fn, self.train_input_fn, self.eval_input_fn, self.train_hooks, self.eval_hooks)).collect()

        search_results = [ParamSearchResult(*result) for result in results]
        self.search_results = sorted(search_results, key=lambda x: x.eval_score)

        (eval_scores, eval_results, train_times, eval_times, params) = zip(*results)
        best_index = np.array(eval_scores).argmin()
        best_score = eval_scores[best_index]
        best_params = param_sets[best_index]
        best_eval_result = eval_results[best_index]

        return best_params, best_score, best_eval_result


class GridParamSearch(BaseParamSearch):
    """Find Estimator parameters by exhaustive search over all parameter value combinations.

    Args:
        model_fn: Estimator model function.
        train_input_fn: A function that provides input data for training. The function must
            return a 'tf.data.Dataset' object.
        eval_input_fn: A function that provides input data for evaluation. The function must
            return a 'tf.data.Dataset' object.
        param_grid : A `dict` of string to sequence, or sequence of such
            The parameter grid to explore, as a dictionary mapping estimator parameters to 
            sequences of allowed values. A sequence of dicts signifies a sequence of grids 
            to search, and is useful to avoid exploring parameter combinations that make no 
            sense.
        train_hooks: List of `SessionRunHook` subclass instances used during training.
        eval_hooks: List of `SessionRunHook` subclass instances used during evaluation.
    """    
    def __init__(self, model_fn, train_input_fn, eval_input_fn, param_grid, train_hooks=[], eval_hooks=[]):
        super(GridParamSearch, self).__init__(model_fn=model_fn, train_input_fn=train_input_fn, eval_input_fn=eval_input_fn,
                                              train_hooks=train_hooks, eval_hooks=eval_hooks)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Return iterator over parameter sets"""
        return ParameterGrid(self.param_grid)


class RandomParamSearch(BaseParamSearch):
    """
    Find Estimator parameters by sampling a given number of candidates from the parameter space.

    Args:
        model_fn: Estimator model function.
        train_input_fn: A function that provides input data for training. The function must
            return a 'tf.data.Dataset' object.
        eval_input_fn: A function that provides input data for evaluation. The function must
            return a 'tf.data.Dataset' object.
        param_distributions : Dictionary where the keys are parameters and values are distributions 
            from which a parameter is to be sampled. Distributions either have to provide a ``rvs`` 
            function to sample from them, or can be given as a list of values.
        n_iter : An `integer`. Number of parameter settings that are produced.
        train_hooks: List of `SessionRunHook` subclass instances used during training.
        eval_hooks: List of `SessionRunHook` subclass instances used during evaluation.
    """
    def __init__(self, model_fn, train_input_fn, eval_input_fn, param_distributions, n_iter, 
                 train_hooks=[], eval_hooks=[]):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        super(RandomParamSearch, self).__init__(model_fn=model_fn, train_input_fn=train_input_fn, eval_input_fn=eval_input_fn,
                                                train_hooks=train_hooks, eval_hooks=eval_hooks)

    def _get_param_iterator(self):
        """Return iterator over parameter sets"""
        return ParameterSampler(self.param_distributions, self.n_iter)
