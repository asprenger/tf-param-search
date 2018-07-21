import os
import time
import random
import numpy as np
from collections import namedtuple
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, ParameterSampler
import tensorflow as tf
from tensorflow.python.client import device_lib
from utils import ts_rand, current_time_ms

ParamSearchResult = namedtuple('ParamSearchResult', ['eval_score', 'model_dir', 'eval_results', 'train_time', 'eval_time', 'params'])

class BaseParamSearch(object):

    def __init__(self, model_fn, train_input_fn, eval_input_fn, model_base_dir, train_hooks, eval_hooks, run_config):
        self.model_fn = model_fn
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn
        self.joblib_backend = 'threading' # 'threading' or 'multiprocessing'
        self.joblib_verbose = 0
        self.model_base_dir = model_base_dir
        self.train_hooks = train_hooks
        self.eval_hooks = eval_hooks
        self.run_config = run_config
        self.report = None
        
    def _get_available_gpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def _get_available_cpus(self):
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'CPU']

    def _get_devices(self):
        devices = self._get_available_gpus()
        if len(devices) == 0:
            devices = self._get_available_cpus()
        return devices    
    
    def _train_and_eval(self, params, model_dir):
        model_dir = os.path.join(model_dir, ts_rand())
        estimator = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=model_dir, 
                                           params=params, config=self.run_config)

        def train_input_fn():
            if 'batch_size' in params:
                return self.train_input_fn().batch(params['batch_size'])
            else:
                return self.train_input_fn()

        start = current_time_ms()
        estimator.train(input_fn=train_input_fn, hooks=self.train_hooks) 
        train_time = current_time_ms() - start
        start = current_time_ms()
        eval_results = estimator.evaluate(input_fn=self.eval_input_fn, hooks=self.eval_hooks)
        eval_time = current_time_ms() - start
        eval_score = eval_results['loss']
        return (eval_score, model_dir, eval_results, train_time, eval_time, params)

    def search(self):
        candidate_params = list(self._get_param_iterator())
        model_dir = os.path.join(self.model_base_dir, ts_rand())
        
        devices = self._get_devices()
        num_devices = len(devices)
        for i, params in enumerate(candidate_params):
            params['_idx'] = i
            params['_device'] = devices[i % num_devices]

        out = Parallel(n_jobs=num_devices, verbose=self.joblib_verbose, backend=self.joblib_backend)(
            delayed(self._train_and_eval, check_pickle=False)(parameters, model_dir) 
                for parameters in candidate_params)

        search_results = [ParamSearchResult(*result) for result in out]
        self.search_results = sorted(search_results, key=lambda x: x.eval_score)

        (eval_scores, model_dirs, eval_results, train_times, eval_times, params) = zip(*out)
        best_index = np.array(eval_scores).argmin()
        best_score = eval_scores[best_index]
        best_params = candidate_params[best_index]
        best_model_dir = model_dirs[best_index]
        best_eval_result = eval_results[best_index]
        return best_params, best_score, best_model_dir, best_eval_result


class GridParamSearch(BaseParamSearch):
    """
     Exhaustive search over all parameter value combinations for an Estimator.
    """
    def __init__(self, model_fn, train_input_fn, eval_input_fn, param_grid, model_base_dir, train_hooks=[], 
                 eval_hooks=[], run_config=None):
        super(GridParamSearch, self).__init__(model_fn=model_fn, train_input_fn=train_input_fn, eval_input_fn=eval_input_fn, 
              train_hooks=train_hooks, eval_hooks=eval_hooks, model_base_dir=model_base_dir, run_config=run_config)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Return iterator over parameter sets"""
        return ParameterGrid(self.param_grid)


class RandomParamSearch(BaseParamSearch):
    """
     Sample a given number of candidates from the parameter space.
    """
    def __init__(self, model_fn, train_input_fn, eval_input_fn, param_distributions, model_base_dir, n_iter, 
                 train_hooks=[], eval_hooks=[], run_config=None):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        super(RandomParamSearch, self).__init__(model_fn=model_fn, train_input_fn=train_input_fn, eval_input_fn=eval_input_fn, 
              train_hooks=train_hooks, eval_hooks=eval_hooks, model_base_dir=model_base_dir, run_config=run_config)

    def _get_param_iterator(self):
        """Return iterator over parameter sets"""
        return ParameterSampler(self.param_distributions, self.n_iter)
