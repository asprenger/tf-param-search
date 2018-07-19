import os
import time
import random
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, ParameterSampler
import tensorflow as tf
from utils import ts_rand, current_time_ms

class BaseParamSearch(object):

    def __init__(self, model_fn, train_input_fn, eval_input_fn, model_base_dir, n_jobs, train_hooks):
        self.model_fn = model_fn
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn
        self.n_jobs = n_jobs
        self.joblib_backend = 'threading'
        self.joblib_verbose = 0
        self.model_base_dir = model_base_dir
        self.train_hooks = train_hooks
        
    def _train_and_eval(self, params, model_dir):
        model_dir = os.path.join(model_dir, ts_rand())
        estimator = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=model_dir, params=params)
        start = current_time_ms()
        estimator.train(input_fn=self.train_input_fn, hooks=self.train_hooks) 
        train_time = current_time_ms() - start
        start = current_time_ms()
        eval_results = estimator.evaluate(input_fn=self.eval_input_fn)
        eval_time = current_time_ms() - start
        eval_score = eval_results['loss']
        return (eval_score, model_dir, eval_results, train_time, eval_time)

    def search(self):
        model_dir = os.path.join(self.model_base_dir, ts_rand())
        candidate_params = list(self.get_param_iterator())
        out = Parallel(n_jobs=self.n_jobs, verbose=self.joblib_verbose, backend=self.joblib_backend)(
            delayed(self._train_and_eval, check_pickle=False)(parameters, model_dir) 
                for parameters in candidate_params)
        (eval_scores, model_dirs, eval_results, train_times, eval_times) = zip(*out)
        best_index = np.array(eval_scores).argmin()
        best_score = eval_scores[best_index]
        best_params = candidate_params[best_index]
        best_model_dir = model_dirs[best_index]
        best_eval_result = eval_results[best_index]
        return best_params, best_score, best_model_dir, best_eval_result


class GridParamSearch(BaseParamSearch):
    def __init__(self, model_fn, train_input_fn, eval_input_fn, param_grid, model_base_dir, n_jobs=1, train_hooks=[]):
        super(GridParamSearch, self).__init__(model_fn=model_fn, train_input_fn=train_input_fn, 
            eval_input_fn=eval_input_fn, n_jobs=n_jobs, train_hooks=train_hooks, model_base_dir=model_base_dir)
        self.param_grid = param_grid

    def get_param_iterator(self):
        """Return iterator over parameter sets"""
        return ParameterGrid(self.param_grid)


class RandomParamSearch(BaseParamSearch):

    def __init__(self, model_fn, train_input_fn, eval_input_fn, param_distributions, model_base_dir, n_iter=10, n_jobs=1, train_hooks=[]):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        super(RandomParamSearch, self).__init__(model_fn=model_fn, train_input_fn=train_input_fn, 
            eval_input_fn=eval_input_fn, n_jobs=n_jobs, train_hooks=train_hooks, model_base_dir=model_base_dir)

    def get_param_iterator(self):
        """Return iterator over parameter sets"""
        return ParameterSampler(self.param_distributions, self.n_iter)

