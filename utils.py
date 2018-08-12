import sys
import shutil
import time
import inspect
import datetime
import random

def filter_params(fn, params):
    """Filters `params` and returns those in `fn`'s arguments.
    # Arguments
        fn : arbitrary function
    # Returns
        res : dictionary containing variables
            in both `sk_params` and `fn`'s arguments.
    """
    res = {}
    for name, value in params.items():
        if has_arg(fn, name):
            res.update({name: value})
    return res


def has_arg(fn, name, accept_all=False):
    """Checks if a callable accepts a given keyword argument.
    For Python 2, checks if there is an argument with the given name.
    For Python 3, checks if there is an argument with the given name, and
    also whether this argument can be called with a keyword (i.e. if it is
    not a positional-only argument).
    # Arguments
        fn: Callable to inspect.
        name: Check if `fn` can be called with `name` as a keyword argument.
        accept_all: What to return if there is no parameter called `name`
                    but the function accepts a `**kwargs` argument.
    # Returns
        bool, whether `fn` accepts a `name` keyword argument.
    """
    if sys.version_info < (3,):
        arg_spec = inspect.getargspec(fn)
        if accept_all and arg_spec.keywords is not None:
            return True
        return (name in arg_spec.args)
    elif sys.version_info < (3, 3):
        arg_spec = inspect.getfullargspec(fn)
        if accept_all and arg_spec.varkw is not None:
            return True
        return (name in arg_spec.args or
                name in arg_spec.kwonlyargs)
    else:
        signature = inspect.signature(fn)
        parameter = signature.parameters.get(name)
        if parameter is None:
            if accept_all:
                for param in signature.parameters.values():
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        return True
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))

def ts_rand():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    random_num = random.randint(1e6, 1e7-1)
    return '%s_%d' % (ts, random_num)

def dict_to_str(d):
    return ','.join(str(k)+'='+str(v) for k, v in d.items())

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def delete_dir(path):
    shutil.rmtree(path, ignore_errors=True)

def current_time_ms():
    return int(time.time()*1000.0)
    