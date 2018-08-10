import os
import sys
import time
import socket
import numpy as np
from pyspark.sql import SparkSession

def _do_search(map_fn):
    def _wrapper_fun(iter):
        return [map_fn(**x) for x in iter]
    return _wrapper_fun

def mnist(alpha, dropout):
    print('START TRAIN', os.getpid(), socket.gethostname())
    time.sleep(5.0)
    print('FINISH TRAIN', alpha, dropout)
    return alpha + dropout

if __name__ == "__main__":
    
    spark_session = SparkSession\
        .builder\
        .appName("TensorFlowOnSpark")\
        .getOrCreate()

    param_sets = [
        {'alpha': 0.01, 'dropout': 0.1},
        {'alpha': 0.02, 'dropout': 0.2},
        {'alpha': 0.03, 'dropout': 0.3},
        {'alpha': 0.04, 'dropout': 0.4},
        {'alpha': 0.05, 'dropout': 0.5},
        {'alpha': 0.06, 'dropout': 0.6},
        {'alpha': 0.07, 'dropout': 0.7},
        {'alpha': 0.08, 'dropout': 0.8},
        {'alpha': 0.09, 'dropout': 0.9},
        {'alpha': 0.10, 'dropout': 1.0}
    ]

    sc = spark_session.sparkContext
    num_partitions = len(param_sets)
    param_rdd = sc.parallelize(param_sets, num_partitions)
    result = param_rdd.mapPartitions(_do_search(mnist)).collect()
    print(result)
    best_index = np.array(result).argmin()
    print('Best index: %d' % best_index)
    print('Best params: %s' % param_sets[best_index])

    spark_session.stop()
