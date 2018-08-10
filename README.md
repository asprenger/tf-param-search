
# Parameter search for TensorFlow Estimators

A prototype to parameter search using TensorFlow estimators. Currently random and grid parameter 
search is supported. The implementation is based on Spark. Spark only handles task coordination
and failover. The train and evaluation data is transfered by the TensorFlow dataset framework.

The current test setup is a Spark cluster with 2 workers. Each worker has a GPU.

Example invocation:

    spark-submit --master yarn \
        --deploy-mode cluster \
        --driver-memory 6g \
        --executor-memory 6g \
        --num-executors 2 \
        --executor-cores 1 \
        --conf spark.yarn.maxAppAttempts=1 \
        --conf spark.executor.memoryOverhead=20000 \
        --py-files mnist_dataset.py,param_search_on_spark.py,utils.py \
        main.py