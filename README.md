
# Parameter search for TensorFlow Estimators

This is a prototype for hyperparameter search using the TensorFlow Estimator API. Currently 
random and grid parameter search is supported, next I'll look into Hyperopt integration. The 
implementation is based on Spark. Spark only handles task coordination and failover. The training
data is transfered using the TensorFlow Dataset API.

There are still lots of open questions/problems:

 * the assignment from Spark executors to GPUs
 * model saving on HDFS or S3 does not work currently
 * currently only single GPU per model training is supported
 * distributed TF would require a different approach
 * should cross-fold validation be supported?
 * how to integrate Hyperopt?

The current test environment is an AWS EMR cluster with single GPU workers. Spark will be 
configured with one single core executor per worker node.

Example invocation:

    spark-submit --master yarn \
        --deploy-mode cluster \
        --driver-memory 6g \
        --executor-memory 6g \
        --num-executors 1 \
        --executor-cores 1 \
        --conf spark.yarn.maxAppAttempts=1 \
        --py-files mnist_dataset.py,param_search_on_spark.py,utils.py,gpu_info.py \
        main.py

`spark.yarn.maxAppAttempts=1` is better for development because in case of errors the jobs terminates
earlier. If not in development go with the defaults. 

Configuration issues for AWS EMR:
 * Use EMR 5.10.0 as this installs CUDA 9.0 and cuDNN 7.0.3 (for TF 1.9), or bring your own AMI
 * Disable Spark dynamic allocation
 * Disable YARN Nodemanager vmem checks because the Python sub-process with TF allocates a huge amount of virtual memory and YARN kills the executor
 * Enable NVIDIA driver persistent mode, otherwise the `nvidia-smi` output is confusing
