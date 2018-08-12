
import logging

import os
import sys
import time
import numpy as np
from collections import namedtuple

from pyspark.sql import SparkSession

import tensorflow as tf
import tensorflow.contrib.layers as layers
import mnist_dataset
from utils import delete_dir, ts_rand, current_time_ms

from scipy.stats.distributions import expon
from param_search_on_spark import GridParamSearch, RandomParamSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s (%(threadName)s-%(process)d) %(message)s")
tf.logging.set_verbosity(tf.logging.INFO)

def build_model(x, hidden_size, keep_prob):
    print('BUILD MODEL(x=%s, hidden_size=%d, keep_prob=%f)' % (x.shape, hidden_size, keep_prob))
    with tf.variable_scope("model"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        conv1 = layers.convolution2d(x_image,
                    num_outputs=32,
                    kernel_size=5,
                    stride=1,
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    scope='conv1')
        pool1 = layers.max_pool2d(
            inputs=conv1,
            kernel_size=2,
            stride=2,
            padding='SAME',
            scope='pool1')
        conv2 = layers.convolution2d(pool1,
                    num_outputs=64,
                    kernel_size=5,
                    stride=1,
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    scope='conv2')
        pool2 = layers.max_pool2d(
            inputs=conv2,
            kernel_size=2,
            stride=2,
            padding='SAME',
            scope='pool2')
        flattened = layers.flatten(pool2)
        fc1 = layers.fully_connected(flattened, 
            hidden_size, 
            activation_fn=tf.nn.relu, 
            scope='fc1')
        drop1 = layers.dropout(
            fc1,
            keep_prob=keep_prob,
            scope='drop1')
        logits = layers.fully_connected(drop1, 
            10, 
            activation_fn=None, 
            scope='fc2')
        return logits

def train_input_fn():
    ds = mnist_dataset.train('/tmp/mnist')
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=50000)
    ds = ds.repeat(5)
    # Do not call `ds.batch(batch_size)` here if `batch_size` is a 
    # hyperparameter, this must be handled in `BaseParamSearch`.
    return ds

def eval_input_fn():
    ds = mnist_dataset.test('/tmp/mnist')
    ds = ds.batch(32)
    return ds

def model_fn(features, labels, mode, params, config):
    '''Model function for Estimator.'''

    image = features
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = build_model(image, params['hidden_size'], 1.0)
        predictions = { 
            "class": tf.argmax(logits, axis=1, output_type=tf.int32),
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = build_model(image, params['hidden_size'], params['keep_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]) 
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step()) 
        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = build_model(image, params['hidden_size'], 1.0)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
        eval_metric_ops = { "accuracy": (acc, acc_op) }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":

    spark_session = SparkSession\
        .builder\
        .appName("ParamSearchOnSpark")\
        .getOrCreate()

    sc = spark_session.sparkContext
    num_executors = int(sc._conf.get("spark.executor.instances"))

    param_grid = {'hidden_size': [256, 512], 'keep_rate': [0.5], 'learning_rate': [1e-4], 'batch_size': [128]}
    param_search = GridParamSearch(model_fn, train_input_fn, eval_input_fn, param_grid)
    best_params, best_score, best_eval_result = param_search.search(sc)

    logging.info('Best score: %f' % best_score)
    logging.info('Best eval result: %s' % best_eval_result)
    logging.info('Best params: %s' % best_params)
    for i, result in enumerate(param_search.search_results):
        logging.info('%d\t%f\t%f\t%s' % (i, result.eval_results['loss'], result.eval_results['accuracy'], result.params))

    spark_session.stop()
