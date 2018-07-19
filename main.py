
import logging
from scipy.stats.distributions import expon
import tensorflow as tf
import tensorflow.contrib.layers as layers
from param_search import GridParamSearch, RandomParamSearch
import dataset


tf.logging.set_verbosity(tf.logging.INFO)
tf_logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for handler in tf_logger.handlers: handler.setFormatter(formatter)


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


def model_fn(features, labels, mode, params, config):
    '''Model function for Estimator.'''

    image = features
    if isinstance(features, dict):
        image = features['X'] # used if input is read from Numpy arrays
    
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

        # Name tensors for with tf.train.LoggingTensorHook
        tf.identity(params['learning_rate'], 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(acc_op, name='train_accuracy')

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


def main():

    data_dir = '/tmp/mnist'
    model_base_dir = '/tmp/param_search'
    batch_size = 128

    def train_input_fn():
        ds = dataset.train(data_dir)
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=50000)
        ds = ds.batch(batch_size)
        ds = ds.repeat(1)
        return ds      

    def eval_input_fn():
        ds = dataset.test(data_dir)
        ds = ds.batch(batch_size)
        return ds

    train_hooks = [tf.train.LoggingTensorHook(tensors=['learning_rate', 'cross_entropy', 'train_accuracy'], every_n_iter=20)]    

    #param_grid = {'hidden_size': [512], 'keep_rate': [0.5], 'learning_rate': [1e-4]}
    #param_search = GridParamSearch(model_fn, train_input_fn, eval_input_fn, param_grid, model_base_dir, n_jobs=1, train_hooks=train_hooks)

    param_distributions = {'hidden_size': [512], 'keep_rate': [0.5], 'learning_rate': expon()}
    param_search = RandomParamSearch(model_fn, train_input_fn, eval_input_fn, param_distributions, 
                                     model_base_dir, n_iter=1, n_jobs=1, train_hooks=train_hooks)

    best_params, best_score, best_model_dir, best_eval_result = param_search.search()
    print('Best score: %f' % best_score)
    print('Best parameters: %s' % str(best_params))
    print('Best model: %s' % best_model_dir)
    print('Best eval result: %s' % best_eval_result)


if __name__ == '__main__':
    main()
