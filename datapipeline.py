import numpy as np
import tensorflow as tf
import glob
from tensorflow.data import Iterator

class Dataset:
    def __init__(self):
        self.loaddata()

    def parse_function(self,example_proto):
        """
        helper function to get image stack(4 frames), action and next frame from tf records files
        """
        feature_set = { 'a_t': tf.FixedLenFeature([], tf.int64),
                        's_t': tf.FixedLenFeature([], tf.string),
                        'x_t_1': tf.FixedLenFeature([], tf.string)
                      }
        features = tf.parse_single_example(example_proto, features=feature_set)
        s_t = tf.decode_raw(features['s_t'], tf.uint8)
        x_t_1 = tf.decode_raw(features['x_t_1'], tf.uint8)
        s_t = tf.reshape(s_t, [84, 84, 12])
        x_t_1 = tf.reshape(x_t_1, [84, 84, 3])
        s_t = tf.cast(s_t, tf.float32)
        x_t_1 = tf.cast(x_t_1, tf.float32)
        a_t = tf.cast(features['a_t'], tf.int32)
        a_t = tf.one_hot(a_t, 9)
        return s_t, a_t, x_t_1

    def loaddata(self):
        """
        loads data queue pipelines for train,test,val which can be switched during run time.
        """
        #change image to float
        scale = (1.0 / 255.0)
        #mean centering
        mean = np.load("DataCollection/Pong-v0/mean.npy")
        # making multi threaded pipelines from tfrecords using feedable iterators to switch queues during run time
        filenames = glob.glob('DataCollection/Pong-v0/'+'train'+'/*.tfrecords')
        datasettrain = tf.data.TFRecordDataset(filenames)
        filenames = glob.glob('DataCollection/Pong-v0/' + 'val' + '/*.tfrecords')
        datasetval = tf.data.TFRecordDataset(filenames)
        filenames = glob.glob('DataCollection/Pong-v0/' + 'test' + '/*.tfrecords')
        datasettest = tf.data.TFRecordDataset(filenames)
        #restore image and action data and shape
        datasettrain = datasettrain.map(map_func= self.parse_function, num_parallel_calls= 4)
        #to iterate over dataset for multiple epochs
        datasettrain = datasettrain.repeat()
        #shuffling and creating a queue buffer
        datasettrain = datasettrain.shuffle(buffer_size=1000)  # specify queue size buffer
        #making batches
        datasettrain = datasettrain.batch(32)
        #string handle for train
        iter_train_handle = datasettrain.make_one_shot_iterator().string_handle()

        datasetval = datasetval.map(map_func=self.parse_function, num_parallel_calls=4)
        datasetval = datasetval.repeat()
        datasetval = datasetval.shuffle(buffer_size=1000)  # specify queue size buffer
        datasetval = datasetval.batch(32)
        # string handle for val
        iter_val_handle = datasetval.make_one_shot_iterator().string_handle()

        datasettest = datasettest.map(map_func=self.parse_function, num_parallel_calls=4)
        datasettest = datasettest.repeat()
        datasettest = datasettest.shuffle(buffer_size=1000)  # specify queue size buffer
        datasettest = datasettest.batch(32)
        # string handle for test
        iter_test_handle = datasettest.make_one_shot_iterator().string_handle()
        #handle to indicate which queue we want to switch to or iterate over
        handle = tf.placeholder(tf.string, shape=[])
        iterator = Iterator.from_string_handle(handle, datasettrain.output_types, datasettrain.output_shapes)
        #get data tensors
        s_t_batch, a_t_batch, x_t_1_batch = iterator.get_next()
        mean_const = tf.constant(mean, dtype=tf.float32)
        #mean centering and convert to float
        s_t_batch = (s_t_batch - tf.tile(mean_const, [1, 1, 4])) * scale
        x_t_1_batch = (x_t_1_batch - mean_const) * scale
        self.s_t_batch = s_t_batch
        self.a_t_batch = a_t_batch
        self.x_t_1_batch = x_t_1_batch
        self.iter_train_handle = iter_train_handle
        self.iter_val_handle = iter_val_handle
        self.iter_test_handle = iter_test_handle
        self.handle = handle

    def __call__(self):
        return self.s_t_batch, self.a_t_batch, self.x_t_1_batch, self.iter_train_handle, self.iter_val_handle, self.handle, self.iter_test_handle