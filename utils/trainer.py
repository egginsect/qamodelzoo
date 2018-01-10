import os
from glob import glob
import multiprocessing
import tensorflow as tf
from utils import TFRecordManager


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()
        self.load_dataset()

    def decode_dataset(self, fns, decode_func):
        dataset = tf.data.TFRecordDataset(fns)
        dataset = dataset.map(decode_func, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.padded_batch(self.config.batch_size, padded_shapes=dataset.output_shapes)
        data_iterator = dataset.make_one_shot_iterator()
        return data_iterator

    def load_dataset(self):
        fns = TFRecordManager.load_tfrecords(self.config.data_dir)
        train_valid_split = len(fns)*.8
        train_fns, valid_fns = fns[:train_valid_split], fns[train_valid_split:]
        decode_func = TFRecordManager.tfrecord_decoder(glob('data/processed/decode_file.json')[0])
        train_iterator = self.decode_dataset(train_fns, decode_func)
        validation_iterator = self.decode_dataset(valid_fns, decode_func)
        self.train_handle = self.sess.run(train_iterator.string_handle())
        self.valid_handle = self.sess.run(validation_iterator.string_handle())
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types)
        self.input_data = iterator.get_next()
