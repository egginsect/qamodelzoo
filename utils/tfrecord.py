import collections
import os
import tensorflow as tf
from misc import flatten, unflatten
import glob
class TFRecordManager(object):
    encode_methods = flatten({
        'int': lambda value: tf.train.Feature(int_list=tf.train.Int64List(value=[value])),
        'float': lambda value: tf.train.Feature(int_list=tf.train.FloatList(value=[value])),
        'list':{
            'int': lambda value: tf.train.Feature(int_list=tf.train.Int64List(value=value)),
            'float': lambda value: tf.train.Feature(int_list=tf.train.FloatList(value=value)),
        }
    })
    decode_methods = flatten({
        'int': lambda value: tf.FixedLenFeature([], tf.int64),
        'float': lambda value: tf.FixedLenFeature([], tf.float32),
        'list': {
            'int': lambda: tf.FixedLenSequenceFeature([], tf.int64),
            'float32': lambda: tf.FixedLenSequenceFeature([], tf.float32)
        }
    })
    output_types = flatten({
        'int': tf.int64,
        'float': tf.float32,
        'list':{
            'int': tf.int64,
            'float': tf.float32,
        }
    })
    output_shapes = flatten({
        'int': tf.TensorShape([]),
        'float': tf.TensorShape([]),
        'list': {
            'int': tf.TensorShape([None]),
            'float': tf.TensorShape([None]),
        }
    })
    def __init__(self):
        self.decode_functions = None
        self.construct_decode_functions = True

    def encode(self, data):
        feature_dict = {}
        for key, value in data.items():
            if isinstance(value, collections.Iterable):
                feature_dict[key] = TFRecordManager.encode_methods['list.'+type(value[0]).__name__](value)
                if self.construct_decode_functions:
                    self.decode_functions[key] = TFRecordManager.decode_methods['list.'+type(value[0]).__name__](value)
            else:
                feature_dict[key] = TFRecordManager.encode_methods[type(value).__name__](value)
                if self.construct_decode_functions:
                    self.decode_functions[key] = TFRecordManager.decode_methods['list.'+type(value[0]).__name__](value)
        if self.construct_decode_functions: self.construct_decode_functions = False
        return tf.train.Example(features=tf.train.Features(feature_dict))


    def write_tfrecords(self, iterable, base_name, out_folder, save_freq = 10):
        if not glob.glob(out_folder):
            os.makedirs(out_folder)
        count = 0
        for datum in map(self.encode, iterable):
            if count % save_freq == 0:
                if count>0:
                    writer.close()
                tfrecordfn = os.path.join(out_folder, base_name+'_'+str(count/save_freq))
                writer = tf.python_io.TFRecordWriter(tfrecordfn)
            writer.write(datum.SerializeToString())
            count+=1
            print('Write datum {} to tfrecord_file: {}'.format(count, tfrecordfn))









