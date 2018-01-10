import collections
import os
import tensorflow as tf
from .misc import flatten, unflatten
import glob
import json
import ipdb
class TFRecordManager(object):
    """
    TF Record Manager that detects type and encode data into TF Records.
    Decode function will be automatically generated when encoding into TF Records.
    """
    encode_methods = flatten({
        'int': lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])),
        'float': lambda value: tf.train.Feature(int64_list=tf.train.FloatList(value=[value])),
        'list':{
            'int': lambda value: tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
            'float': lambda value: tf.train.Feature(float_list=tf.train.FloatList(value=value)),
        }
    })
    decode_methods = flatten({
        'int': lambda : tf.FixedLenFeature([], tf.int64),
        'float': lambda : tf.FixedLenFeature([], tf.float32),
        'list': {
            'int': lambda: tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'float32': lambda: tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
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
        self.decode_functions = {}
        self.construct_decode_functions = True

    def encode(self, data):
        """Encode data to TF Record"""
        feature_dict = {}
        for key, value in data.items():
            if isinstance(value, collections.Iterable):
                feature_dict[key] = TFRecordManager.encode_methods['list.'+type(value[0]).__name__](value)
                if self.construct_decode_functions:
                    self.decode_functions[key] = 'list.'+type(value[0]).__name__
            else:
                feature_dict[key] = TFRecordManager.encode_methods[type(value).__name__](value)
                if self.construct_decode_functions:
                    self.decode_functions[key] = type(value).__name__
        if self.construct_decode_functions:
            self.construct_decode_functions = False
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def write_tfrecords(self, iterable, base_name, out_folder, save_freq = 10):
        if not glob.glob(out_folder):
            os.makedirs(out_folder)
        count = 0
        for datum in map(self.encode, iterable):
            if count % save_freq == 0:
                if count>0:
                    writer.close()
                tfrecordfn = os.path.join(out_folder, base_name+'_'+str(int(count/save_freq))+'.tfrecord')
                writer = tf.python_io.TFRecordWriter(tfrecordfn)
            writer.write(datum.SerializeToString())
            count+=1
            print('Write datum {} to tfrecord_file: {}'.format(count, tfrecordfn))
        ipdb.set_trace()
        with open(os.path.join(out_folder,'decode_file.json'), 'w') as f:
            json.dump(self.decode_functions, f)

    @staticmethod
    def tfrecord_decoder(decode_json_file):
        """Function to generate decoder function given decode json file"""
        with open(decode_json_file) as f:
            decode_dict = json.load(f)
        feature_def = {}
        for key, value in decode_dict.items():
            feature_def[key] = TFRecordManager.decode_methods[value]()

        def decode_tfrecord(datum):
            return tf.parse_single_example(datum, features=feature_def)

        return decode_tfrecord

    @staticmethod
    def construct_place_holder(decode_json_file):
        with open(decode_json_file) as f:
            decode_dict = json.load(f)
        feature_def = {}
        for key, value in decode_dict.items():
            feature_def[key] = TFRecordManager.decode_methods[value]()
    @staticmethod
    def load_tfrecords(tfrecord_folder):
        tfrecord_fns = glob.glob(os.path.join(tfrecord_folder, '*.tfrecord'))
        return tfrecord_fns









