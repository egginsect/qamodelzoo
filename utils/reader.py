import numpy as np
from collections import namedtuple
import json
import ipdb
import tensorflow as tf
from indexer import Indexer
from tfrecord import TFRecordManager
from glob import glob
import multiprocessing

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class QAPair(namedtuple('QAPair', 'question answer_start answer passage id')):
    """Namedtuple used to store question, answer passage triplets"""
    def __new__(cls, question, ansdict, passage, id):

        return super(QAPair, cls).__new__(cls, question, ansdict['answer_start'], ansdict['text'], passage, id)

class SQuADData(object):
    """SQuaD Data Object"""
    def __init__(self, data):
        self.title = data['title']
        self.data = data['paragraphs']
    def __iter__(self):
        for item in self.data:
            for qapair in item['qas']:
                for answer in qapair['answers']:
                    yield QAPair(qapair['question'], answer, item['context'], qapair['id'])

class TrainingData(object):
    "SQuAD data iterator"
    def __init__(self, infile, vocabfile):
        self.indexer = Indexer(vocabfile, allow_vocab_growth=True)
        with open(infile) as f:
            self.data  = json.load(f)
    def __iter__(self):
        for item in self.data['data']:
            for qapair in SQuADData(item):
                data = {'question':qapair.question,
                        'answer':qapair.answer,
                        'passage':qapair.passage}
                yield {k:self.indexer.index_sentence(text.split()) for k, text in data.items()}

if __name__=="__main__":
    reader = TrainingData('../data/train-v1.1.json', '../glove/glove.6B.100d.txt')
    tfmanager = TFRecordManager()
    tfmanager.write_tfrecords(reader, 'squad', '../data/processed', save_freq = 10)
    with open(os.path.join('../data/processed', 'extended_vocab.json'), 'w') as f:
         json.dump(reader.indexer.vocab, f)
    fns = tfmanager.load_tfrecords('../data/processed')
    dataset = tf.data.TFRecordDataset(fns)
    decode_func = TFRecordManager.tfrecord_decoder(glob('../data/processed/decode_dict.json')[0])

    dataset = dataset.map(decode_func, num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.padded_batch(10, padded_shapes=dataset.output_shapes)
    sess = tf.InteractiveSession()
    train_iterator = dataset.make_one_shot_iterator()
    train_iterator_handle = sess.run(train_iterator.string_handle())
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types)
    next_element = iterator.get_next()
    ipdb.set_trace()
    train_data = sess.run(next_element, feed_dict={handle: train_iterator_handle})

