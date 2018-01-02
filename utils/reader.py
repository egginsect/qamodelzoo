import numpy as np
from collections import namedtuple
import json
import ipdb
import tensorflow as tf
from indexer import Indexer
from tfrecord import TFRecordManager


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
    def __new__(cls, question, ansdict, passage, id):

        return super(QAPair, cls).__new__(cls, question, ansdict['answer_start'], ansdict['text'], passage, id)

class SQuADData(object):
    def __init__(self, data):
        self.title = data['title']
        self.data = data['paragraphs']
    def __iter__(self):
        for item in self.data:
            for qapair in item['qas']:
                for answer in qapair['answers']:
                    yield QAPair(qapair['question'], answer, item['context'], qapair['id'])

class TrainingData(object):
    def __init__(self, infile, vocabfile):
        self.indexer = Indexer(vocabfile, allow_growth=True)
        with open(infile) as f:
            self.data  = json.load(f)
    def __iter__(self):
        for item in self.data['data']:
            for qapair in SQuADData(item):
                data = {'question':qapair.question,
                        'answer':qapair.answer}
                yield tuple([lambda text: self.indexer.index_sentence(text.split()), triplet))

if __name__=="__main__":
    reader = TrainingData('../data/train-v1.1.json', '../glove/glove.6B.100d.txt')
    tfmanager = TFRecordManager()
    ipdb.set_trace()
    tfmanager.write_tfrecords(reader, 'squad', 'data/processed')
    ipdb.set_trace()
    dataset = tf.data.Dataset.from_generator(reader.__iter__, (tf.int64, tf.int64, tf.int64),
                           (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])))
    sess = tf.InteractiveSession()
    train_iterator = dataset.make_one_shot_iterator()
    train_iterator_handle = sess.run(train_iterator.string_handle())
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types)
    next_element = iterator.get_next()
    ipdb.set_trace()
    train_data = sess.run(next_element, feed_dict={handle: train_iterator_handle})
    ipdb.set_trace()

