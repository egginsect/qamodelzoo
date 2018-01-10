from collections import namedtuple
import json
import ipdb
import tensorflow as tf
from .preprocessor import Preprocessor
from .tfrecord import TFRecordManager
from glob import glob
import os
import multiprocessing

class QAPair(namedtuple('QAPair', 'question passage answer answer_start')):
    """Namedtuple used to store question, answer passage triplets"""
    def __new__(cls, question, passage, answer, char_ans_start):
        return super(QAPair, cls).__new__(cls, question, passage, answer, char_ans_start)

class SQuADData(object):
    """SQuaD Data Object"""
    def __init__(self, data):
        self.title = data['title']
        self.data = data['paragraphs']

    def __iter__(self):
        for item in self.data:
            for qapair in item['qas']:
                for answer in qapair['answers']:
                    yield QAPair(qapair['question'], item['context'], answer['text'], answer['answer_start'])


class TrainingData(object):
    "SQuAD data iterator"
    def __init__(self, infile, vocabfile):
        self.preprocessor = Preprocessor(vocabfile=vocabfile, allow_vocab_growth=True)
        with open(infile) as f:
            self.data  = json.load(f)
    def __iter__(self):
        for item in self.data['data']:
            for qapair in SQuADData(item):
                processed = self.preprocessor.preprocess(qapair)
                yield processed
