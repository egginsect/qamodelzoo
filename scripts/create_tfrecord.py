import argparse
from utils import TrainingData, TFRecordManager
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default='data/train-v1.1.json', help='Input file for SQuAD training data', type=str)
parser.add_argument('-o', '--output_directory', default='data/processed', help='Output Folder for saving tfrecords', type=str)
parser.add_argument('-v', '--vocab_file', default='glove/glove.6B.100d.txt', help='Vocabulary File', type=str)

config = parser.parse_args()
reader = TrainingData(config.input_file, config.vocab_file)

tfmanager = TFRecordManager()
tfmanager.write_tfrecords(reader, 'squad', config.output_directory, save_freq = 10)
with open(os.path.join('data/processed', 'extended_vocab.json'), 'w') as f:
    json.dump(reader.preprocessor.vocab, f)

