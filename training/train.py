import numpy as np
import pandas as pd
import csv
import sys
import os

sys.path.append('.')

from autosubsync import features
from autosubsync import model
from autosubsync import preprocessing

def read_training_data(index_file):
    base_path = os.path.dirname(index_file)
    locate = lambda f: os.path.join(base_path, f)
    file_number = 0
    with open(index_file) as index:
        for item in csv.DictReader(index):
            file_number += 1
            sound_data, sub_vec, sample_rate = preprocessing.import_item(locate(item['sound']), locate(item['subtitles']))
            yield(sound_data, sub_vec, sample_rate, item['language'], file_number)

def compute_feature_table(index_file):
    print('computing features')
    all_x = []
    all_y = []
    all_numbers = []
    all_languages = []
    for sound_data, subvec, sample_rate, language, file_number in read_training_data(index_file):
        print('file %d' % file_number)
        training_x, training_y = features.compute(sound_data, subvec, sample_rate)
        all_x.append(training_x)
        all_y.extend(training_y)
        all_numbers.extend([file_number]*len(training_y))
        all_languages.extend([language]*len(training_y))

    meta = pd.DataFrame(np.array([all_y, all_numbers, all_languages]).T,
                        columns=['label', 'file_number', 'language'])

    return np.vstack(all_x), meta

def load_features(feature_file='training/data/features.npy', meta_file='training/data/meta.csv'):
    data_x = np.load(feature_file)
    data_meta = pd.read_csv(meta_file, index_col=0)
    return data_x, data_meta

if __name__ == '__main__':
    def parse_args():
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('--index_file', default='training/data/index.csv')
        p.add_argument('--meta_file', default='training/data/meta.csv')
        p.add_argument('--features_file', default='training/data/features.npy')
        p.add_argument('--model_file', default='trained-model.bin')
        p.add_argument('--compute_features', action='store_true')
        return p.parse_args()

    args = parse_args()

    if args.compute_features or not os.path.exists(args.features_file):
        data_x, data_meta = compute_feature_table(args.index_file)
        data_meta.to_csv(args.meta_file)
        np.save(args.features_file, data_x)

    data_x, data_meta = load_features(args.features_file, args.meta_file)

    print('training data extracted, shape ' + str(data_x.shape))

    print('training...')
    trained = model.train(data_x, data_meta.label.values, data_meta, verbose=True)

    target_file = args.model_file
    print('serializing model to ' + target_file)
    model.save(trained, target_file)
