import numpy as np
import features
import preprocessing
import model

def get_training_features(index_file):
    file_number = 0
    all_x = []
    all_y = []
    all_meta = []
    for sound_data, subvec, sample_rate in preprocessing.read_training_data(index_file):
        training_x, training_y = features.compute(sound_data, subvec, sample_rate)
        all_x.append(training_x)
        all_y.append(training_y)
        all_meta.append(np.ones(training_y.shape) * file_number)
        file_number += 1

    return np.vstack(all_x), np.ravel(all_y), np.ravel(all_meta)

data_x, data_y, data_file = get_training_features('training-data/index.csv')

trained = model.train(data_x, data_y)
with open('trained-model.bin', 'wb') as f:
    f.write(model.serialize(trained))