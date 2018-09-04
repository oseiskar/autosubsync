import argparse
import numpy as np

import find_transform
import features
import model

def main(trained_model, sound_data, subvec, sample_rate, n_processes = 3, **kwargs):
    features_x, shifted_y = features.compute(sound_data, subvec, sample_rate, n_processes=n_processes)
    y_scores = model.predict(trained_model, features_x)
    return find_transform.find_transform(shifted_y, y_scores, **kwargs)
