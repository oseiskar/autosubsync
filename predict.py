import argparse
import numpy as np

import find_transform
import features
import model

def parse_skew(skew):
    if skew is None: return None
    if '/' in skew:
        a, b = [float(x) for x in skew.split('/')]
        return a / b
    else:
        return float(skew)

def main(trained_model, sound_data, subvec, sample_rate, n_processes=3, fixed_skew=None, **kwargs):
    if kwargs.get('verbose', False):
        print(('computing features for %d audio samples ' + \
            'using %d parallel process(es)') % (len(sound_data), n_processes))

    fixed_skew = parse_skew(fixed_skew)

    features_x, shifted_y = features.compute(sound_data, subvec, sample_rate, n_processes=n_processes)
    y_scores = model.predict(trained_model, features_x)
    return find_transform.find_transform(shifted_y, y_scores, n_processes=n_processes, fixed_skew=fixed_skew, **kwargs)
