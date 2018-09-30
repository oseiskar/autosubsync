import argparse
import numpy as np

from . import find_transform
from . import features
from . import model

def parse_skew(skew):
    if skew is None: return None
    if '/' in skew:
        a, b = [float(x) for x in skew.split('/')]
        return a / b
    else:
        return float(skew)

def main(trained_model, sound_data, subvec, sample_rate, parallelism=3, fixed_skew=None, **kwargs):
    if kwargs.get('verbose', False):
        print(('computing features for %d audio samples ' + \
            'using %d parallel process(es)') % (len(sound_data), parallelism))

    fixed_skew = parse_skew(fixed_skew)

    features_x, shifted_y = features.compute(sound_data, subvec, sample_rate, parallelism=parallelism)
    y_scores = model.predict(trained_model, features_x)
    bias = trained_model[1]
    return find_transform.find_transform(shifted_y, y_scores, parallelism=parallelism, fixed_skew=fixed_skew, bias=bias, **kwargs)
