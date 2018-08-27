import argparse
import numpy as np

import auto_shift
import features
import preprocessing
import model

def main(trained_model, sound_file, subtitle_file):
    sound_data, subvec, sample_rate = preprocessing.import_item(sound_file, subtitle_file)
    features_x, shifted_y = features.compute(sound_data, subvec, sample_rate)
    
    y_scores = model.predict(trained_model, features_x)
    return auto_shift.best_shift(shifted_y, y_scores)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_file', default='trained-model.bin')
    p.add_argument('sound_file')
    p.add_argument('subtitle_file')
    
    args = p.parse_args()
    
    with open(args.model_file, 'rb') as f:
        trained_model = model.deserialize(f.read())
    
    r = main(trained_model, args.sound_file, args.subtitle_file)
    print(r)