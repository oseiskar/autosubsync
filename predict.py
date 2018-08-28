import argparse
import numpy as np

import auto_shift
import features
import preprocessing
import model

def main(trained_model, sound_file, subtitle_file, max_shift_secs):
    sound_data, subvec, sample_rate = preprocessing.import_item(sound_file, subtitle_file)
    features_x, shifted_y = features.compute(sound_data, subvec, sample_rate)
    
    y_scores = model.predict(trained_model, features_x)
    return auto_shift.best_shift(shifted_y, y_scores, max_shift_secs=max_shift_secs)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('sound_file')
    p.add_argument('subtitle_file')
    p.add_argument('--model_file', default='trained-model.bin')
    p.add_argument('--max_shift_secs', default=2.0, type=float)
    p.add_argument('--plot_scores', action='store_true')
    
    args = p.parse_args()
    
    with open(args.model_file, 'rb') as f:
        trained_model = model.deserialize(f.read())
    
    best_shift, scores = main(trained_model, args.sound_file, args.subtitle_file, args.max_shift_secs)
    print(" --- Optimal shift = %g seconds" % best_shift)
    
    if args.plot_scores:
        import matplotlib.pyplot as plt
        plt.plot(scores)
        plt.show()
    