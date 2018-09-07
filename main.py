#!/usr/bin/python3
import argparse
import numpy as np
import os
import sys

import model
import predict
import preprocessing

from quality_of_fit import threshold

def main(model_file, video_file, subtitle_file, output_file, **kwargs):
    trained_model = model.load(model_file)

    target_data = preprocessing.import_target_files(video_file, subtitle_file)
    print('audio and subtitles extracted, fitting...')
    transform_func, quality = predict.main(trained_model, *target_data, verbose=True, **kwargs)

    print('quality of fit: %g, threshold %g' % (quality, threshold))
    if quality > threshold:
        print('---> SUCCESS!')
    else:
        sys.stderr.write(" *** WARNING: low quality of fit. Wrong subtitle file?\n")

    print('Fit complete. Performing resync, writing to ' + output_file)
    preprocessing.transform_srt(subtitle_file, output_file, transform_func)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('video_file')
    p.add_argument('subtitle_file')
    p.add_argument('output_file')
    p.add_argument('--model_file', default='trained-model.bin')
    p.add_argument('--max_shift_secs', default=20.0, type=float)
    p.add_argument('--parallelism', default=3, type=int)
    p.add_argument('--fixed_skew', default=None)
    args = p.parse_args()

    scores = main(args.model_file, args.video_file, args.subtitle_file, args.output_file,\
                  max_shift_secs=args.max_shift_secs, \
                  n_processes=args.parallelism,
                  fixed_skew=args.fixed_skew)
