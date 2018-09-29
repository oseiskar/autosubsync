#!/usr/bin/python3
import argparse
import numpy as np
import os
import sys

def main(model_file, video_file, subtitle_file, output_file, **kwargs):
    # these are here to enable running as python3 autosubsync/main.py
    from autosubsync import model
    from autosubsync import predict
    from autosubsync import preprocessing
    from autosubsync.quality_of_fit import threshold

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

def cli(model_file=None):
    p = argparse.ArgumentParser()
    p.add_argument('video_file')
    p.add_argument('subtitle_file')
    p.add_argument('output_file')

    # Make model file an argument only in the non-packaged version
    if model_file is None:
        p.add_argument('--model_file', default='trained-model.bin')

    p.add_argument('--max_shift_secs', default=20.0, type=float)
    p.add_argument('--parallelism', default=3, type=int)
    p.add_argument('--fixed_skew', default=None)
    args = p.parse_args()

    if model_file is None:
        model_file = args.model_file

    scores = main(model_file, args.video_file, args.subtitle_file, args.output_file,\
                  max_shift_secs=args.max_shift_secs, \
                  n_processes=args.parallelism,
                  fixed_skew=args.fixed_skew)

def cli_packaged():
    """Entry point in the packaged, pip-installable version"""
    from pkg_resources import resource_filename
    cli(model_file=resource_filename(__name__, '../trained-model.bin'))

if __name__ == '__main__':
    # Entry point for running from repository root folder
    sys.path.append('.')
    cli()
