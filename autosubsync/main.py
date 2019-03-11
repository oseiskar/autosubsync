#!/usr/bin/python3
import argparse
import numpy as np
import gc
import os
import sys

def parse_skew(skew):
    "helper function, parse maybe fractional notation like 24/24 to float"
    if skew is None: return None
    if '/' in skew:
        a, b = [float(x) for x in skew.split('/')]
        return a / b
    else:
        return float(skew)

def synchronize(video_file, subtitle_file, output_file, verbose=False, \
    parallelism=3, fixed_skew=None, model_file=None, return_parameters=False, \
    **kwargs):
    """
    Automatically synchronize subtitles with audio in a video file.
    Uses FFMPEG to extract the audio from the video file and the command line
    tool "ffmpeg" must be available. Uses temporary files which are deleted
    automatically.

    Args:
        video_file (string): Input video file name
        subtitle_file (string): Input SRT subtitle file name
        output_file (string): Output (synchronized) SRT subtitle file name
        verbose (boolean): If True, print progress information to stdout
        return_parameters (boolean): If True, returns the syncrhonization
            parameters instead of just the success flag
        other arguments: Search parameters, see ``autosubsync --help``

    Returns:
        If return_parameters is False (default), returns
        True on success (quality of fit test passed), False if failed.

        If return_parameters is True, returns a tuple of four values

            success (boolean)   success flag as above
            quality (float)     metric used to determine the value of "success"
            skew (float)        best fit skew/speed (unitless)
            shift (float)       best fit shift in seconds

    """

    # these are here to enable running as python3 autosubsync/main.py
    from autosubsync import features
    from autosubsync import find_transform
    from autosubsync import model
    from autosubsync import preprocessing
    from autosubsync import quality_of_fit
    from autosubsync import srt_io

    # first check that the SRT file is valid before extracting any audio data
    srt_io.check_file(subtitle_file)

    # argument parsing
    if model_file is None:
        from pkg_resources import resource_filename
        model_file = resource_filename(__name__, '../trained-model.bin')

    fixed_skew = parse_skew(fixed_skew)

    # load model
    trained_model = model.load(model_file)

    if verbose: print('Extracting audio using ffmpeg and reading subtitles...')
    sound_data, subvec = preprocessing.import_target_files(video_file, subtitle_file)

    if verbose: print(('computing features for %d audio samples ' + \
        'using %d parallel process(es)') % (len(subvec), parallelism))

    features_x, shifted_y = features.compute(sound_data, subvec, parallelism=parallelism)

    if verbose: print('extracted features of size %s, performing speech detection' % \
        str(features_x.shape))

    y_scores = model.predict(trained_model, features_x)

    # save some memory before parallelization fork so we look less bad
    del features_x, sound_data, subvec
    gc.collect()

    if verbose:
        print('computing best fit with %d frames' % len(y_scores))

    skew, shift, quality = find_transform.find_transform_parameters(\
        shifted_y, y_scores, \
        parallelism=parallelism, fixed_skew=fixed_skew, bias=trained_model[1], \
        verbose=verbose, **kwargs)

    success = quality > quality_of_fit.threshold
    if verbose:
        print('quality of fit: %g, threshold %g' % (quality, quality_of_fit.threshold))
        print('Fit complete. Performing resync, writing to ' + output_file)

    transform_func = find_transform.parameters_to_transform(skew, shift)
    preprocessing.transform_srt(subtitle_file, output_file, transform_func)

    if verbose and success: print('success!')

    if return_parameters:
        return success, quality, skew, shift
    else:
        return success

def cli(packaged_model=False):
    p = argparse.ArgumentParser(description=synchronize.__doc__.split('\n\n')[0])
    p.add_argument('video_file', help='Input video file')
    p.add_argument('subtitle_file', help='Input SRT subtitle file')
    p.add_argument('output_file', help='Output (auto-synchronized) SRT subtitle file')

    # Make model file an argument only in the non-packaged version
    if not packaged_model:
        p.add_argument('--model_file', default='trained-model.bin')

    p.add_argument('--max_shift_secs', default=20.0, type=float,
        help='Maximum subtitle shift in seconds (default 20)')
    p.add_argument('--parallelism', default=3, type=int,
        help='Number of parallel worker processes (default 3)')
    p.add_argument('--fixed_skew', default=None,
        help='Use a fixed skew (e.g. 1) instead of auto-detection')
    p.add_argument('--silent', action='store_true',
        help='Do not print progress information')
    args = p.parse_args()

    if packaged_model:
        model_file = None
    else:
        model_file = args.model_file

    success = synchronize(args.video_file, args.subtitle_file, args.output_file,\
                  verbose = not args.silent,
                  model_file = model_file,
                  max_shift_secs = args.max_shift_secs, \
                  parallelism = args.parallelism,
                  fixed_skew = args.fixed_skew)

    if not success:
        sys.stderr.write("\nWARNING: low quality of fit. Wrong subtitle file?\n")
        sys.exit(1)

def cli_packaged():
    """Entry point in the packaged, pip-installable version"""
    cli(packaged_model=True)

if __name__ == '__main__':
    # Entry point for running from repository root folder
    sys.path.append('.')
    cli(packaged_model=False)
