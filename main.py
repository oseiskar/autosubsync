#!/usr/bin/python3
import argparse
import numpy as np
import os

import model
import predict
import preprocessing

def main(model_file, video_file, subtitle_file, output_file, **kwargs):
    trained_model = model.load(model_file)
        
    target_data = preprocessing.import_target_files(video_file, subtitle_file)
    print('------------ sound and subtitles extracted, fitting...')
    transform_func = predict.main(trained_model, *target_data, verbose=True, **kwargs)
    
    print('------------ fit complete, performing resync')
    preprocessing.transform_srt(subtitle_file, output_file, transform_func)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('video_file')
    p.add_argument('subtitle_file')
    p.add_argument('output_file')
    p.add_argument('--model_file', default='trained-model.bin')
    p.add_argument('--max_shift_secs', default=10.0, type=float)
    args = p.parse_args()
    
    scores = main(args.model_file, args.video_file, args.subtitle_file, args.output_file,\
                  max_shift_secs=args.max_shift_secs)