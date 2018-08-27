#!/usr/bin/python3
import argparse
import numpy as np
import os
import tempfile

import model
import predict
import preprocessing

def main(model_file, video_file, subtitle_file, output_file):
    
    with open(model_file, 'rb') as f:
        trained_model = model.deserialize(f.read())
        
    tmp_dir = tempfile.mkdtemp()
    sound_file = os.path.join(tmp_dir, 'sound.flac')
    subs_tmp = os.path.join(tmp_dir, 'subs.csv')
    
    def clear():
        for to_delete in [sound_file, subs_tmp]:
            #print('deleting', to_delete)
            try: os.unlink(to_delete)
            except: pass
        try: os.rmdir(tmp_dir)
        except: pass
    
    try:
        preprocessing.extract_sound(video_file, sound_file)
        preprocessing.convert_subs_to_csv(subtitle_file, subs_tmp)
        print('------------ sound and subtitles extracted, performing sync')
        
        best_shift = predict.main(trained_model, sound_file, subs_tmp)
        print('optimal shift: %g' % best_shift)
        clear()
    
        print('------------ sound and subtitles extracted, performing sync')
        with open(output_file, 'w') as out_file:
            out_srt = preprocessing.srt_writer(out_file)
            for _, begin, end, text in preprocessing.read_srt(subtitle_file):
                out_srt.write(begin + best_shift, end + best_shift, text)
        
    finally:
        clear()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_file', default='trained-model.bin')
    p.add_argument('video_file')
    p.add_argument('subtitle_file')
    p.add_argument('output_file')
    args = p.parse_args()
    
    main(args.model_file, args.video_file, args.subtitle_file, args.output_file)