import csv
import os
import sys
import numpy as np
import srt_io

def convert_subs_to_csv(input_file, output_file):
    with open(output_file, 'wt') as out:
        writer = csv.writer(out)
        writer.writerow(['begin', 'end', 'text'])
        for _, begin, end, text in srt_io.read_file(input_file):
            writer.writerow([begin, end, text])

def import_sound(sound_path):
    import soundfile
    sound_data, sample_rate = soundfile.read(sound_path)
    return sound_data, sample_rate

def build_sub_vec(subs, sample_rate, n, sub_filter=None):
    subvec = np.zeros(n)
    to_index = lambda x: int(sample_rate*x)
    for _, line in subs.iterrows():
        if sub_filter is not None and not sub_filter(line['text']): continue
        begin = line['begin']
        end = line['end']
        subvec[to_index(begin):to_index(end)] = 1
    return subvec

def import_sub_csv(subs_csv_path, sample_rate, n, **kwargs):
    import pandas as pd
    audio_length = n / float(sample_rate)
    subs = pd.read_csv(subs_csv_path)
    if subs.shape[1] > 0:
        subs_length = subs.end.max()
        rel_err = abs(subs_length - audio_length) / max(subs_length, audio_length)
        if rel_err > 0.25: # warning threshold
            sys.stderr.write(" *** WARNING: subtitle and audio lengths " + \
                             "differ by %d%%. Wrong subtitle file?\n" % int(rel_err*100))
            
    else:
        sys.stderr.write(" *** WARNING: empty subtitle file\n")
    return build_sub_vec(subs, sample_rate, n, **kwargs)

def import_item(sound_file, subtitle_file, **kwargs):
    sound_data, sample_rate = import_sound(sound_file)
    n = len(sound_data)
    sub_vec = import_sub_csv(subtitle_file, sample_rate, n, **kwargs)
    return sound_data, sub_vec, sample_rate

def extract_sound(input_video_file, output_sound_file):
    import subprocess
    convert_cmd = [
        'ffmpeg',
        '-y', # overwrite if exists
        '-loglevel', 'error',
        '-i', input_video_file, # input
        '-ac', '1', # convert to mono
        output_sound_file
    ]
    subprocess.call(convert_cmd)

def read_training_data(index_file):
    base_path = os.path.dirname(index_file)
    locate = lambda f: os.path.join(base_path, f)
    file_number = 0
    with open(index_file) as index:
        for item in csv.DictReader(index):
            file_number += 1
            sound_data, sub_vec, sample_rate = import_item(locate(item['sound']), locate(item['subtitles']))
            yield(sound_data, sub_vec, sample_rate, item['language'], file_number)

def import_target_files(video_file, subtitle_file, **kwargs):
    "Import prediction target files using a temporary directory"
    import tempfile
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
        print('Cleared temporary data')
    
    try:
        print('Extracting audio using ffmpeg and reading subtitles...')
        extract_sound(video_file, sound_file)
        convert_subs_to_csv(subtitle_file, subs_tmp)
        return import_item(sound_file, subs_tmp, **kwargs)
        
    finally:
        clear()

def transform_srt(in_srt, out_srt, transform_func):
    with open(out_srt, 'wb') as out_file:
        out_srt = srt_io.writer(out_file)
        for _, begin, end, text in srt_io.read_file(in_srt):
            out_srt.write(transform_func(begin), transform_func(end), text)