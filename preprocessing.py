import csv
import os
import numpy as np
    
def read_srt(input_file):
    """
    Read an SRT file to (seq, begin, end, text) tuples, where
    begin and end are float timestamps in seconds
    """
    import io
    def read_line_block(f):
        lines = []
        for line in f:
            if len(line.strip()) == 0:
                return lines
            lines.append(line.rstrip())
        return lines

    def parse_time(timestamp):
        hours, minutes, secs = timestamp.split(':')
        return (int(hours)*60 + int(minutes))*60 + float(secs.replace(',', '.'))

    with io.open(input_file) as f:
        while True:
            block = read_line_block(f)
            if len(block) == 0: break
            seq = int(block[0].replace('\ufeff', '')) # remove BOM
            times = block[1]
            begin, _, end = times.partition(' --> ')
            text = '\n'.join(block[2:])
            yield(seq, parse_time(begin), parse_time(end), text)
            
class srt_writer:
    def __init__(self, file):
        self.seq = 1
        self.file = file
    
    def write(self, begin, end, text):
        self._write_row(self.seq)
        self._write_row(self._format_time(begin) + ' --> ' + self._format_time(end))
        self._write_row(text)
        self._write_row('')
        self.seq += 1
    
    def _write_row(self, text):
        text = str(text).rstrip().replace('\n', '\r\n')
        self.file.write(text + '\r\n')
    
    def _format_time(self, t_secs):
        msecs = round(t_secs*1000)
        secs = int(msecs / 1000) % 60
        mins = int(msecs / (60*1000)) % 60
        hours = int(msecs / (60*60*1000))
        msecs = msecs % 1000
        return "%02d:%02d:%02d,%03d" % (hours, mins, secs, msecs)
            
def convert_subs_to_csv(input_file, output_file):
    with open(output_file, 'wt') as out:
        writer = csv.writer(out)
        writer.writerow(['begin', 'end', 'text'])
        for _, begin, end, text in read_srt(input_file):
            writer.writerow([begin, end, text])

def import_sound(sound_path):
    import soundfile
    sound_data, sample_rate = soundfile.read(sound_path)
    return sound_data, sample_rate

def build_sub_vec(subs, sample_rate, n):
    subvec = np.zeros(n)
    to_index = lambda x: int(sample_rate*x)
    for _, line in subs.iterrows():
        begin = line['begin']
        end = line['end']
        subvec[to_index(begin):to_index(end)] = 1
    return subvec

def import_sub_csv(subs_csv_path, sample_rate, n):
    import pandas as pd
    return build_sub_vec(pd.read_csv(subs_csv_path), sample_rate, n)

def import_item(sound_file, subtitle_file):
    sound_data, sample_rate = import_sound(sound_file)
    n = len(sound_data)
    sub_vec = import_sub_csv(subtitle_file, sample_rate, n)
    return sound_data, sub_vec, sample_rate

def extract_sound(input_video_file, output_sound_file):
    import subprocess
    convert_cmd = [
        'ffmpeg',
        '-y', # overwrite if exists
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

def import_target_files(video_file, subtitle_file):
    "Import prediction target files using a temporary directory"
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    sound_file = os.path.join(tmp_dir, 'sound.flac')
    subs_tmp = os.path.join(tmp_dir, 'subs.csv')
    
    def clear():
        print('--- clearing temporary data')
        for to_delete in [sound_file, subs_tmp]:
            #print('deleting', to_delete)
            try: os.unlink(to_delete)
            except: pass
        try: os.rmdir(tmp_dir)
        except: pass
    
    try:
        extract_sound(video_file, sound_file)
        convert_subs_to_csv(subtitle_file, subs_tmp)
        return import_item(sound_file, subs_tmp)
        
    finally:
        clear()