#!/usr/bin/python3
import csv, os, subprocess, io

TRAINING_FOLDER = 'training-data/'

if not os.path.exists(TRAINING_FOLDER):
    os.makedirs(TRAINING_FOLDER)

def read_srt(input_file):
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

def convert_subs(input_file, output_file):
    with open(output_file, 'wt') as out:
        writer = csv.writer(out)
        writer.writerow(['begin', 'end', 'text'])
        for _, begin, end, text in read_srt(input_file):
            writer.writerow([begin, end, text])

indexfile = os.path.join(TRAINING_FOLDER, 'index.csv')
sound_id = 1
with open('training-sources.csv') as f, open(indexfile, 'wt') as index:
    indexwriter = csv.writer(index)
    indexwriter.writerow(['sound', 'subtitles'])
    for line in csv.DictReader(f):
        sound_file = os.path.join(TRAINING_FOLDER, 'sound_%03d.flac' % sound_id)
        sub_file = os.path.join(TRAINING_FOLDER, 'subs_%03d.csv' % sound_id)
        sound_id += 1

        convert_cmd = [
            'ffmpeg',
            '-y', # overwrite if exists
            '-i', line['video'], # input
            '-ac', '1', # convert to mono
            sound_file
        ]
        subprocess.call(convert_cmd)
        convert_subs(line['subtitles'], sub_file)
        indexwriter.writerow((sound_file, sub_file))
