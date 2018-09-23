#!/usr/bin/python3
import csv, os, subprocess

TRAINING_FOLDER = 'training-data/'

if not os.path.exists(TRAINING_FOLDER):
    os.makedirs(TRAINING_FOLDER)

from preprocessing import extract_sound
from shutil import copyfile

indexfile = os.path.join(TRAINING_FOLDER, 'index.csv')
sound_id = 1
with open('training-sources.csv') as f, open(indexfile, 'wt') as index:
    indexwriter = csv.writer(index)
    indexwriter.writerow(['sound', 'subtitles', 'language'])
    for line in csv.DictReader(f):
        print(line)
        sound_file = 'sound_%03d.flac' % sound_id
        sub_file = 'subs_%03d.srt' % sound_id
        sound_id += 1

        extract_sound(line['video'], os.path.join(TRAINING_FOLDER, sound_file))
        copyfile(line['subtitles'], os.path.join(TRAINING_FOLDER, sub_file))

        indexwriter.writerow((sound_file, sub_file, line['language']))
