#!/usr/bin/python3
import csv, os, subprocess

TRAINING_FOLDER = 'training-data/'

if not os.path.exists(TRAINING_FOLDER):
    os.makedirs(TRAINING_FOLDER)
    
from preprocessing import convert_subs_to_csv, extract_sound

indexfile = os.path.join(TRAINING_FOLDER, 'index.csv')
sound_id = 1
with open('training-sources.csv') as f, open(indexfile, 'wt') as index:
    indexwriter = csv.writer(index)
    indexwriter.writerow(['sound', 'subtitles'])
    for line in csv.DictReader(f):
        sound_file = 'sound_%03d.flac' % sound_id
        sub_file = 'subs_%03d.csv' % sound_id
        sound_id += 1

        extract_sound(line['video'], os.path.join(TRAINING_FOLDER, sound_file))
        convert_subs_to_csv(line['subtitles'], os.path.join(TRAINING_FOLDER, sub_file))
        indexwriter.writerow((sound_file, sub_file))
