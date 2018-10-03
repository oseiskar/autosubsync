import os
import sys
import tempfile
import subprocess
import numpy as np
from . import srt_io

def import_sound(sound_path):
    import soundfile

    target_sample_rate = 20000
    warning_threshold = 8000

    # minimize memory usage by reading audio as 16-bit integers instead of
    # float or double. This should be the maximum precision of the samples
    # anyway
    samples, sample_rate = soundfile.read(sound_path, dtype='int16')
    data_range = 2**15

    # even 8-bit audio would work here, but already affects performance
    # samples, data_range = (samples / 256).astype(np.int8), 128

    ds_factor = max(int(np.floor(sample_rate / target_sample_rate)), 1)
    samples = samples[::ds_factor]
    sample_rate = sample_rate / ds_factor

    if sample_rate < warning_threshold:
        # probably never happens but checking anyway
        sys.stderr.write('warning: low sound sample rate %d Hz\n' % sample_rate)

    return samples, sample_rate, data_range

def build_sub_vec(subs, sample_rate, n, sub_filter=None):
    subvec = np.zeros(n, np.bool)
    to_index = lambda x: int(sample_rate*x)
    for line in subs:
        if sub_filter is not None and not sub_filter(line.text): continue
        subvec[to_index(line.begin):to_index(line.end)] = 1
    return subvec

def import_subs(srt_filename, sample_rate, n, **kwargs):
    audio_length = n / float(sample_rate)
    subs = list(srt_io.read_file(srt_filename))
    if len(subs) > 0:
        subs_length = np.max([s.end for s in subs])
        rel_err = abs(subs_length - audio_length) / max(subs_length, audio_length)
        if rel_err > 0.25: # warning threshold
            sys.stderr.write(" *** WARNING: subtitle and audio lengths " + \
                             "differ by %d%%. Wrong subtitle file?\n" % int(rel_err*100))

    else:
        sys.stderr.write(" *** WARNING: empty subtitle file\n")
    return build_sub_vec(subs, sample_rate, n, **kwargs)

def import_item(sound_file, subtitle_file, **kwargs):
    sound_data = import_sound(sound_file)
    samples, sample_rate, data_range = sound_data
    n = len(samples)
    sub_vec = import_subs(subtitle_file, sample_rate, n, **kwargs)
    return sound_data, sub_vec

def extract_sound(input_video_file, output_sound_file):
    convert_cmd = [
        'ffmpeg',
        '-y', # overwrite if exists
        '-loglevel', 'error',
        '-i', input_video_file, # input
        '-ac', '1', # convert to mono
        output_sound_file
    ]
    subprocess.call(convert_cmd)

def import_target_files(video_file, subtitle_file, **kwargs):
    "Import prediction target files using a temporary directory"
    tmp_dir = tempfile.mkdtemp()
    sound_file = os.path.join(tmp_dir, 'sound.flac')

    def clear():
        try: os.unlink(sound_file)
        except: pass

        try: os.rmdir(tmp_dir)
        except: pass

    try:
        extract_sound(video_file, sound_file)
        return import_item(sound_file, subtitle_file, **kwargs)

    finally:
        clear()

def transform_srt(in_srt, out_srt, transform_func):
    with open(out_srt, 'wb') as out_file:
        out_srt = srt_io.writer(out_file)
        for sub in srt_io.read_file(in_srt):
            out_srt.write(transform_func(sub.begin), transform_func(sub.end), sub.text)
