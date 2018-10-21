import numpy as np
import soundfile

from autosubsync import srt_io

def generate_intervals(file_length_seconds,
    mean_speech_length_seconds = 5,
    mean_pause_length_seconds = 4,
    end_silence_percent=5):

    t = 0
    paused = True
    while t < file_length_seconds * (1-end_silence_percent*0.01):
        if paused:
            mean_length = mean_pause_length_seconds
        else:
            mean_length = mean_speech_length_seconds
        delta_t = np.random.exponential(scale=mean_length)

        if not paused:
            yield (t, t + delta_t)

        t += delta_t
        paused = not paused

def write_texts(file_name, intervals):
    text = list('test sentence äö å.'*3)
    with open(file_name, 'wb') as f:
        srt = srt_io.writer(f)
        for t0, t1 in intervals:
            np.random.shuffle(text)
            line = ''.join(text[:(np.random.randint(len(text))+5)]).strip()
            srt.write(t0, t1, bytearray(line, 'utf-8'))

def generate_sound(intervals, file_length_seconds, skew, shift_seconds):

    sample_rate = 20000 # Hz
    sound = np.zeros(int(sample_rate*file_length_seconds))

    sync_noise_seconds = 0.1
    noise = lambda: sync_noise_seconds * np.random.randn()
    for interval in intervals:
        t0, t1 = [t * skew + shift_seconds + noise() for t in interval]
        s = slice(int(t0*sample_rate), int(t1*sample_rate))
        sound[s] = np.random.randn(len(sound[s]))

    return sound, sample_rate

def generate(sound_file_name, srt_file_name, skew, shift_seconds):

    file_length_seconds = 15 * 60
    intervals = list(generate_intervals(file_length_seconds))

    sound, sample_rate = \
        generate_sound(intervals, file_length_seconds, skew, shift_seconds)

    soundfile.write(sound_file_name, sound, sample_rate)
    write_texts(srt_file_name, intervals)

def set_seed(s):
    np.random.seed(s)

if __name__ == '__main__':
    # example
    set_seed(0)
    skew = 24/25.0
    shift_seconds = 4.0
    generate('/tmp/sound.flac', '/tmp/subs.srt', skew, shift_seconds)
