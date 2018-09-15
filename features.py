import numpy as np
import os
import sys

import preprocessing

frame_secs = 0.05

def split_to_frames(data, frame_size):
    n_frames = int(len(data)/frame_size)
    data = data[:(n_frames*frame_size)]
    return np.reshape(data, (n_frames, frame_size))

def shift_frames(frames, delta):
    if delta == 0:
        return frames
    result = frames*0
    if delta > 0:
        result[delta:,:] = frames[:-delta]
    else:
        result[:delta,:] = frames[-delta:,:]
    return result

def expand_to_adjacent(frames, width=1):
    return np.hstack([shift_frames(frames, delta) for delta in range(-width,width+1)])

def rolling_aggregates(frames, width=1, aggregate=np.mean):
    # compute per feature
    result = np.empty(frames.shape)
    shifts = range(-width, width+1)
    for feature_index in range(result.shape[1]):
        feature_col = frames[:,feature_index][:,np.newaxis]
        windows = np.hstack([shift_frames(feature_col, delta) for delta in shifts])
        result[:, feature_index] = aggregate(windows, axis=1)
    return result

def apply_windowing(frames):
    extended = expand_to_adjacent(frames)
    window = np.hanning(extended.shape[1])
    return extended * window[np.newaxis,:]

def compute_spectra(windowed, window_length_secs):
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    frequency = np.arange(spectrum.shape[1]) / window_length_secs
    audible = (frequency > 20) & (frequency < 20000)

    spectrum = spectrum[:, audible]
    return spectrum

def compute_banks(spectra, n_banks):
    bank_size = int(spectra.shape[1] / n_banks)
    banks = []
    for i in range(n_banks):
        begin = i*bank_size
        end = (i+1)*bank_size
        bank = spectra[:,begin:end]
        banks.append(np.log1p(np.sqrt(np.sum(bank**2, axis=1))[:,np.newaxis]))
    return np.hstack(banks)

def split_to_chunks(n, chunk_size):
    i_begin = 0
    while i_begin < n:
        i_end = min(n, i_begin+chunk_size)
        yield(slice(i_begin, i_end))
        i_begin = i_end

def maybe_parallel_starmap(f, data, n_processes=1):
    if n_processes > 1:
        from multiprocessing import Pool
        with Pool(n_processes) as p:
            return p.starmap(f, data)
    else:
        return [f(*c) for c in data]

def compute_chunk_features(sound_data_chunk, frame_size, frame_secs):
    training_x = split_to_frames(sound_data_chunk, frame_size)
    spectra = compute_spectra(apply_windowing(training_x), 3*frame_secs)
    return compute_banks(spectra, n_banks=50)

def compute(sound_data, subvec, sample_rate, n_processes=3):
    frame_size = int(frame_secs*sample_rate)

    # compute features in chunks
    chunk_size_secs = 120
    chunk_size = int(chunk_size_secs / frame_secs)
    chunks = list(split_to_chunks(len(sound_data), chunk_size))

    def compute_chunk_labels(chunk):
        return np.round(np.mean(split_to_frames(subvec[chunk], frame_size), axis=1))

    data_chunks = [(sound_data[c], frame_size, frame_secs) for c in chunks]
    all_x = np.vstack(maybe_parallel_starmap(compute_chunk_features, data_chunks, n_processes))
    all_y = np.hstack([compute_chunk_labels(c) for c in chunks])

    return all_x, all_y

def read_training_data(index_file):
    import csv
    base_path = os.path.dirname(index_file)
    locate = lambda f: os.path.join(base_path, f)
    file_number = 0
    with open(index_file) as index:
        for item in csv.DictReader(index):
            file_number += 1
            sound_data, sub_vec, sample_rate = preprocessing.import_item(locate(item['sound']), locate(item['subtitles']))
            yield(sound_data, sub_vec, sample_rate, item['language'], file_number)

def compute_table(index_file):
    all_x = []
    all_y = []
    all_numbers = []
    all_languages = []
    for sound_data, subvec, sample_rate, language, file_number in read_training_data(index_file):
        print('file %d' % file_number)
        training_x, training_y = compute(sound_data, subvec, sample_rate)
        all_x.append(training_x)
        all_y.extend(training_y)
        all_numbers.extend([file_number]*len(training_y))
        all_languages.extend([language]*len(training_y))

    import pandas as pd
    meta = pd.DataFrame(np.array([all_y, all_numbers, all_languages]).T,
                        columns=['label', 'file_number', 'language'])

    return np.vstack(all_x), meta

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--index_file', default='training-data/index.csv')
    p.add_argument('--meta_output_file', default='training-data/meta.csv')
    p.add_argument('--features_output_file', default='training-data/features.npy')
    args = p.parse_args()

    data_x, data_meta = compute_table(args.index_file)
    data_meta.to_csv(args.meta_output_file)
    np.save(args.features_output_file, data_x)
