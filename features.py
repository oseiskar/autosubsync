import numpy as np
import sys

frame_secs = 0.05

def split_to_frames(data, sample_rate, frame_length_sec):
    frame_size = int(frame_length_sec*sample_rate)
    n_frames = int(len(data)/frame_size)
    data = data[:(n_frames*frame_size)]
    return np.reshape(data, (n_frames, frame_size))

def expand_to_adjacent(frames):
    before = frames * 0
    before[1:,:] = frames[:-1,:]

    after = frames * 0
    after[:-1,:] = frames[1:,:]

    return np.hstack((before, frames, after))

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

def compute(sound_data, subvec, sample_rate):
    training_x = split_to_frames(sound_data, sample_rate, frame_secs)
    training_y = np.round(np.mean(split_to_frames(subvec, sample_rate, frame_secs), axis=1))

    spectra = compute_spectra(apply_windowing(training_x), 3*frame_secs)
    features = compute_banks(spectra, n_banks=50)

    return features, training_y

def compute_table(index_file):
    import preprocessing
    all_x = []
    all_y = []
    all_numbers = []
    all_languages = []
    for sound_data, subvec, sample_rate, language, file_number in preprocessing.read_training_data(index_file):
        print('computing features for file %d' % file_number)
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
