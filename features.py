import numpy as np

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
    


