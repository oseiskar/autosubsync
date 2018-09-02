import numpy as np

from features import frame_secs
import quality_of_fit

def sub_score_transform(y_true, y_probs, func):
    y_true = np.array(list(y_true))
    n = len(y_true)
    shifted_indices = np.round(func(np.arange(n)*frame_secs)/frame_secs).astype(np.int)
    y_shift = y_true*0
    valid_indices = (shifted_indices >= 0) & (shifted_indices < n)
    y_shift[shifted_indices[valid_indices]] = y_true[valid_indices]
    
    return np.mean(np.round(y_shift) == np.round(y_probs)) * np.mean(valid_indices)

def sub_score(y_true, y_probs, shift=0, skew=1.0):
    return sub_score_transform(y_true, y_probs, lambda x: x*skew + shift*frame_secs)

def best_shift(y_subs, y_probs, max_shift_secs=2.0, skew=1.0):
    max_shift = int(max_shift_secs/frame_secs)+1
    shifts = range(-max_shift, max_shift)
    scores = [sub_score(y_subs, y_probs, shift, skew) for shift in shifts]
    quality = quality_of_fit.compute_quality(scores)
    best_idx = np.argmax(scores)
    return shifts[best_idx]*frame_secs, scores[best_idx], quality

def get_skew_pairs(frame_rates):
    skew_pairs = [[a,b] for a in frame_rates for b in frame_rates]
    skew_pairs.insert(0, [1,1])
    skew_pairs = np.array(skew_pairs)
    skews = skew_pairs[:,0]/skew_pairs[:,1]
    uniq_idx = np.unique(skews, return_index=True)[1]
    return skews[uniq_idx], ['%g/%g' % (skew_pairs[i,0], skew_pairs[i,1]) for i in uniq_idx]

def find_transform(y_subs, y_probs, max_shift_secs=10.0, frame_rates=[23.976, 24, 25], verbose=False):
    skews, skew_labels = get_skew_pairs(frame_rates)
    if verbose:
        print('max shift %gs, test increments %gs' % (max_shift_secs, frame_secs))
        print('testing with skews: ' + ', '.join(skew_labels))

    shift_score_quality = np.array([list(best_shift(y_subs, y_probs, max_shift_secs, skew)) for skew in skews])
    if verbose:
        print(shift_score_quality)
        
    best_idx = np.argmax(shift_score_quality[:,1])
    
    shift = shift_score_quality[best_idx,0]
    skew = skews[best_idx]
    quality = shift_score_quality[best_idx,2]
        
    if verbose:
        skew_label = skew_labels[best_idx]
        print('optimal shift: %g seconds, skew: %s' % (shift, skew_label))
    
    return lambda x: x * skew + shift, quality