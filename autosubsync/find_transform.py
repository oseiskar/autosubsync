import numpy as np

from .features import frame_secs, maybe_parallel_map
from . import quality_of_fit

def score_function(labels, probs):
    "Score for binary labels vs probabilistic predictions"
    # Computes "expected accuracy" of labels vs predicted, where the
    # components of predicted are independent and Bernoulli distributed
    # with probabilities given by probs. Has a similar effect than using
    # sklearn.roc_auc_score but this is faster to compute
    labels = labels == 1
    return (np.sum(probs[labels]) + np.sum(1.0 - probs[~labels]))/float(len(labels))

def sub_score_transform(y_true, y_probs, func):
    y_true = np.array(list(y_true))
    n = len(y_true)
    shifted_indices = np.round(func(np.arange(n)*frame_secs)/frame_secs).astype(np.int)
    y_shift = y_true*0
    valid_indices = (shifted_indices >= 0) & (shifted_indices < n)
    y_shift[shifted_indices[valid_indices]] = y_true[valid_indices]

    missed_fraction = np.sum(y_true[~valid_indices])
    penalty_factor = 1.0 - missed_fraction / float(n)

    return score_function(y_shift, y_probs) * penalty_factor

def sub_score(y_true, y_probs, shift=0, skew=1.0):
    return sub_score_transform(y_true, y_probs, lambda x: x*skew + shift*frame_secs)

def compute_shift_scores(y_subs, y_probs, max_shift_secs=20.0, skew=1.0, base_shift_secs=0.0):
    min_shift = int((base_shift_secs - max_shift_secs)/frame_secs)
    max_shift = int((base_shift_secs + max_shift_secs)/frame_secs)+1
    shifts = range(min_shift, max_shift)
    scores = [sub_score(y_subs, y_probs, shift, skew) for shift in shifts]
    return shifts, scores

def best_shift(*args, **kwargs):
    shifts, scores = compute_shift_scores(*args, **kwargs)
    quality = quality_of_fit.compute_quality(scores)
    best_idx = np.argmax(scores)
    return [shifts[best_idx]*frame_secs, scores[best_idx], quality]

def _best_shift_star(args):
    return best_shift(*args)

def get_skew_pairs(frame_rates, fixed_skew=None):
    if fixed_skew is not None:
        return [fixed_skew], [str(fixed_skew)]

    skew_pairs = [[a,b] for a in frame_rates for b in frame_rates]
    skew_pairs.insert(0, [1,1])
    skew_pairs = np.array(skew_pairs)
    skews = skew_pairs[:,0]/skew_pairs[:,1]
    uniq_idx = np.unique(skews, return_index=True)[1]
    return skews[uniq_idx], ['%g/%g' % (skew_pairs[i,0], skew_pairs[i,1]) for i in uniq_idx]

def find_transform_parameters(y_subs, y_probs, max_shift_secs=20.0, frame_rates=[23.976, 24, 25], bias=0, fixed_skew=None, verbose=False, parallelism=3):
    skews, skew_labels = get_skew_pairs(frame_rates, fixed_skew=fixed_skew)
    if verbose:
        print('max shift %gs, test increments %gs' % (max_shift_secs, frame_secs))
        print('testing with skews: ' + ', '.join(skew_labels))
        print('bias', bias)

    shift_score_quality = np.array(maybe_parallel_map( \
        _best_shift_star, \
        [(y_subs, y_probs, max_shift_secs, skew) for skew in skews], \
        parallelism))

    if verbose:
        print('shift\tscore\tquality\tskew')
        for i in range(len(shift_score_quality)):
            print('\t'.join(["%.3g" % s for s in shift_score_quality[i]]) + '\t' + skew_labels[i])

    best_idx = np.argmax(shift_score_quality[:,1])

    shift = shift_score_quality[best_idx,0] + bias
    skew = skews[best_idx]
    quality = shift_score_quality[best_idx,2]

    if verbose:
        skew_label = skew_labels[best_idx]
        print('optimal shift: %g seconds, skew: %s' % (shift, skew_label))

    return skew, shift, quality

def parameters_to_transform(skew, shift):
    return lambda x: x * skew + shift

def find_transform(*args, **kwargs):
    skew, shift, quality = find_transform_parameters(*args, **kwargs)
    return parameters_to_transform(skew, shift), quality
