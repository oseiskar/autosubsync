import numpy as np
from sklearn import metrics

from features import frame_secs

def sub_score(y_true, y_probs, shift=0):
    y_shift = np.roll(y_true, shift)
    if shift != 0:
        y_shift[:np.abs(shift)] = 0
        y_shift[-np.abs(shift):] = 0
    
    return metrics.roc_auc_score(y_shift, y_probs)

def best_shift(y_subs, y_probs, max_shift_secs=2.0):
    max_shift = int(max_shift_secs/frame_secs)+1
    shifts = range(-max_shift, max_shift)
    scores = [sub_score(y_subs, y_probs, shift) for shift in shifts]
    return shifts[np.argmax(scores)]*frame_secs