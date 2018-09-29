import numpy as np
from .features import frame_secs

def find_max_window(shift_scores, window_half_size_secs = 2.0):
    wnd = int(np.ceil(window_half_size_secs/frame_secs))
    best = np.argmax(shift_scores)
    low_bound = max(0, best-wnd)
    high_bound = min(best+wnd+1, len(shift_scores))

    return low_bound, best, high_bound

def find_peak(shift_scores, **kwargs):
    low_bound, best, high_bound = find_max_window(shift_scores, **kwargs)

    before = shift_scores[low_bound:best+1]
    after = shift_scores[best:high_bound]

    threshold = 0.2

    low = min(np.min(before), np.min(after))
    high = before[-1]
    limit = low + (high-low)*threshold

    end = np.argmax(after < limit)
    if end == 0: end = -1

    begin = np.argmax(before[::-1] < limit)
    if begin == 0: begin = len(before)
    begin = len(before)-begin

    before_peak = shift_scores[:low_bound+begin]
    peak_up = before[begin:]
    peak_down = after[:end]
    after_peak = shift_scores[best+end:]
    return peak_up, peak_down, before_peak, after_peak

def metric_peak_monotonicity(shift_scores, **kwargs):
    before, after = find_peak(shift_scores, **kwargs)[:2]
    def mean_raising(vec):
        if len(vec) < 2: return 0
        return np.mean(np.diff(vec) > 0)

    return min(mean_raising(before), mean_raising(after[::-1]))

def metric_non_edgeness(shift_scores, **kwargs):
    "How close to the data edge is the peak? if further than window_half_size_secs, then 1"
    low_bound, best, high_bound = find_max_window(shift_scores, **kwargs)
    return min(high_bound-best, best-low_bound)/float(max(high_bound-best, best-low_bound))

def metric_peak_prominence(shift_scores, **kwargs):
    before, after, before_peak, after_peak = find_peak(shift_scores, **kwargs)

    def one_sided(peak, others):
        if len(peak) == 0: return 0
        if len(others) == 0: return 1.0
        peak_high = np.max(peak)
        peak_low = np.min(peak)
        peak_height = peak_high - peak_low
        if peak_height <= 0:
            return 0
        return min(1.0 - (np.max(others) - peak_low) / float(peak_height), 1.0)

    return min(one_sided(before, before_peak), one_sided(after, after_peak))

compute_quality = lambda x: metric_peak_monotonicity(x) * metric_peak_prominence(x) * metric_non_edgeness(x, window_half_size_secs=0.5)

# found by cross-validation, TODO: don't hard-code
# highest seen false fit: 0.7901189719371496, p99: 0.6274763843859568
# lowest valid fit: 0.8409090909090909
threshold = 0.75
