# Automatic subtitle synchronization tool

[![Build Status](https://travis-ci.org/oseiskar/autosubsync.svg?branch=master)](https://travis-ci.org/oseiskar/autosubsync)
[![PyPI](https://img.shields.io/pypi/v/autosubsync.svg)](https://pypi.python.org/pypi/autosubsync)

Did you know that hundreds of movies, especially from the 1950s and '60s,
are now in public domain and available online? Great! Let's download
_Plan 9 from Outer Space_. As a non-native English speaker, I prefer watching
movies with subtitles, which can also be found online for free. However, sometimes
there is a problem: the subtitles are not in sync with the movie.

But fear not. This tool can resynchronize the subtitles without any human input.
A correction for both shift and playing speed can be found automatically...
[using "AI & machine learning"](#methods)

## Installation

### macOS / OSX
Prerequisites: Install [Homebrew](https://brew.sh/) and [pip](https://stackoverflow.com/questions/17271319/how-do-i-install-pip-on-macos-or-os-x). Then install FFmpeg and this package

```
brew install ffmpeg
pip install autosubsync
```

### Linux (Debian & Ubuntu)

Make sure you have Pip, e.g., `sudo apt-get install python-pip`.
Then install [FFmpeg](https://www.ffmpeg.org/) and this package
```
sudo apt-get install ffmpeg
sudo pip install autosubsync
```

Note: If you are running Ubuntu 14 (but not 12 and 16, which are fine), you'll need to [jump some more hoops to install FFmpeg](https://www.faqforge.com/linux/how-to-install-ffmpeg-on-ubuntu-14-04/).

## Usage

```
autosubsync [input movie] [input subtitles] [output subs]

# for example
autosubsync plan-9-from-outer-space.avi \
  plan-9-out-of-sync-subs.srt \
  plan-9-subtitles-synced.srt
```
See `autosubsync --help` for more details.

## Features

 * Automatic speed and shift correction
 * Typical synchronization accuracy ~0.15 seconds (see [performance](#performance))
 * Wide video format support through [ffmpeg](https://www.ffmpeg.org/)
 * Supports all reasonably encoded SRT files in any language
 * Should work with any language in the audio (only tested with a few though)
 * Quality-of-fit metric for checking sync success
 * Python API. Example (save as `batch_sync.py`):

    ```python
    "Batch synchronize video files in a folder: python batch_sync.py /path/to/folder"

    import autosubsync
    import glob, os, sys

    if __name__ == '__main__':
        for video_file in glob.glob(os.path.join(sys.argv[1], '*.mp4')):
            base = video_file.rpartition('.')[0]
            srt_file = base + '.srt'
            synced_srt_file = base + '_synced.srt'

            # see help(autosubsync.synchronize) for more details
            autosubsync.synchronize(video_file, srt_file, synced_srt_file)
    ```

## Development

### Training the model

 1. Collect a bunch of well-synchronized video and subtitle files and put them
    in a file called `training/sources.csv` (see `training/sources.csv.example`)
 2. Run (and see) `train_and_test.sh`. This
    * populates the `training/data` folder
    * creates `trained-model.bin`
    * runs cross-validation

### Synchronization (predict)

Assumes trained model is available as `trained-model.bin`

    python3 autosubsync/main.py input-video-file input-subs.srt synced-subs.srt

### Build and distribution

 * Create virtualenv: `python3 -m venv venvs/test-python3`
 * Activate venv: `source venvs/test-python3/bin/activate`
 * `pip install -e .`
 * `pip install wheel`
 * `python setup.py bdist_wheel`

## Methods

The basic idea is to first detect speech on the audio track, that is, for each
point in time, _t_, in the film, to estimate if speech is heard. The method
[described below](#speech-detection) produces this estimate as a probability
of speech _p(t)_.
Another input to the program is the unsynchronized subtitle file containing the
timestamps of the actual subtitle intervals.

Synchronization is done by finding a time transformation _t_ → _f(t)_ that
makes _s(f(t))_, the synchronized subtitles, best [match](#loss-function),
_p(t)_, the detected speech. Here _s(t)_  is the (unsynchronized) subtitle
indicator function whose value is 1 if any subtitles are visible at time _t_
and 0 otherwise.

### Speech detection (VAD)

[Speech detection][4] is done by first computing a [spectrogram][2] of the audio,
that is, a matrix of features, where each column corresponds to a frame of
duration _Δt_ and each row a certain frequency band. Additional features are
engineered by computing a rolling maximum of the spectrogram with a few
different periods.

Using a collection of correctly synchronized media files, one can create a
training data set, where the each feature column is associated with a correct
label. This allows training a machine learning model to predict the labels, that
is, detect speech, on any previously unseen audio track - as the probability of
speech _p_(_iΔt_) on frame number _i_.

The weapon of choice in this project is [logistic regression][3], a common
baseline method in machine learning, which is simple to implement.
The accuracy of speech detection achieved with this model is not very good, only
around 72% (AURoC). However, the speech detection results are not the final
output of this program but just an input to the synchronization parameter
search. As mentioned in the [performance](#performance) section, the overall
_synchronization accuracy_ is quite fine even though the speech detection
is not.

### Synchronization parameter search

This program only searches for linear transformations of the form
_f_(_t_) = _a t + b_, where _b_ is shift and _a_ is speed correction.
The optimization method is brute force grid search where _b_ is limited to a
certain range and _a_ is one of the [common skew factors](#speed-correction).
The parameters minimizing the loss function are selected.

### Loss function

The data produced by the speech detection phase is a vector representing the
speech probabilities in frames of duration _Δt_. The metric used for evaluating
match quality is expected linear loss:

&nbsp; &nbsp; loss(_f_) = Σ<sub>_i_</sub> _s_(_f<sub>i</sub>_)
(1 - _p<sub>i</sub>_) + (1 - _s_(_f<sub>i</sub>_)) _p<sub>i</sub>_,

where _p<sub>i</sub>_ = _p_(_iΔt_) is the probability of speech and
_s_(_f<sub>i</sub>_) = _s_(_f_(_iΔt_)) = _s_(_a iΔt + b_) is the subtitle
indicator resynchronized using the transformation _f_ at frame number _i_.

### Speed correction

Speed/skew detection is based on the assumption that an error in playing speed
is not an arbitrary number but caused by frame rate mismatch, which constraints
the possible playing speed multiplier to be ratio of two common frame rates
sufficiently close to one. In particular, it must be one of the following values

 * 24/23.976 = 30/29.97 = 60/59.94 = 1001/1000
 * 25/24
 * 25/23.976

or the reciprocal (1/x).

The reasoning behind this is that if the frame rate of (digital) video footage
needs to be changed and the target and source frame rates are close enough,
the conversion is often done by skipping any re-sampling and just changing the
nominal frame rate. This effectively changes the playing speed of the video
and the pitch of the audio by a small factor which is the ratio
of these frame rates.

### Performance

Based on somewhat limited testing, the typical shift error in auto-synchronization
seems to be around 0.15 seconds (cross-validation RMSE) and generally below 0.5
seconds. In other words, it seems to work well enough in most cases but could be
better. [Speed correction](#speed-correction) errors did not occur.

Auto-syncing a full-length movie currently takes about 3 minutes and utilizes
around 1.5 GB of RAM.

## References

I first checked Google if someone had already tried to solve the same problem and found
[this great blog post][1] whose author had implemented a solution using more or less the
same approach that I had in mind. The post also included good points that I had not realized,
such as using correctly synchronized subtitles as training data for speech detection.

Instead of starting from the code linked in that blog post I decided to implement my
own version from scratch, since this might have been a good application for trying out
RNNs, which turned out to be unnecessary, but this was a nice project nevertheless.

  [1]: https://albertosabater.github.io/Automatic-Subtitle-Synchronization/
  [2]: https://en.wikipedia.org/wiki/Spectrogram
  [3]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  [4]: https://en.wikipedia.org/wiki/Voice_activity_detection

### Other similar projects

 * https://github.com/tympanix/subsync Apparently based on the blog post above, looks good
 * https://github.com/smacke/subsync Newer project, uses WebRTC VAD
    (instead of DIY machine learning) for speech detection
 * https://github.com/Koenkk/PyAMC/blob/master/autosubsync.py
 * https://github.com/pulasthi7/AutoSubSync-old & https://github.com/pulasthi7/AutoSubSync (looks inactive)
