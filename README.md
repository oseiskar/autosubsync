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
[using AI & machine learning](#references)

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
 * Wide video format support through [ffmpeg](https://www.ffmpeg.org/)
 * Supports all reasonably encoded SRT files in any language
 * Should work with any language in the audio (only tested with a few though)
 * Quality-of-fit metric for checking sync success
 * Python API

         import autosubsync
         autosubsync.syncrhonize("movie.avi", "subs.srt", "synced.srt")

         # see help(autosubsync.syncrhonize) for more details

## Performance

Based on somewhat limited testing, the typical shift error in auto-synchronization
seems to be around 0.15 seconds (cross-validation RMSE) and generally below 0.5
seconds. In other words, it seems to work well enough in most cases but could be
better. [Speed correction](#speed-correction) errors did not occur.

Auto-syncing a full-length movie currently takes about 3 minutes and utilizes
around 1.5 GB of RAM.

## Development

### Training the model

 1. Collect a bunch of well-synchronized video and subtitle files and put them
    in a file called `training/sources.csv` (see `training/sources.csv.example`)
 2. Run (and see) `train_and_test.sh`. This
    * populates the `training/data` folder
    * creates `trained-model.bin`
    * runs cross-validation

### Syncronization (predict)

Assumes trained model is available as `trained-model.bin`

    python3 autosubsync/main.py input-video-file input-subs.srt synced-subs.srt

### Build and distribution

 * Create virtualenv: `python3 -m venv venvs/test-python3`
 * Activate venv: `source venvs/test-python3/bin/activate`
 * `pip install -e .`
 * `pip install wheel`
 * `python setup.py bdist_wheel`

## Methods

### Speed correction

Speed/skew detection is based on the assumption that an error in playing speed
is not an arbitrary number but caused by frame rate mismatch, which constraints
the possible playing speed multiplier to be ratio of two common frame rates
sufficiently close to one. In particular, it must be one of the following values

 * 24/23.976 = 30/29.97 = 60/59.94 = 1001/1000
 * 25/24
 * 25/23.976

or the reciprocial (1/x).

The reasoning behind this is that if the frame rate of (digital) video footage
needs to be changed and the target and source frame rates are close enough,
the conversion is often done by skipping any resampling and just changing the
nominal frame rate. This effectively changes the playing speed of the video
and the pitch of the audio by a small factor which is the ratio
of these frame rates.

## References

I first checked Google if someone had already tried to solve the same problem and found
[this great blog post](https://albertosabater.github.io/Automatic-Subtitle-Synchronization/)
whose author had implemented a solution using more or less the same approach that
I had in mind. The post also included good points that I had not realized, such as
using correctly synchronized subtitles as training data for speech detection.

Instead of starting from the code linked in that blog post I decided to implement my
own version from scratch, since this might have been a good application for trying out
RNNs, which turned out to be unnecessary, but this was a nice project nevertheless.

### Other similar projects

 * https://github.com/tympanix/subsync Apparently based on the blog post above, looks good
 * https://github.com/Koenkk/PyAMC/blob/master/autosubsync.py
 * https://github.com/pulasthi7/AutoSubSync-old & https://github.com/pulasthi7/AutoSubSync (looks inactive)
