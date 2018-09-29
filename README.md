# Automatic subtitle synchronization tool

Did you know that hundreds of movies, especially from the 1950s and '60s,
are now in public domain and available online? Great! Let's download
_Plan 9 from Outer Space_. As a non-native English speaker, I prefer watching
movies with subtitles, which can also be found online for free. However, sometimes
there is a problem: the subtitles are not in sync with the movie.

But fear not. This tool can resynchronize the subtitles without any human input.
A correction for both shift and playing speed can be found automatically.

## Features

 * Automatic speed and shift correction
 * Wide video format support through [ffmpeg](https://www.ffmpeg.org/)
 * Supports all reasonably encoded SRT files
 * Quality-of-fit metric for checking sync success

## Requirements

 * Python 3
 * ffmpeg

## Development

### Training the model

 1. Collect a bunch of well-synchronized video and subtitle files and put them
    in a file called `training/sources.csv` (see `training/sources.csv.example`)
 2. Run `python3 training/build_training_data.py` to populate the folder `training/data/`
 3. Run `python3 training/train.py`

### Syncronization (predict)

Assumes trained model is available as `trained-model.bin`

    python3 autosubsync/main.py input-video-file input-subs.srt synced-subs.srt

### Cross-validation

    python3 training/cross_validate.py

### Build and distribution

 * Create virtualenv: `python3 -m venv venvs/test-python3`
 * Activate venv: `source venvs/test-python3/bin/activate`
 * `pip install -e .`
 * `pip install wheel`
 * `python setup.py bdist_wheel`

## References

I first checked Google if someone had already tried to solve the same problem and found
[this great blog post](https://albertosabater.github.io/Automatic-Subtitle-Synchronization/)
whose author had implemented a solution using more or less the same approach that
I had in mind. The post also included good points that I had not realized, such as
using correctly synchronized subtitles as training data for speech detection.

Instead of starting from the code linked in that blog post I decided to implement my
own version from scratch, since this might have been a good application for trying out
RNNs, which turned out to be unnecessary, but this was a nice project nevertheless.
