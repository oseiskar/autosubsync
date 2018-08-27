# Automatic subtitle synchronization tool

### Requirements

 * Python 3
 * ffmpeg

### Training the model

 1. Collect a bunch of well-synchronized video and subtitle files and put them
    in a file called `training-sources.csv` (see `training-sources.csv.example`)
 2. Run `python3 build_training_data.py` to populate the folder `training-data/`
 3. Run `python3 train.py`

### Conversion

Assumes trained model is available as `trained-model.bin`

    python3 main.py input-video-file input-subs.srt synced-subs.srt

### References

Very similar to https://albertosabater.github.io/Automatic-Subtitle-Synchronization/
