#!/bin/bash
set -eux
PYTHON=python3
$PYTHON training/build_training_data.py # populate the folder training/data/
$PYTHON training/train.py --compute_features # force recomputation of features
$PYTHON training/cross_validate.py
