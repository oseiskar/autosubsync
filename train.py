import numpy as np
import features
import preprocessing
import model

data_x, data_meta = features.compute_table('training-data/index.csv')
trained = model.train(data_x, data_meta.label)

with open('trained-model.bin', 'wb') as f:
    f.write(model.serialize(trained))