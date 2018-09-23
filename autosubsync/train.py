import numpy as np
import pandas as pd
import model

if __name__ == '__main__':
    data_x = np.load('training-data/features.npy')
    data_meta = pd.read_csv('training-data/meta.csv', index_col=0)

    print('training data extracted, shape ' + str(data_x.shape))

    print('training...')
    trained = model.train(data_x, data_meta.label.values, data_meta, verbose=True)

    target_file = 'trained-model.bin'
    print('serializing model to ' + target_file)

    model.save(trained, target_file)
