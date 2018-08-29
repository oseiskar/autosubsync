import numpy as np
import features
import preprocessing
import model

if __name__ == '__main__':
    data_x, data_meta = features.compute_table('training-data/index.csv')
    print('training data extracted, shape ' + str(data_x.shape))
    
    print('training...')
    trained = model.train(data_x, data_meta.label)
    
    target_file = 'trained-model.bin'
    print('serializing model to ' + target_file)

    with open(target_file, 'wb') as f:
        f.write(model.serialize(trained))