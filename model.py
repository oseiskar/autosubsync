import numpy as np
import features

def transform(data_x):
    return np.hstack([
        features.expand_to_adjacent(data_x, width=1),
        features.rolling_aggregates(data_x, width=2, aggregate=np.max),
        features.rolling_aggregates(data_x, width=5, aggregate=np.max)
    ])

def normalize(data_x):
    #from sklearn.preprocessing import StandardScaler
    #return StandardScaler().fit_transform(data_x)
    #return data_x - np.mean(data_x, axis=0)
    return data_x

def normalize_by_file(data_x, file_labels=None):
    if file_labels is None:
        return normalize(data_x)

    result_x = np.empty(data_x.shape)
    for label in np.unique(file_labels):
        part = file_labels == label
        result_x[part, :] = normalize(data_x[part])
    return result_x

def train(training_x, training_y, training_file_labels=None):
    from sklearn.linear_model import LogisticRegression as classifier
    #from sklearn.ensemble import GradientBoostingClassifier as classifier

    model = classifier(penalty='l1', C=0.001)
    model.fit(transform(normalize_by_file(training_x, training_file_labels)), training_y)
    return model

def predict(model, test_x, file_labels=None):
    return model.predict_proba(transform(normalize_by_file(test_x, file_labels)))[:,1]

def serialize(model):
    import pickle
    return pickle.dumps(model)

def deserialize(data):
    import pickle
    return pickle.loads(data)

def load(model_file):
    with open(model_file, 'rb') as f:
        return deserialize(f.read())

def save(trained_model, target_file):
    with open(target_file, 'wb') as f:
        f.write(serialize(trained_model))
