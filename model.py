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

def train(training_x, training_y, training_meta):
    from sklearn.linear_model import LogisticRegression as classifier
    #from sklearn.ensemble import GradientBoostingClassifier as classifier

    file_labels = training_meta.file_number.values

    #selected_part = features.balance_file_lengths(file_labels)
    #selected_part = features.balance_by_group(training_meta.language, file_labels)
    #training_x = training_x[selected_part, :]
    #training_y = training_y[selected_part]
    #file_labels = file_labels[selected_part]

    training_weights = features.weight_by_group_and_file(training_meta.language, file_labels)
    #training_weights = features.weight_by_group(file_labels)

    print(np.unique(training_weights))
    training_x = features.normalize_by_file(training_x, normalize, file_labels)
    training_x = transform(training_x)

    model = classifier(penalty='l1', C=0.001)
    model.fit(training_x, training_y, sample_weight=training_weights)
    return model

def predict(model, test_x, file_labels=None):
    test_x = features.normalize_by_file(test_x, normalize, file_labels)
    test_x = transform(test_x)
    return model.predict_proba(test_x)[:,1]

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
