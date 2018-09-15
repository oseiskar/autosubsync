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
        result_x[part, :] = normalize(data_x[part, :])
    return result_x

def balance_file_lengths(data_x, data_y, file_labels=None):
    if file_labels is None:
        raise RuntimeError("must have file labels to balance by file")

    unique_labels = np.unique(file_labels)
    lengths = [np.sum(file_labels == label) for label in unique_labels]
    selected_length = int(np.median(lengths))

    print(lengths)
    print(selected_length)

    result_xs = []
    result_ys = []
    result_labels = []

    for label in unique_labels:
        part = file_labels == label
        selected = part & (np.cumsum(part) <= selected_length)

        result_xs.append(data_x[selected, :])
        result_ys.append(np.ravel(data_y[selected]))
        result_labels.append(np.ravel(file_labels[selected]))

    return np.vstack(result_xs), np.hstack(result_ys), np.hstack(result_labels)


def train(training_x, training_y, training_file_labels=None):
    from sklearn.linear_model import LogisticRegression as classifier
    #from sklearn.ensemble import GradientBoostingClassifier as classifier

    training_x, training_y, training_file_labels = \
        balance_file_lengths(training_x, training_y, training_file_labels)

    training_x = normalize_by_file(training_x, training_file_labels)
    training_x = transform(training_x)

    model = classifier(penalty='l1', C=0.001)
    model.fit(training_x, training_y)
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
