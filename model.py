from sklearn.linear_model import LogisticRegression
import features

def transform(data_x):
    return features.expand_to_adjacent(data_x, width=2)

def train(training_x, training_y):
    model = LogisticRegression()
    model.fit(transform(training_x), training_y)
    return model

def predict(model, test_x):
    return model.predict_proba(transform(test_x))[:,1]

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
