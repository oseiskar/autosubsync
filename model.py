from sklearn.linear_model import LogisticRegression    
    
def train(training_x, training_y):
    model = LogisticRegression()
    model.fit(training_x, training_y)
    return model
    
def predict(model, test_x):
    return model.predict_proba(test_x)[:,1]

def serialize(model):
    import pickle
    return pickle.dumps(model)

def deserialize(data):
    import pickle
    return pickle.loads(data)