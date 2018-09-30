import numpy as np

class TrainedLogisticRegression:
    """
    Pre-trained logistic regression model using numpy (but not sklearn).
    The idea of this class is to avoid the run-time dependency to sklearn
    """

    def __init__(self, coefficients, bias):
        """
        Logistic regression model for a dependent variable y of two classes
        (y = 0 or y = 1) is defined as

        :: math

            logit(p) = log(p / (1 - p)) = b_0 + b_1 x_1 + ... + b_n x_n

        where p is the probability of y = 1 given predictor values
        x_1, ..., x_n

        Args:
            coefficients (list-like): array of coefficients b_1, ..., b_n
            bias (float): scalar intercept / bias b_0
        """
        self.coefficients = list(np.ravel(coefficients))
        self.bias = bias

    def predict_proba(self, data_x):
        """
        Compute probabilies of 0 and 1 classes given a matrix of predictors.
        Mimics the corresponding method in the scikit learn classs
        ``sklearn.linear_model.LogisticRegression``

        Args:
            data_x (np.array): matrix of predictors, each row is an observation

        Returns:
            numpy array ``x`` where ``x[:, 0]`` and ``x[:, 1]`` are the
            predictors for classes
        """
        #logit = lambda p: np.log(p/(1-p))
        logistic = lambda a: 1 / (1 + np.exp(-a))
        probs = logistic(np.dot(data_x, self.coefficients) + self.bias)
        probs = probs[:, np.newaxis]
        return np.hstack([1.0 - probs, probs])

    @staticmethod
    def from_sklearn(sklearn_model):
        """
        Convert an sklearn trained logistic regression model

        Args:
            sklearn_model (sklearn.linear_model.LogisticRegression): trained model

        Returns:
            the corresponding TrainedLogisticRegression object
        """
        return TrainedLogisticRegression(\
            sklearn_model.coef_,
            sklearn_model.intercept_[0])

    @staticmethod
    def from_dict(data):
        "Deserialize Python dictionary to a TrainedLogisticRegression object"
        return TrainedLogisticRegression(data['coef'], data['bias'])

    def to_dict(self):
        "Serialize TrainedLogisticRegression to Python dictionary"
        return { 'coef': self.coefficients, 'bias': self.bias }
