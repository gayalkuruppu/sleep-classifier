import numpy as np
from source.algorithms.MyDecisionTreeRegressor import MyDecisionTreeRegressor

class MyGradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, response_method='predict'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.response_method = response_method
        self.estimators = []
        self.classes = None

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
        }

    def fit(self, X, y):
        self.classes = np.unique(y)

        for class_label in self.classes:
            # Convert multiclass problem to binary problem (one vs. rest)
            binary_y = (y == class_label).astype(int)
            weak_classifier = MyDecisionTreeRegressor(max_depth=self.max_depth)
            residuals = binary_y.copy()

            for _ in range(self.n_estimators):
                weak_classifier.fit(X, residuals)
                prediction = weak_classifier.predict(X)
                residuals -= self.learning_rate * prediction.astype('int64')

                self.estimators.append((class_label, weak_classifier))

    def predict(self, X):
        # Initialize predictions for each class
        class_predictions = {class_label: np.zeros(len(X)) for class_label in self.classes}

        for class_label, weak_classifier in self.estimators:
            binary_prediction = weak_classifier.predict(X)
            class_predictions[class_label] += self.learning_rate * binary_prediction.astype('int64')

        if self.response_method == 'predict_proba':
            # For multiclass, convert to probability
            exp_predictions = np.exp(list(class_predictions.values()))
            proba_predictions = exp_predictions / np.sum(exp_predictions, axis=0)
            return proba_predictions.T
        elif self.response_method == 'predict':
            # For multiclass, convert to class labels
            return np.array([class_label for class_label in self.classes[np.argmax(list(class_predictions.values()), axis=0)]])
        else:
            raise ValueError("Invalid response_method. Should be 'predict_proba' or 'predict'.")
