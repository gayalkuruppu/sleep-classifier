import numpy as np

class MyDecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return np.mean(y)

        num_features = X.shape[1]
        best_feature, best_threshold = None, None
        best_mse = float('inf')

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                left_mse = np.mean((y[left_indices] - np.mean(y[left_indices]))**2)
                right_mse = np.mean((y[right_indices] - np.mean(y[right_indices]))**2)

                mse = left_mse + right_mse
                if mse < best_mse:
                    best_feature, best_threshold = feature, threshold
                    best_mse = mse

        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices, :], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if isinstance(tree, (float, np.float64)):  # Leaf node
            return tree

        feature, threshold, left_tree, right_tree = tree
        if x[feature] <= threshold:
            return self._predict_tree(x, left_tree)
        else:
            return self._predict_tree(x, right_tree)
