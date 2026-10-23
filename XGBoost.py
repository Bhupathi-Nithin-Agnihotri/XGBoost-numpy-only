import numpy as np

# Hardcoded hyperparameters for the initial implementation
lr = 0.1
gamma = 0.0
Lambda = 1.0
max_depth = 3
min_child_weight = 1.0
n_estimators = 100


class Node:
    def __init__(self, value=None, feature=None, threshold=None,
                 left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.feature is None


class XGBoostScratch:
    def __init__(self):
        self.trees = []
        self.base_score = 0.0

    def _build_tree(self, X, grad, hess, depth):
        G = np.sum(grad)
        H = np.sum(hess)

        # Optimal prediction for this node if it remains a leaf
        leaf_value = -G / (H + Lambda)

        # Stop condn for growth
        if depth >= max_depth or len(grad) < 2:
            return Node(value=leaf_value)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        # Try every feature and every valid threshold
        for feature in range(n_features):
            order = np.argsort(X[:, feature])
            x_sorted = X[order, feature]
            g_sorted = grad[order]
            h_sorted = hess[order]

            G_prefix = np.cumsum(g_sorted)
            H_prefix = np.cumsum(h_sorted)

            for i in range(n_samples - 1):
                # Equal feature values cannot be separated
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                GL = G_prefix[i]
                HL = H_prefix[i]
                GR = G - GL
                HR = H - HL

                if HL < min_child_weight or HR < min_child_weight:
                    continue

                gain = 0.5 * (
                    GL**2 / (HL + Lambda)
                    + GR**2 / (HR + Lambda)
                    - G**2 / (H + Lambda)
                ) - gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (
                        x_sorted[i] + x_sorted[i + 1]
                    ) / 2.0

        # No beneficial split exists
        if best_feature is None:
            return Node(value=leaf_value)

        mask = X[:, best_feature] <= best_threshold

        left = self._build_tree(
            X[mask], grad[mask], hess[mask], depth + 1
        )
        right = self._build_tree(
            X[~mask], grad[~mask], hess[~mask], depth + 1
        )

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left,
            right=right
        )

    def _predict_tree_row(self, node, x):
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _predict_tree(self, tree, X):
        return np.array([
            self._predict_tree_row(tree, row) for row in X
        ])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # Initial prediction for squared-error regression
        self.base_score = np.mean(y)
        pred = np.full(len(y), self.base_score)

        self.trees = []

        for iteration in range(n_estimators):
            # First and second derivatives of  the obj (0.5 * (y - pred)^2)
            grad = pred - y
            hess = np.ones_like(grad)

            tree = self._build_tree(X, grad, hess, depth=0)
            update = self._predict_tree(tree, X)

            pred += lr * update
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        pred = np.full(len(X), self.base_score)

        for tree in self.trees:
            pred += lr * self._predict_tree(tree, X)

        return pred