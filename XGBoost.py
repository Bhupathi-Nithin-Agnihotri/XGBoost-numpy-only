import numpy as np

# Hardcoded hyperparameters for the initial implementation
lr = 0.1
gamma = 0.0
Lambda = 1.0
max_depth = 3
min_child_weight = 1.0
n_estimators = 100


class Node:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
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

        # Try every feature and evaluate all thresholds using vectorized NumPy
        for feature in range(n_features):
            order = np.argsort(X[:, feature])
            x_sorted = X[order, feature]
            g_sorted = grad[order]
            h_sorted = hess[order]

            G_prefix = np.cumsum(g_sorted)
            H_prefix = np.cumsum(h_sorted)

            # Candidate positions where adjacent feature values differ
            positions = np.flatnonzero(x_sorted[:-1] != x_sorted[1:])

            if positions.size == 0:
                continue

            GL = G_prefix[positions]
            HL = H_prefix[positions]
            GR = G - GL
            HR = H - HL

            # Enforce minimum child weight
            valid = (HL >= min_child_weight) & (HR >= min_child_weight)

            if not np.any(valid):
                continue

            # Calculate split gains for all valid thresholds at once
            gains = (
                0.5 * (
                    GL**2 / (HL + Lambda)
                    + GR**2 / (HR + Lambda)
                    - G**2 / (H + Lambda)
                )
                - gamma
            )

            gains[~valid] = -np.inf

            best_idx = np.argmax(gains)
            gain = gains[best_idx]

            if gain > best_gain:
                pos = positions[best_idx]
                best_gain = gain
                best_feature = feature
                best_threshold = (
                    x_sorted[pos] + x_sorted[pos + 1]
                ) / 2.0

        # No beneficial split exists
        if best_feature is None:
            return Node(value=leaf_value)

        mask = X[:, best_feature] <= best_threshold

        left = self._build_tree(X[mask], grad[mask], hess[mask], depth + 1)
        right = self._build_tree(X[~mask], grad[~mask], hess[~mask], depth + 1)

        return Node(
            feature=best_feature, threshold=best_threshold, left=left, right=right
        )

    def _predict_tree(self, tree, X):
        X = np.asarray(X, dtype=float)
        predictions = np.empty(X.shape[0], dtype=float)
        stack = [(tree, np.arange(X.shape[0]))]

        while stack:
            node, indices = stack.pop()

            if indices.size == 0:
                continue

            if node.is_leaf():
                predictions[indices] = node.value
                continue

            go_left = X[indices, node.feature] <= node.threshold

            stack.append((node.right, indices[~go_left]))
            stack.append((node.left, indices[go_left]))

        return predictions

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
