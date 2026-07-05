from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np


@dataclass
class Node:
    prediction: int
    n_samples: int
    impurity: float
    w_star: Optional[np.ndarray] = None
    tau_star: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None or self.right is None


class WeightedObliqueDecisionTree:
    """
    Arvore obliqua com pesos nas amostras.

    Ela e usada como classificador base do AdaBoost. Cada no escolhe um
    hiperplano local z = X @ w e um limiar tau usando ganho de Gini ponderado.
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        n_projections: int = 80,
        max_thresholds: int = 256,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_projections = n_projections
        self.max_thresholds = max_thresholds
        self.rng = np.random.default_rng(random_state)
        self.root: Optional[Node] = None
        self.n_classes_: int = 0

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
    ) -> "WeightedObliqueDecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        sample_weight = np.asarray(sample_weight, dtype=float)
        self.n_classes_ = int(np.max(y)) + 1
        self.root = self._grow(X, y, sample_weight, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("A arvore precisa ser treinada antes de prever.")

        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row) for row in X], dtype=int)

    def _predict_one(self, row: np.ndarray) -> int:
        node = self.root
        while node is not None and not node.is_leaf:
            z = float(row @ node.w_star)
            if z <= node.tau_star:
                node = node.left
            else:
                node = node.right
        return int(node.prediction)

    def _grow(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, depth: int
    ) -> Node:
        prediction = weighted_majority_class(y, sample_weight, self.n_classes_)
        impurity = weighted_gini(y, sample_weight, self.n_classes_)
        node = Node(prediction=prediction, n_samples=len(y), impurity=impurity)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or impurity == 0.0
        ):
            return node

        w_star, tau_star, gain = self._best_split(X, y, sample_weight, impurity)
        if w_star is None or gain <= 0.0:
            return node

        z = X @ w_star
        left_mask = z <= tau_star
        right_mask = ~left_mask

        if (
            np.sum(left_mask) < self.min_samples_leaf
            or np.sum(right_mask) < self.min_samples_leaf
        ):
            return node

        node.w_star = w_star
        node.tau_star = tau_star
        node.left = self._grow(
            X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1
        )
        node.right = self._grow(
            X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1
        )
        return node

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        parent_impurity: float,
    ) -> tuple[Optional[np.ndarray], Optional[float], float]:
        best_w = None
        best_threshold = None
        best_gain = 0.0
        _, n_features = X.shape

        for w in self._candidate_projections(X, y, sample_weight, n_features):
            z = X @ w
            threshold, gain = best_threshold_for_projection(
                z,
                y,
                sample_weight,
                parent_impurity,
                self.n_classes_,
                self.min_samples_leaf,
                self.max_thresholds,
            )

            if threshold is not None and gain > best_gain:
                best_w = w
                best_threshold = threshold
                best_gain = gain

        return best_w, best_threshold, best_gain

    def _candidate_projections(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        n_features: int,
    ) -> list[np.ndarray]:
        projections = [self._random_projection(n_features) for _ in range(self.n_projections)]

        classes = np.unique(y)
        global_mean = weighted_mean(X, sample_weight)
        class_means = {}

        for klass in classes:
            mask = y == klass
            class_means[klass] = weighted_mean(X[mask], sample_weight[mask])

        for klass in classes:
            normalized = normalize(class_means[klass] - global_mean)
            if normalized is not None:
                projections.append(normalized)

        for index, klass_a in enumerate(classes):
            for klass_b in classes[index + 1 :]:
                normalized = normalize(class_means[klass_a] - class_means[klass_b])
                if normalized is not None:
                    projections.append(normalized)

        return projections

    def _random_projection(self, n_features: int) -> np.ndarray:
        w = self.rng.normal(loc=0.0, scale=1.0, size=n_features)
        normalized = normalize(w)
        if normalized is None:
            w = np.zeros(n_features)
            w[0] = 1.0
            return w
        return normalized


class SAMMEObliqueAdaBoost:
    """
    AdaBoost multiclasse SAMME com arvores obliquas.

    A cada iteracao, uma arvore obliqua e treinada com pesos nas amostras.
    Exemplos errados recebem peso maior para a proxima arvore.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.4,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        n_projections: int = 100,
        max_thresholds: int = 256,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_projections = n_projections
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.estimators: list[WeightedObliqueDecisionTree] = []
        self.estimator_weights: list[float] = []
        self.n_classes_: int = 0
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAMMEObliqueAdaBoost":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = int(np.max(y)) + 1
        self.estimators = []
        self.estimator_weights = []

        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_[self.feature_std_ == 0.0] = 1.0
        X_scaled = self._scale(X)

        n_samples = len(y)
        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=float)

        for _ in range(self.n_estimators):
            tree_seed = int(self.rng.integers(0, 2**31 - 1))
            tree = WeightedObliqueDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_projections=self.n_projections,
                max_thresholds=self.max_thresholds,
                random_state=tree_seed,
            )
            tree.fit(X_scaled, y, sample_weight)
            prediction = tree.predict(X_scaled)
            incorrect = prediction != y
            error = float(np.sum(sample_weight[incorrect]))

            max_allowed_error = 1.0 - (1.0 / self.n_classes_)
            if error <= 1e-12:
                alpha = 1.0
                self.estimators.append(tree)
                self.estimator_weights.append(alpha)
                break
            if error >= max_allowed_error:
                continue

            alpha = self.learning_rate * (
                np.log((1.0 - error) / error) + np.log(self.n_classes_ - 1)
            )
            sample_weight *= np.exp(alpha * incorrect)
            sample_weight /= np.sum(sample_weight)

            self.estimators.append(tree)
            self.estimator_weights.append(float(alpha))

        if not self.estimators:
            raise RuntimeError("AdaBoost nao conseguiu treinar nenhum estimador valido.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators:
            raise RuntimeError("O modelo precisa ser treinado antes de prever.")

        X = np.asarray(X, dtype=float)
        X_scaled = self._scale(X)
        scores = np.zeros((len(X_scaled), self.n_classes_), dtype=float)

        for alpha, tree in zip(self.estimator_weights, self.estimators):
            prediction = tree.predict(X_scaled)
            scores[np.arange(len(X_scaled)), prediction] += alpha

        return np.argmax(scores, axis=1).astype(int)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean_) / self.feature_std_


def best_threshold_for_projection(
    z: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    parent_impurity: float,
    n_classes: int,
    min_samples_leaf: int,
    max_thresholds: int,
) -> tuple[Optional[float], float]:
    order = np.argsort(z)
    z_sorted = z[order]
    y_sorted = y[order]
    weight_sorted = sample_weight[order]

    if z_sorted[0] == z_sorted[-1]:
        return None, 0.0

    class_weight_matrix = np.zeros((len(y_sorted), n_classes), dtype=float)
    class_weight_matrix[np.arange(len(y_sorted)), y_sorted] = weight_sorted
    cumulative_left = np.cumsum(class_weight_matrix, axis=0)
    total_counts = cumulative_left[-1]

    valid_positions = np.flatnonzero(z_sorted[:-1] < z_sorted[1:])
    valid_positions = valid_positions[
        (valid_positions + 1 >= min_samples_leaf)
        & (len(y_sorted) - valid_positions - 1 >= min_samples_leaf)
    ]
    if len(valid_positions) == 0:
        return None, 0.0

    if len(valid_positions) > max_thresholds:
        selected = np.linspace(0, len(valid_positions) - 1, max_thresholds).astype(int)
        valid_positions = valid_positions[selected]

    left_counts = cumulative_left[valid_positions]
    right_counts = total_counts - left_counts
    left_weight = left_counts.sum(axis=1)
    right_weight = right_counts.sum(axis=1)
    total_weight = left_weight + right_weight

    left_gini = 1.0 - np.sum((left_counts / left_weight[:, None]) ** 2, axis=1)
    right_gini = 1.0 - np.sum((right_counts / right_weight[:, None]) ** 2, axis=1)
    weighted_impurity = (
        left_weight * left_gini + right_weight * right_gini
    ) / total_weight
    gains = parent_impurity - weighted_impurity

    best_index = int(np.argmax(gains))
    best_position = int(valid_positions[best_index])
    threshold = float((z_sorted[best_position] + z_sorted[best_position + 1]) / 2.0)
    return threshold, float(gains[best_index])


def weighted_gini(y: np.ndarray, sample_weight: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(y, weights=sample_weight, minlength=n_classes)
    total = np.sum(counts)
    if total == 0.0:
        return 0.0
    probabilities = counts / total
    return float(1.0 - np.sum(probabilities * probabilities))


def weighted_majority_class(
    y: np.ndarray, sample_weight: np.ndarray, n_classes: int
) -> int:
    counts = np.bincount(y, weights=sample_weight, minlength=n_classes)
    return int(np.argmax(counts))


def weighted_mean(X: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    total_weight = np.sum(sample_weight)
    if total_weight == 0.0:
        return X.mean(axis=0)
    return (X * sample_weight[:, None]).sum(axis=0) / total_weight


def normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return None
    return vector / norm


def stratified_train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_indices = []
    validation_indices = []

    for klass in np.unique(y):
        klass_indices = np.flatnonzero(y == klass)
        rng.shuffle(klass_indices)
        n_validation = max(1, int(round(len(klass_indices) * validation_size)))
        validation_indices.extend(klass_indices[:n_validation])
        train_indices.extend(klass_indices[n_validation:])

    train_indices = np.array(train_indices)
    validation_indices = np.array(validation_indices)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        X[train_indices],
        X[validation_indices],
        y[train_indices],
        y[validation_indices],
    )


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    size = int(np.max(labels)) + 1
    matrix = np.zeros((size, size), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def load_data(path: str = "data.npz") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X_train"], data["y_train"], data["X_test"]


def save_predictions(path: str, predictions: np.ndarray) -> None:
    ids = np.arange(len(predictions))
    output = np.column_stack([ids, predictions])
    np.savetxt(
        path,
        output,
        delimiter=",",
        header="id,label",
        comments="",
        fmt="%d",
    )


def main() -> None:
    X_train_full, y_train_full, X_test = load_data("data.npz")
    X_train, X_val, y_train, y_val = stratified_train_validation_split(
        X_train_full,
        y_train_full,
        validation_size=0.2,
        random_state=42,
    )

    model = SAMMEObliqueAdaBoost(
        n_estimators=120,
        learning_rate=0.4,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        n_projections=100,
        max_thresholds=256,
        random_state=42,
    )

    start = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - start

    start = perf_counter()
    y_val_pred = model.predict(X_val)
    predict_time = perf_counter() - start

    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acuracia de validacao: {accuracy:.4f}")
    print(f"Tempo de treino: {fit_time:.3f} s")
    print(f"Tempo de predicao na validacao: {predict_time:.3f} s")
    print(f"Estimadores usados: {len(model.estimators)}")
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    print(confusion_matrix(y_val, y_val_pred))

    final_model = SAMMEObliqueAdaBoost(
        n_estimators=120,
        learning_rate=0.4,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        n_projections=100,
        max_thresholds=256,
        random_state=42,
    )
    final_model.fit(X_train_full, y_train_full)
    test_predictions = final_model.predict(X_test)
    save_predictions("predicoes_adaboost_arvore_obliqua.csv", test_predictions)
    print("Predicoes salvas em predicoes_adaboost_arvore_obliqua.csv")


if __name__ == "__main__":
    main()
