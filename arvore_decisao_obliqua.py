from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np


@dataclass
class No:
    prediction: int
    n_samples: int
    impurity: float
    w_star: Optional[np.ndarray] = None
    tau_star: Optional[float] = None
    left: Optional["No"] = None
    right: Optional["No"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None or self.right is None


class ObliqueDecisionTree:
    def __init__(
        self,
        max_depth: int = 12,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        n_projections: int = 40,
        max_thresholds: int = 128,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_projections = n_projections
        self.max_thresholds = max_thresholds
        self.rng = np.random.default_rng(random_state)
        self.root: Optional[No] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ObliqueDecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.unique(y)
        self.n_classes_ = int(np.max(y)) + 1

        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_[self.feature_std_ == 0.0] = 1.0
        X_scaled = self._scale(X)

        self.root = self._cresce_arvore(X_scaled, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("A arvore precisa ser treinada antes de prever.")

        X = np.asarray(X, dtype=float)
        X_scaled = self._scale(X)
        return np.array([self._prediz_linha(row) for row in X_scaled], dtype=int)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean_) / self.feature_std_

    def _prediz_linha(self, row: np.ndarray) -> int:
        node = self.root
        while node is not None and not node.is_leaf:
            z = float(row @ node.w_star)
            if z <= node.tau_star:
                node = node.left
            else:
                node = node.right
        return int(node.prediction)

    def _cresce_arvore(self, X: np.ndarray, y: np.ndarray, depth: int) -> No:
        prediction = classe_mais_frequente(y)
        impurity = gini(y, self.n_classes_)
        node = No(prediction=prediction, n_samples=len(y), impurity=impurity)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or impurity == 0.0
        ):
            return node

        w_star, tau_star, gain = self._melhor_corte(X, y, impurity)
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
        node.left = self._cresce_arvore(X[left_mask], y[left_mask], depth + 1)
        node.right = self._cresce_arvore(X[right_mask], y[right_mask], depth + 1)
        return node

    def _melhor_corte(
        self, X: np.ndarray, y: np.ndarray, parent_impurity: float
    ) -> tuple[Optional[np.ndarray], Optional[float], float]:
        best_w = None
        best_threshold = None
        best_gain = 0.0
        n_samples, n_features = X.shape

        for w in self._direcoes_candidatas(X, y, n_features):
            z = X @ w
            thresholds = limiares_candidatos(z, self.max_thresholds)

            for th in thresholds:
                left_mask = z <= th
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]
                weighted_impurity = (
                    n_left * gini(y_left, self.n_classes_)
                    + n_right * gini(y_right, self.n_classes_)
                ) / n_samples
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_w = w
                    best_threshold = float(th)
                    best_gain = float(gain)

        return best_w, best_threshold, best_gain

    def _direcoes_candidatas(
        self, X: np.ndarray, y: np.ndarray, n_features: int
    ) -> list[np.ndarray]:
        projections = []

        for _ in range(self.n_projections):
            projections.append(self._random_projection(n_features))

        classes = np.unique(y)
        class_means = {klass: X[y == klass].mean(axis=0) for klass in classes}
        global_mean = X.mean(axis=0)

        for klass in classes:
            direction = class_means[klass] - global_mean
            normalized = normaliza(direction)
            if normalized is not None:
                projections.append(normalized)

        for index, klass_a in enumerate(classes):
            for klass_b in classes[index + 1 :]:
                direction = class_means[klass_a] - class_means[klass_b]
                normalized = normaliza(direction)
                if normalized is not None:
                    projections.append(normalized)

        return projections

    def _random_projection(self, n_features: int) -> np.ndarray:
        w = self.rng.normal(loc=0.0, scale=1.0, size=n_features)
        normalized = normaliza(w)
        if normalized is None:
            w = np.zeros(n_features)
            w[0] = 1.0
            return w
        return normalized


def limiares_candidatos(values: np.ndarray, max_thresholds: int) -> np.ndarray:
    unique_values = np.unique(values)
    if len(unique_values) <= 1:
        return np.array([], dtype=float)

    midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
    if len(midpoints) <= max_thresholds:
        return midpoints

    positions = np.linspace(0, len(midpoints) - 1, max_thresholds).astype(int)
    return np.unique(midpoints[positions])


def normaliza(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return None
    return vector / norm


def gini(y: np.ndarray, n_classes: int) -> float:
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities * probabilities))


def classe_mais_frequente(y: np.ndarray) -> int:
    counts = np.bincount(y)
    return int(np.argmax(counts))


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


def acuracia(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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

    model = ObliqueDecisionTree(
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        n_projections=60,
        max_thresholds=160,
        random_state=42,
    )

    start = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - start

    start = perf_counter()
    y_val_pred = model.predict(X_val)
    predict_time = perf_counter() - start

    accuracy = acuracia(y_val, y_val_pred)
    print(f"Acuracia de validacao: {accuracy:.4f}")
    print(f"Tempo de treino: {fit_time:.3f} s")
    print(f"Tempo de predicao na validacao: {predict_time:.3f} s")
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    print(matriz_confusao(y_val, y_val_pred))

    final_model = ObliqueDecisionTree(
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        n_projections=60,
        max_thresholds=160,
        random_state=42,
    )
    final_model.fit(X_train_full, y_train_full)
    test_predictions = final_model.predict(X_test)
    save_predictions("predicoes_arvore_obliqua.csv", test_predictions)
    print("Predicoes salvas em predicoes_arvore_obliqua.csv")


if __name__ == "__main__":
    main()
