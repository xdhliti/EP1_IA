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
    i_star: Optional[int] = None
    tau_star: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None or self.right is None


class OrthogonalDecisionTree:
    """
    Arvore de decisao ortogonal implementada do zero com NumPy.

    Cada no guarda (i*, tau*) e testa uma unica feature:
    X[:, i*] <= tau*.
    O melhor split e escolhido pelo maior ganho de impureza de Gini.
    """

    def __init__(
        self,
        max_depth: int = 12,
        min_samples_split: int = 8,
        min_samples_leaf: int = 4,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrthogonalDecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.n_classes_ = int(np.max(y)) + 1
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("A arvore precisa ser treinada antes de prever.")

        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row) for row in X], dtype=int)

    def _predict_one(self, row: np.ndarray) -> int:
        node = self.root
        while node is not None and not node.is_leaf:
            if row[node.i_star] <= node.tau_star:
                node = node.left
            else:
                node = node.right
        return int(node.prediction)

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        prediction = majority_class(y)
        impurity = gini(y, self.n_classes_)
        node = Node(prediction=prediction, n_samples=len(y), impurity=impurity)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or impurity == 0.0
        ):
            return node

        i_star, tau_star, gain = self._best_split(X, y, impurity)
        if i_star is None or gain <= 0.0:
            return node

        left_mask = X[:, i_star] <= tau_star
        right_mask = ~left_mask

        if (
            np.sum(left_mask) < self.min_samples_leaf
            or np.sum(right_mask) < self.min_samples_leaf
        ):
            return node

        node.i_star = i_star
        node.tau_star = tau_star
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, parent_impurity: float
    ) -> tuple[Optional[int], Optional[float], float]:
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        n_samples, n_features = X.shape

        # Mesmo formato do enunciado:
        # for j in range(m):
        #     thresholds = unique(X[:, j])
        #     for th in thresholds:
        for j in range(n_features):
            values = X[:, j]
            thresholds = np.unique(values)

            for th in thresholds:
                left_mask = values <= th
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
                    best_feature = j
                    best_threshold = float(th)
                    best_gain = float(gain)

        return best_feature, best_threshold, best_gain


def gini(y: np.ndarray, n_classes: int) -> float:
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities * probabilities))


def majority_class(y: np.ndarray) -> int:
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

    model = OrthogonalDecisionTree(
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=4,
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
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    print(confusion_matrix(y_val, y_val_pred))

    final_model = OrthogonalDecisionTree(
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=4,
    )
    final_model.fit(X_train_full, y_train_full)
    test_predictions = final_model.predict(X_test)
    save_predictions("predicoes_arvore_ortogonal.csv", test_predictions)
    print("Predicoes salvas em predicoes_arvore_ortogonal.csv")


if __name__ == "__main__":
    main()
