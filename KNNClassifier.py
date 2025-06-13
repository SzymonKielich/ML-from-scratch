import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.manifold import TSNE


class KNNClassifier:
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p
        self.y_train = None
        self.X_train = None

    def fit(self, X_train, y_train):
        """Zapisanie danych modelu klasyfikatora"""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        """Przyjmuje dane testowe i zwraca etykiety"""
        y_pred = [self.predict_one(x) for x in np.asarray(X_test)]
        return np.array(y_pred)

    def predict_one(self, x):
        distances = np.array([self.minkowski_distance(x, x_t) for x_t in self.X_train])
        k_indexes = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indexes]
        counts = np.bincount(k_nearest_labels)
        return np.argmax(counts)

    def minkowski_distance(self, x1, x2):
        return np.power(np.sum(np.power(np.abs(x1 - x2), self.p)), 1 / self.p)


def run(X_train, X_test, y_train, y_test, class_names, k=3, p=2):
    knn = KNNClassifier(k=k, p=p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Precision (Micro): {precision(y_test, y_pred, average='micro'):.4f}")
    print(f"Precision (Macro): {precision(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (Micro): {recall(y_test, y_pred, average='micro'):.4f}")
    print(f"Recall (Macro): {recall(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score (Micro): {f1_score(y_test, y_pred, average='micro'):.4f}")
    print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

    plot_confusion_matrix(y_test, y_pred, class_names)


def confusion_matrix(y_true, y_pred, labels=None):
    """Implementacja macierzy pomyłek

    dokumentacja sklearn:
    -- Thus in binary classification, the count of true negatives is C(0,0),
    false negatives is C(1,0),
    true positives is C(1,1)
    and false positives is C(0,1).

    -- Returns ndarray of shape (n_classes, n_classes) Confusion matrix
    whose i-th row and j-th column entry indicates the number of samples with
    true label being i-th class and predicted label being j-th class.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = max(np.max(y_true), np.max(y_pred))

    cmatrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    for i in range(len(y_true)):
        cmatrix[y_true[i], y_pred[i]] += 1

    return cmatrix


def plot_confusion_matrix(y_true, y_pred, class_names):
    """ Wizualizuje macierz pomyłek. """
    cmatrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names,
                     yticklabels=class_names)

    ax.set(xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    plt.tight_layout()
    plt.show()


def classification_report():
    """Wywolanie wszystkich metryk."""
    pass


def accuracy(y_true, y_pred):
    """Oblicza dokładność klasyfikacji."""
    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred, average='macro'):
    """Oblicza precyzję. Obsługuje micro/macro averaging."""
    cmatrix = confusion_matrix(y_true, y_pred)
    if average == 'micro':
        tp_sum = 0
        tp_fp_sum = 0
        for c in range(len(cmatrix)):
            tp_sum += cmatrix[c, c]
            tp_fp_sum += np.sum(cmatrix[:, c])
        return tp_sum / tp_fp_sum if tp_fp_sum > 0 else 0
    else:  # macro averaging
        precisions = []
        for class_idx in range(len(cmatrix)):
            tp = cmatrix[class_idx, class_idx]
            fp = np.sum(cmatrix[:, class_idx]) - tp
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(class_precision)
        return np.mean(precisions)


def recall(y_true, y_pred, average='macro'):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    classes = np.unique(np.concatenate((y_true, y_pred)))
    cmatrix = confusion_matrix(y_true, y_pred)

    if average == 'micro':
        tp_sum = 0
        tp_fn_sum = 0
        for c in classes:
            tp_sum += cmatrix[c, c]
            tp_fn_sum += np.sum(cmatrix[c, :])
        return tp_sum / tp_fn_sum if tp_fn_sum > 0 else 0
    else:
        recalls = []
        for c in classes:
            tp = cmatrix[c, c]
            fn = np.sum(cmatrix[c, :]) - tp
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(class_recall)
        return np.mean(recalls)


def f1_score(y_true, y_pred, average='macro'):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""
    prec = precision(y_true, y_pred, average=average)
    rec = recall(y_true, y_pred, average=average)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)


def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def cross_validation(X, y, k, p, folds=5):
    n_samples = len(X)
    indexes = np.arange(n_samples)
    np.random.shuffle(indexes)
    fold_sizes = np.full(folds, n_samples // folds, dtype=int)
    fold_sizes[:n_samples % folds] += 1
    current = 0
    scores = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indexes[start:stop]
        train_idx = np.concatenate([indexes[:start], indexes[stop:]])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = KNNClassifier(k=k, p=p).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy(y_test, y_pred))
        current = stop
    return np.mean(scores)


if __name__ == "__main__":
    np.random.seed()
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    print(y)

    class_names = data.target_names
    print(f"Liczba próbek: {X.shape[0]}")
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Liczba klas: {len(np.unique(y))}")
    print("\nRozkład klas:")
    print(y.value_counts())
    print(data)
    train_size = 0.8
    train_idx = X.sample(frac=train_size, random_state=50).index
    X_train, X_test = X.loc[train_idx], X.drop(train_idx)
    y_train, y_test = y.loc[train_idx], y.drop(train_idx)

    print("\n---- Wyniki przed standaryzacją ----")
    run(X_train, X_test, y_train, y_test, class_names, k=5, p=2)

    X_train_scaled = standard_scaler(X_train)
    X_test_scaled = standard_scaler(X_test)

    print("\n---- Wyniki po standaryzacji -----")
    run(X_train_scaled, X_test_scaled, y_train, y_test, class_names, k=5, p=2)
    print("Przykładowe dane przed normalizacją:")
    print(X.head())

    X_norm = pd.DataFrame(standard_scaler(X.values), columns=X.columns)
    print("\nPrzykładowe dane po normalizacji:")
    print(X_norm.head())

    train_idx = X_norm.sample(frac=train_size, random_state=50).index
    X_train = X_norm.loc[train_idx].values
    X_test = X_norm.drop(train_idx).values
    y_train = y.loc[train_idx].values
    y_test = y.drop(train_idx).values

    print("\nOptymalizacja hiperparametru k przy użyciu walidacji krzyżowej")

    k_range = range(1, 16)
    cv_scores = []
    for k in k_range:
        score = cross_validation(X_train, y_train, k=k, p=2, folds=5)
        cv_scores.append(score)
    errors = [1 - score for score in cv_scores]

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, errors, marker='o')
    plt.xlabel('Liczba sąsiadów - k')
    plt.ylabel('Błąd klasyfikacji (1 - accuracy)')
    plt.title('Wykres zależności błędu klasyfikacji od k')
    plt.grid(True)
    plt.show()

    best_k = k_range[np.argmin(errors)]
    print(f"\nNajlepsza wartość k: {best_k} (minimalny błąd: {min(errors):.4f})")

    print("\nOptymalizacja hiperparametru p przy użyciu walidacji krzyżowej")
    cv_p_scores = []
    p_values = [1, 1.25, 1.5, 1.75, 2, 3, 4, 6]
    for p in p_values:
        score = cross_validation(X_train, y_train, k=best_k, p=p, folds=5)
        cv_p_scores.append(score)
        print(f"p = {p}: accuracy = {score:.4f}")
    errors_p = [1 - score for score in cv_p_scores]

    plt.figure(figsize=(8, 5))
    plt.plot(p_values, errors_p, marker='s', color='r')
    plt.xlabel('Wartość parametru p')
    plt.ylabel('Błąd klasyfikacji (1 - accuracy)')
    plt.title('Wykres zależności błędu klasyfikacji od p')
    plt.grid(True)
    plt.show()

    best_p = p_values[np.argmin(errors_p)]
    print(f"\nNajlepsza wartość p: {best_p} (minimalny błąd: {min(errors_p):.4f})")

    best_model = KNNClassifier(k=best_k, p=best_p)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\nDokładność modelu dla najlepszych parametrów: {accuracy(y_test, y_pred):.4f}, gdzie k={best_k}, p={best_p}")

    print("Wizualizacja z wykorzystaniem TSNE")
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(8, 6))
    for class_label in np.unique(y):
        plt.scatter(X_embedded[y == class_label, 0], X_embedded[y == class_label, 1], label=f'Class {class_label}')

    plt.legend()
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.title("Wizualizacja danych przy użyciu TSNE")
    plt.show()
