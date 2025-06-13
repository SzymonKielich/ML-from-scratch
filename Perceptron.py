from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=None, tol=1e-5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.tol = tol
        self.error_history = []

    def fit(self, X_train, y_train):

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = np.random.randn(X_train.shape[1]) * 0.01
        self.bias = np.random.randn() * 0.01

        for it in range(self.max_iter):
            errors = 0

            old_weights = np.copy(self.weights)
            old_bias = self.bias

            for idx, x_i in enumerate(X_train):
                output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.heaviside_function(output)

                if y_predicted != y_train[idx]:
                    update = self.learning_rate * (y_train[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1

            self.error_history.append(errors)

            if np.sum(np.abs(self.weights - old_weights)) < self.tol and np.abs(self.bias - old_bias) < self.tol:
                print("Przerwano. Wartości wag i biasu nie zmieniły się znacząco dla iteracji", iter)
                break

    def heaviside_function(self, output):
        return np.where(output >= 0, 1, 0)

    def predict(self, X_test):
        output = np.dot(X_test, self.weights) + self.bias
        y_predicted = self.heaviside_function(output)
        return y_predicted


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


def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


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


if __name__ == "__main__":
    data = pd.read_csv('data_banknote_authentication.txt', header=None,
                       names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

    print(data.head())
    print("\nInformacje o zbiorze:")
    print(data.info())
    print("\nStatystyki opisowe:")
    print(data.describe())
    print("\nRozkład klas:")
    print(data['class'].value_counts())

    print("\nmin i max przed normalizacją:")
    for col in data.columns[:-1]:
        print(f"{col}: min={data[col].min():.4f}, max={data[col].max():.4f}")
    X = data.drop('class', axis=1).values

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data.drop('class', axis=1))
    plt.title('Rozkład cech')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('cechy_boxplot.png')
    plt.show()

    y = data['class'].values
    X_normalized = standard_scaler(X)
    normalized_df = pd.DataFrame(X_normalized, columns=data.columns[:-1])
    normalized_df['class'] = y

    print("\nmin i max po normalizacji:")
    for col in normalized_df.columns[:-1]:
        print(f"{col}: min = {normalized_df[col].min()}, max = {normalized_df[col].max()}")

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(data.columns[:-1], 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=data, x=col, hue="class", kde=True)
        plt.title(f'Rozkład cechy {col}')
        plt.ylabel('Częstość')
    plt.tight_layout()
    plt.savefig('cechy_czestosc.png')
    plt.show()

    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X)

    tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2', 'TSNE3'])

    tsne_df['klasa'] = y
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], tsne_df['TSNE3'],
                         c=tsne_df['klasa'])
    ax.set_title('Wizualizacja 3D TSNE')
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')
    ax.set_zlabel('TSNE 3')
    ax.legend(*scatter.legend_elements(), title="Klasy")
    plt.tight_layout()
    plt.savefig('tsne_3d.png')
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_normalized, y, test_size=0.2,
                                                                            random_state=42)
    print("\n---- Perceptron przed standaryzacja ----")
    perc = Perceptron()
    perc.fit(X_train, y_train)
    y_pred = perc.predict(X_test)

    class_names = ['Fałszywy', 'Prawdziwy']
    print(f"Dokładność: {accuracy(y_test, y_pred):.4f}")
    print(f"Precyzja (macro): {precision(y_test, y_pred, 'macro'):.4f}")
    print(f"Precyzja (micro): {precision(y_test, y_pred, 'micro'):.4f}")
    print(f"Recall (macro): {recall(y_test, y_pred, 'macro'):.4f}")
    print(f"Recall (micro): {recall(y_test, y_pred, 'micro'):.4f}")
    print(f"F1 (macro): {f1_score(y_test, y_pred, 'macro'):.4f}")
    print(f"F1 (micro): {f1_score(y_test, y_pred, 'micro'):.4f}")
    plot_confusion_matrix(y_test, y_pred, class_names)

    print("\n--- Perceptron po standaryzacji ----")
    perc_norm = Perceptron()
    perc_norm.fit(X_train_norm, y_train_norm)
    y_pred_norm = perc_norm.predict(X_test_norm)

    print(f"Dokładność: {accuracy(y_test_norm, y_pred_norm):.4f}")
    print(f"Precyzja (macro): {precision(y_test_norm, y_pred_norm, 'macro'):.4f}")
    print(f"Precyzja (micro): {precision(y_test_norm, y_pred_norm, 'micro'):.4f}")
    print(f"Recall (macro): {recall(y_test_norm, y_pred_norm, 'macro'):.4f}")
    print(f"Recall (micro): {recall(y_test_norm, y_pred_norm, 'micro'):.4f}")
    print(f"F1 (macro): {f1_score(y_test_norm, y_pred_norm, 'macro'):.4f}")
    print(f"F1 (micro): {f1_score(y_test_norm, y_pred_norm, 'micro'):.4f}")
    plot_confusion_matrix(y_test_norm, y_pred_norm, class_names)

    learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 30.0]
    results = []

    for lr in learning_rates:
        perc_lr = Perceptron(learning_rate=lr, max_iter=30, tol=0.00001)
        perc_lr.fit(X_train, y_train)
        y_pred_lr = perc_lr.predict(X_test)

        acc = accuracy(y_test, y_pred_lr)
        prec_macro = precision(y_test_norm, y_pred_lr, 'macro')
        prec_micro = precision(y_test_norm, y_pred_lr, 'micro')
        rec_macro = recall(y_test_norm, y_pred_lr, 'macro')
        rec_micro = recall(y_test_norm, y_pred_lr, 'micro')
        f1_macro = f1_score(y_test_norm, y_pred_lr, 'macro')
        f1_micro = f1_score(y_test_norm, y_pred_lr, 'micro')
        total_errors = perc_lr.error_history[-1]
        iterations = len(perc_lr.error_history)

        results.append({
            'learning_rate': lr,
            'accuracy': acc,
            'precision_macro': prec_macro,
            'precision_micro': prec_micro,
            'recall_macro': rec_macro,
            'recall_micro': rec_micro,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'total_errors': total_errors,
            'iterations': iterations,
            'error_history': perc_lr.error_history
        })

        print(f"Learning rate: {lr:.4f}, Accuracy: {acc:.4f}, F1 macro: {f1_macro:.4f}, "
              f"Iteracje: {iterations}, Błędy: {total_errors}")

    lr_values = [r['learning_rate'] for r in results]
    acc_values = [r['accuracy'] for r in results]
    f1_macro_values = [r['f1_macro'] for r in results]
    f1_micro_values = [r['f1_micro'] for r in results]
    total_errors = [r['total_errors'] for r in results]
    iterations = [r['iterations'] for r in results]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(lr_values, acc_values, 'o-', label='Accuracy')
    plt.plot(lr_values, f1_macro_values, 's-', label='F1 macro')
    plt.plot(lr_values, f1_micro_values, '^-', label='F1 micro')
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Wartość metryki')
    plt.title('Wpływ współczynnika uczenia na metryki')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(lr_values, iterations, 'o-', color='green')
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Liczba iteracji')
    plt.title('Wpływ współczynnika uczenia na liczbę iteracji')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.bar(range(len(lr_values)), total_errors)
    plt.xticks(range(len(lr_values)), [str(lr) for lr in lr_values], rotation=45)
    plt.xlabel('Learning rate')
    plt.ylabel('Liczba błędów')
    plt.title('Całkowita liczba błędów dla różnych współczynników uczenia')

    plt.subplot(2, 2, 4)
    for i, lr in enumerate(learning_rates):
        plt.plot(results[i]['error_history'], label=f'LR = {lr}')
    plt.xlabel('Iteracja')
    plt.ylabel('Liczba błędów')
    plt.title('Historia błędów dla różnych learning_rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('analiza_lr.png')
    plt.show()

    model_3d = Perceptron(learning_rate=0.01, max_iter=20, random_state=42)
    model_3d.fit(X_tsne, y)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                         c=y, alpha=0.8)

    x_min, x_max = X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1
    y_min, y_max = X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1
    z_min, z_max = X_tsne[:, 2].min() - 1, X_tsne[:, 2].max() + 1

    step = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    # wyliczenie granicy decyzyjnej z rownania hiperplaszczyzny
    zz = (-model_3d.weights[0] * xx - model_3d.weights[1] * yy - model_3d.bias) / model_3d.weights[2]
    zz = np.clip(zz, z_min, z_max)
    ax.plot_surface(xx, yy, zz, alpha=0.3)

    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    ax.set_zlabel('TSNE3')
    ax.set_title('Granica decyzyjna 3D TSNE')
    legend = ax.legend(*scatter.legend_elements(), title="Klasy")
    ax.add_artist(legend)
    plt.tight_layout()
    plt.savefig('tsne_3d_granica_decyzyjna.png')
    plt.show()
