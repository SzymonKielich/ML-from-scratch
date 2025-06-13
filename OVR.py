from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


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

            # if np.sum(np.abs(self.weights - old_weights)) < self.tol and np.abs(self.bias - old_bias) < self.tol:
            #     print("Przerwano. Wartości wag i biasu nie zmieniły się znacząco dla iteracji", iter)
            #     break

    def heaviside_function(self, output):
        return np.where(output >= 0, 1, 0)

    def predict(self, X_test):
        output = self.predict_proba(X_test)
        y_predicted = self.heaviside_function(output)
        return y_predicted

    def predict_proba(self, X_test):
        output = np.dot(X_test, self.weights) + self.bias
        return output

class OVR:
    def __init__(self, classifier=Perceptron, **kwargs):
        self.classes = None
        self.classifier = classifier
        self.classifiers = []
        self.kwargs = kwargs


    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for target_class in self.classes:
            y_binary = np.where(y_train == target_class, 1, 0)
            classifier = self.classifier(**self.kwargs)
            classifier.fit(X_train, y_binary)
            self.classifiers.append((target_class, classifier))
        return self

    def predict(self, X_test):
        scores = []

        for target_class, classifier in self.classifiers:
            if isinstance(classifier, Perceptron):
                scores.append(classifier.predict_proba(X_test))
            elif isinstance(classifier, SVC):
                scores.append(classifier.predict_proba(X_test)[:, 1])
            else: scores.append(classifier.predict(X_test))

        scores = np.array(scores) # [3, samples]
        predictions = np.argmax(scores, axis=0)
        return self.classes[predictions]

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


def plot_confusion_matrix(y_true, y_pred, class_names, model_name='Confusion Matrix'):
    """ Wizualizuje macierz pomyłek. """
    cmatrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cmatrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names,
                     yticklabels=class_names)

    ax.set(xlabel='Predicted', ylabel='True', title=model_name)
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
    np.random.seed()
    data = load_wine()
    X = data.data
    y = data.target
    class_names = data.target_names
    print(f"Liczba próbek: {X.shape[0]}")
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Liczba klas: {len(np.unique(y))}")

    X_scaled = X

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    ovr_perceptron = OVR(classifier=Perceptron, learning_rate=0.01, max_iter=1000)
    ovr_perceptron.fit(X_train, y_train)

    ovr_svc = OVR(classifier=SVC, kernel='linear', probability=True)
    ovr_svc.fit(X_train, y_train)

    y_pred_perceptron = ovr_perceptron.predict(X_test)
    y_pred_svc = ovr_svc.predict(X_test)

    plot_confusion_matrix(y_test, y_pred_perceptron, class_names)

    plot_confusion_matrix(y_test, y_pred_svc, class_names)

    print("\n-- Perceptron OVR po standaryzacji: ")
    print(f"Accuracy: {accuracy(y_test, y_pred_perceptron):.4f}")
    print(f"Precision (Micro): {precision(y_test, y_pred_perceptron, average='micro'):.4f}")
    print(f"Precision (Macro): {precision(y_test, y_pred_perceptron, average='macro'):.4f}")
    print(f"Recall (Micro): {recall(y_test, y_pred_perceptron, average='micro'):.4f}")
    print(f"Recall (Macro): {recall(y_test, y_pred_perceptron, average='macro'):.4f}")
    print(f"F1-score (Micro): {f1_score(y_test, y_pred_perceptron, average='micro'):.4f}")
    print(f"F1-score (Macro): {f1_score(y_test, y_pred_perceptron, average='macro'):.4f}")

    print("\n-- SVC OVR:")
    print(f"Accuracy: {accuracy(y_test, y_pred_svc):.4f}")
    print(f"Precision (Micro): {precision(y_test, y_pred_svc, average='micro'):.4f}")
    print(f"Precision (Macro): {precision(y_test, y_pred_svc, average='macro'):.4f}")
    print(f"Recall (Micro): {recall(y_test, y_pred_svc, average='micro'):.4f}")
    print(f"Recall (Macro): {recall(y_test, y_pred_svc, average='macro'):.4f}")
    print(f"F1-score (Micro): {f1_score(y_test, y_pred_svc, average='micro'):.4f}")
    print(f"F1-score (Macro): {f1_score(y_test, y_pred_svc, average='macro'):.4f}")

    tsne = TSNE(n_components=2)
    X_2d = tsne.fit_transform(X_scaled)
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.2)

    ovr_perceptron_2d = OVR(classifier=Perceptron, learning_rate=0.01, max_iter=1000)
    ovr_perceptron_2d.fit(X_train_2d, y_train_2d)

    ovr_svc_2d = OVR(classifier=SVC, kernel='linear', probability=True)
    ovr_svc_2d.fit(X_train_2d, y_train_2d)

    def plot_decision_boundaries(X, y, model, title, class_names):
        h = 0.02 #krok

          # czerwone białe różowe
        cmap_light = ListedColormap(['#d4797d', '#f1fa6e', '#f288c6'])
        cmap_bold = ListedColormap(['#a82c2c', '#dbc744', '#ab22ab'])

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # wyliczenie przewidywanej klasy
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))

        # kolorowanie
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
        # granice dla poszczegolnych klasyfikatorow
        line_styles = ['--', ':', '-.']
        line_colors = ['red', 'darkorange', 'darkviolet']
        for i, (target_class, classifier) in enumerate(model.classifiers):
            line_style = line_styles[i % len(line_styles)]
            line_color = line_colors[i % len(line_colors)]

            if isinstance(classifier, Perceptron):
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                Z_boundary = classifier.predict_proba(grid_points)
                Z_boundary = Z_boundary.reshape(xx.shape)
                #  wyjście = 0
                plt.contour(xx, yy, Z_boundary, levels=[0],
                            colors=[line_color], linestyles=[line_style],
                            linewidths=2, alpha=0.8)

            elif isinstance(classifier, SVC):

                grid_points = np.c_[xx.ravel(), yy.ravel()]
                Z_boundary = classifier.decision_function(grid_points)
                Z_boundary = Z_boundary.reshape(xx.shape)
                # decision_function=0
                plt.contour(xx, yy, Z_boundary, levels=[0],
                            colors=[line_color], linestyles=[line_style],
                            linewidths=2, alpha=0.8)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(title)
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')

        legend_labels = [f'{name}' for name in class_names]
        point_legend = plt.legend(handles=scatter.legend_elements()[0],
                                  labels=legend_labels,
                                  title="Klasy",
                                  loc="upper left")
        plt.gca().add_artist(point_legend)

        boundary_handles = []
        for i, (target_class, _) in enumerate(model.classifiers):
            line_style = line_styles[i % len(line_styles)]
            line_color = line_colors[i % len(line_colors)]
            boundary_handles.append(plt.Line2D([0], [0], color=line_color, lw=2,
                                               linestyle=line_style,
                                               label=f'{class_names[target_class]} vs Rest'))

        plt.legend(handles=boundary_handles,
                   title="Granice binarne",
                   loc="upper right")

        plt.tight_layout()
        plt.show()

    plot_decision_boundaries(X_2d, y, ovr_perceptron_2d, "Granice decyzyjne 2D dla Perceptronu OVR TSNE",
                             class_names)

    plot_decision_boundaries(X_2d, y, ovr_svc_2d, "Granice decyzyjne 2D dla SVC OVR", class_names)

    def run(n_runs=50):
        data = load_wine()
        X = data.data
        y = data.target

        perceptron_accuracies = []
        svc_accuracies = []

        for i in range(n_runs):
            X_scaled = standard_scaler(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=i*2)

            ovr_perceptron = OVR(classifier=Perceptron, learning_rate=0.01, max_iter=1000, random_state=i*2)
            ovr_perceptron.fit(X_train, y_train)
            y_pred_perceptron = ovr_perceptron.predict(X_test)
            perceptron_acc = accuracy(y_test, y_pred_perceptron)
            perceptron_accuracies.append(perceptron_acc)

            ovr_svc = OVR(classifier=SVC, kernel='linear', probability=True, random_state=i*2)
            ovr_svc.fit(X_train, y_train)
            y_pred_svc = ovr_svc.predict(X_test)
            svc_acc = accuracy(y_test, y_pred_svc)
            svc_accuracies.append(svc_acc)

            print(f"it {i}; Perceptron: {perceptron_acc:.4f}, SVC: {svc_acc:.4f}")

        mean_perceptron = np.mean(perceptron_accuracies)
        std_perceptron = np.std(perceptron_accuracies)
        mean_svc = np.mean(svc_accuracies)
        std_svc = np.std(svc_accuracies)

        print(f"OVR Perceptron-dokl: {mean_perceptron:.4f}, std {std_perceptron:.4f}")
        print(f"OVR SVC: {mean_svc:.4f}, std {std_svc:.4f}")
        plt.figure(figsize=(8, 6))

        plt.bar([0,1], [mean_perceptron, mean_svc], color=['lightblue', 'lightgreen'], tick_label=['OVR Perceptron', 'OVR SVC'])

        plt.text(0, mean_perceptron, f'{mean_perceptron:.4f}', ha='center', va='bottom', fontsize=10)
        plt.text(1, mean_svc, f'{mean_svc:.4f}', ha='center', va='bottom', fontsize=10)

        plt.title(f'Porównanie dokładności dla{n_runs} uruchomień')
        plt.ylabel('Dokładność', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        return perceptron_accuracies, svc_accuracies


    run(n_runs=40)