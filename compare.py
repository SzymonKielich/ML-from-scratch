import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from KNNClassifier import KNNClassifier, accuracy, precision, recall, f1_score, cross_validation
from OVR import OVR, Perceptron, standard_scaler, plot_confusion_matrix
def evaluate(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    print("-----", name, "-----")

    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Precision (Micro): {precision(y_test, y_pred, average='micro'):.4f}")
    print(f"Precision (Macro): {precision(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (Micro): {recall(y_test, y_pred, average='micro'):.4f}")
    print(f"Recall (Macro): {recall(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score (Micro): {f1_score(y_test, y_pred, average='micro'):.4f}")
    print(f"F1-score (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
    plot_confusion_matrix(y_test, y_pred, class_names, f'Confusion Matrix - {name}')

    return accuracy(y_test, y_pred)
if __name__ == "__main__":

    np.random.seed()
    data = load_wine()
    X = data.data
    y = data.target
    class_names = data.target_names
    print(f"Liczba próbek: {X.shape[0]}")
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Liczba klas: {len(np.unique(y))}")

    X_scaled = standard_scaler(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
    # X_test_2, X_val_2, y_test_2, y_val_2 = train_test_split(X_test, y_test,
    #                                                 test_size=test_ratio / (test_ratio + validation_ratio), stratify=y_test)
    print(f"Zbiór treningowy: {len(X_train)/len(X):.1%}")
    print(f"Zbiór testowy: {len(X_test)/len(X):.1%}")


    print("Modele z domyślnymi parametrami")
    results_before = {}

    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    results_before['KNN'] = evaluate(knn, "KNN przed tuningiem parametrów", X_test, y_test)

    perc = OVR(classifier=Perceptron)
    perc.fit(X_train, y_train)
    results_before['Perceptron'] = evaluate(perc, "Perceptron przed tuningiem parametrów", X_test, y_test)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    results_before['RandomForest'] = evaluate(rf, "RandomForest przed tuningiem parametrów", X_test, y_test)

    bag = BaggingClassifier()
    bag.fit(X_train, y_train)
    results_before['Bagging'] = evaluate(bag, "Bagging przed tuningiem parametrów", X_test, y_test)

    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    results_before['AdaBoost'] = evaluate(ada, "AdaBoost przed tuningiem parametrów", X_test, y_test)


    print("\nOptymalizacja hiperparametru k przy użyciu walidacji krzyżowej dla KNN")

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
    KNN_after = KNNClassifier(k=best_k, p=best_p)
    KNN_after.fit(X_train, y_train)
    results_after = {}
    results_after['KNN'] = evaluate(KNN_after, f"KNN po tuningu", X_test, y_test)
    
    best_perc, best_score = None, -1
    best_lr = None
    for lr in [0.0001, 0.001, 0.01, 0.2, 1.0, 5.0]:
        model = OVR(classifier=Perceptron, learning_rate=lr, max_iter=1000)
        model.fit(X_train, y_train)
        score = accuracy(y_test, model.predict(X_test))
        if score > best_score:
            best_score = score
            best_perc = model
            best_lr = lr
    print(f"\nNajlepszy Perceptron OVR: learning_rate={best_lr}")
    results_after['Perceptron'] = evaluate(best_perc, "Perceptron OVR po tuningu", X_test, y_test)


    print("\nTuning modeli sklearn")
    def grid_search_tuning(name, model_cls, param_grid):
        grid = GridSearchCV(model_cls(), param_grid, cv=5)
        grid.fit(X_train, y_train)
        print(f"\nNajlepsze parametry {name}: {grid.best_params_}")
        return evaluate(grid.best_estimator_, f"{name} po tuningu", X_test, y_test)

    results_after['RandomForest'] = grid_search_tuning("RandomForest", RandomForestClassifier, {
        'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20]
    })

    results_after['Bagging'] = grid_search_tuning("Bagging", BaggingClassifier, {
        'n_estimators': [5, 10, 20, 50], 'max_samples': [0.6, 0.8, 1.0]
    })

    results_after['AdaBoost'] = grid_search_tuning("AdaBoost", AdaBoostClassifier, {
        'n_estimators': [30, 50, 100], 'learning_rate': [0.1, 0.25, 0.5, 0.75, 1.0]
    })

    models = list(results_before.keys())
    scores_before = [results_before[m] for m in models]
    scores_after = [results_after[m] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, scores_before, width, label='Przed')
    plt.bar(x + width/2, scores_after, width, label='Po')
    plt.xticks(x, models)
    plt.ylabel("accuracy")
    plt.title("Porównanie accuracy przed i po tuningu", pad=20)
    for i in range(len(models)):
        plt.text(i - width/2, scores_before[i] + 0.01, f"{scores_before[i]:.3f}", ha='center')
        plt.text(i + width/2, scores_after[i] + 0.01, f"{scores_after[i]:.3f}", ha='center')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


