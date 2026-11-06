"""Эталонные решения задач типа D. Запуск: python tools/reference_d.py

Условия не требуют масштабирования, поэтому для D1, D2, D4 (как и для D5) допустимы оба варианта:
без масштабирования (буквально по условию) и со StandardScaler. В D3 условие не требует stratify и не
называет метрику — допустима любая метрика редкого класса (F1, recall, PR-AUC).
Исправлено после прогона: первая версия эталонов D1–D4 включала шаги, которых нет в условии.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_digits, load_iris, load_wine, make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, adjusted_rand_score, average_precision_score, f1_score,
                             r2_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def d1(scale):
    X, y = load_wine(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = make_pipeline(*([StandardScaler()] if scale else []), KNeighborsClassifier())
    grid = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": range(1, 16)}, cv=5)
    grid.fit(X_tr, y_tr)
    return accuracy_score(y_te, grid.predict(X_te))


def d2(scale):
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(*([StandardScaler()] if scale else []), Ridge())
    grid = GridSearchCV(pipe, {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="r2")
    grid.fit(X_tr, y_tr)
    return r2_score(y_te, grid.predict(X_te))


def d3():
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                               weights=[0.95], random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    rf = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    return {"F1": f1_score(y_te, pred), "recall": recall_score(y_te, pred),
            "PR-AUC": average_precision_score(y_te, rf.predict_proba(X_te)[:, 1])}


def d4(scale):
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = make_pipeline(*([StandardScaler()] if scale else []), PCA(n_components=0.95),
                         LogisticRegression(max_iter=5000))
    pipe.fit(X_tr, y_tr)
    return accuracy_score(y_te, pipe.predict(X_te))


def d5():
    X, y = load_iris(return_X_y=True)
    raw = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
    scaled = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(
        StandardScaler().fit_transform(X))
    return adjusted_rand_score(y, raw), adjusted_rand_score(y, scaled)


if __name__ == "__main__":
    for f in (d1, d2, d4):
        print(f.__name__, "без масштабирования:", round(f(False), 4), " со StandardScaler:", round(f(True), 4))
    print("d3", {k: round(v, 4) for k, v in d3().items()})
    print("d5 без масштабирования / со StandardScaler:", np.round(d5(), 4))
