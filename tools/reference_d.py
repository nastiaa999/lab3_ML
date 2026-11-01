"""Эталонные решения задач типа D. Запуск: python tools/reference_d.py"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_digits, load_iris, load_wine, make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def d1():
    X, y = load_wine(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    grid = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": range(1, 16)}, cv=5)
    grid.fit(X_tr, y_tr)
    return accuracy_score(y_te, grid.predict(X_te))


def d2():
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), Ridge())
    grid = GridSearchCV(pipe, {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="r2")
    grid.fit(X_tr, y_tr)
    return r2_score(y_te, grid.predict(X_te))


def d3():
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                               weights=[0.95], random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    rf = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
    return f1_score(y_te, rf.predict(X_te))  # F1 редкого класса 1


def d4():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95),
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
    for f in (d1, d2, d3, d4, d5):
        print(f.__name__, np.round(f(), 4))
