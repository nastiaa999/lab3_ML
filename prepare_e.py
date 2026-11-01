"""Тип E: подготовка AI4I 2020, разбиение, текстовое описание строк и классические бейзлайны.

Запуск: python prepare_e.py  ->  data/e_split.json, results/e_baselines.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_TEST, N_TEST_FAIL = 100, 20  # тест обогащён отказами: 20% вместо естественных 3,4%
SHOTS = {5: 1, 20: 4}          # сколько примеров и сколько из них отказов (та же доля 20%, что в тесте)

FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET = "Machine failure"
TYPE_RU = {"L": "L (низкое качество)", "M": "M (среднее качество)", "H": "H (высокое качество)"}


def load():
    df = pd.read_csv("data/ai4i2020.csv")
    # UDI и Product ID — идентификаторы; TWF, HDF, PWF, OSF, RNF — виды отказа = утечка целевой переменной
    return df[FEATURES + [TARGET]]


def row_to_text(row):
    """Строка таблицы -> описание на русском, которое получает LLM."""
    return (f"Тип продукта: {TYPE_RU[row['Type']]}. "
            f"Температура воздуха: {row['Air temperature [K]']} K. "
            f"Температура процесса: {row['Process temperature [K]']} K. "
            f"Скорость вращения: {row['Rotational speed [rpm]']} об/мин. "
            f"Крутящий момент: {row['Torque [Nm]']} Н·м. "
            f"Износ инструмента: {row['Tool wear [min]']} мин.")


def to_matrix(df):
    """Признаки для классических моделей: Type кодируем порядком качества L < M < H."""
    X = df[FEATURES].copy()
    X["Type"] = X["Type"].map({"L": 0, "M": 1, "H": 2})
    return X.to_numpy(dtype=float)


def make_split(df):
    rng = np.random.default_rng(SEED)
    fail, ok = df.index[df[TARGET] == 1], df.index[df[TARGET] == 0]
    test = np.concatenate([rng.choice(fail, N_TEST_FAIL, replace=False),
                           rng.choice(ok, N_TEST - N_TEST_FAIL, replace=False)])
    train = df.index.difference(test)
    train_fail = rng.permutation(train.intersection(fail))
    train_ok = rng.permutation(train.intersection(ok))
    # 20 примеров = 4 отказа + 16 норм; 5 примеров — первые 1 отказ + 4 нормы из них (5 ⊂ 20)
    shots20 = list(train_fail[:4]) + list(train_ok[:16])
    shots5 = list(train_fail[:1]) + list(train_ok[:4])
    to_int = lambda idx: [int(i) for i in idx]
    return {"test": to_int(rng.permutation(test)), "shots5": to_int(shots5), "shots20": to_int(shots20)}


def metrics(y_true, y_pred):
    return {"f1_fail": f1_score(y_true, y_pred, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))}


def baselines(df, split):
    test = df.loc[split["test"]]
    X_te, y_te = to_matrix(test), test[TARGET].to_numpy()
    rows = []
    const = DummyClassifier(strategy="most_frequent").fit(to_matrix(df), df[TARGET])
    rows.append({"model": "константа «нет отказа»", "shots": 0, **metrics(y_te, const.predict(X_te))})
    for k in (5, 20):
        part = df.loc[split[f"shots{k}"]]
        X_tr, y_tr = to_matrix(part), part[TARGET].to_numpy()
        for name, clf in [("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression())),
                          ("GaussianNB", GaussianNB())]:
            clf.fit(X_tr, y_tr)
            rows.append({"model": name, "shots": k, **metrics(y_te, clf.predict(X_te))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    split = make_split(df)
    with open("data/e_split.json", "w", encoding="utf-8") as f:
        json.dump(split, f)
    res = baselines(df, split)
    res.to_csv("results/e_baselines.csv", index=False)
    print(f"Отказов во всём датасете: {df[TARGET].mean():.1%}")
    print(res.round(3).to_string(index=False))
    print("Пример текста:", row_to_text(df.loc[split["test"][0]]))
