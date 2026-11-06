"""Каппа Коэна между двумя проверяющими C и D и итоговые баллы моделей. Запуск: python tools/kappa.py"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

COLS = {"балл C (0-2)": "C", "D: результат в допуске (0/1)": "D_result",
        "D: методология (1 нет ошибок / 0 есть)": "D_method"}

r1 = pd.read_csv("review/бланк_проверяющий_1_заполнен.csv", sep=";", dtype=str, encoding="utf-8-sig")
r2 = pd.read_csv("review/бланк_проверяющий_2_заполнен.csv", sep=";", dtype=str, encoding="utf-8-sig")
assert list(r1["код"]) == list(r2["код"])
key = pd.read_csv("review/ключ.csv")

rows, scores = [], key.copy()
for col, name in COLS.items():
    mask = r1[col].notna() & (r1[col] != "—")  # C — только строки C, D — только строки D
    a, b = r1.loc[mask, col].astype(int), r2.loc[mask, col].astype(int)
    rows.append({"оценка": name, "n": int(mask.sum()), "совпало": int((a == b).sum()),
                 "kappa": round(cohen_kappa_score(a, b), 3),
                 "kappa взвеш.": round(cohen_kappa_score(a, b, weights="linear"), 3) if name == "C" else None})
    scores.loc[mask.values, name] = (a.values + b.values) / 2  # итог — среднее двух проверяющих
    diff = r1.loc[mask & (r1[col] != r2[col]), "код"].tolist()
    if diff:
        print(f"{name}: расхождения в {diff}")

print(pd.DataFrame(rows).to_string(index=False))
scores.to_csv("results/cd_scores.csv", index=False)
print(scores.groupby("model")[["C", "D_result", "D_method"]].mean().round(2))

# B по рубрике 0–2 (отдельные бланки, tools/make_review_b.py)
b1 = pd.read_csv("review/бланк_B_проверяющий_1_заполнен.csv", sep=";", dtype=str, encoding="utf-8-sig")
b2 = pd.read_csv("review/бланк_B_проверяющий_2_заполнен.csv", sep=";", dtype=str, encoding="utf-8-sig")
assert list(b1["код"]) == list(b2["код"])
x, y = b1["балл B (0-2)"].astype(int), b2["балл B (0-2)"].astype(int)
print(f"B: совпало {(x == y).sum()} из {len(x)}, kappa = {cohen_kappa_score(x, y):.3f}")
b_scores = pd.read_csv("review/ключ_B.csv").assign(B=(x.values + y.values) / 2)
b_scores.to_csv("results/b_scores.csv", index=False)
print(b_scores.groupby("model")["B"].mean().round(2))
