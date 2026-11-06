"""Бланки слепой проверки C и D для двух проверяющих.

Ответы всех моделей на C и D перемешиваются (seed 42) и получают коды R01..R40, имя модели в бланке не указано.
Код из ответов D запускается автоматически (это механика), а оценки ставят люди.
Результат: review/бланк_проверяющий_1.csv, review/бланк_проверяющий_2.csv (одинаковые),
review/ключ.csv — какой код какой модели (проверяющим НЕ открывать до конца проверки).
"""
import os
import random
import re
import subprocess
import sys
import tempfile

import pandas as pd

bank = pd.read_csv("data/bank.csv").set_index("id")
ans = pd.read_csv("results/bank_answers.csv")
ans = ans[ans["type"].isin(["C", "D"])].reset_index(drop=True)

rng = random.Random(42)
rows = []
for task_id, group in ans.groupby("id", sort=False):
    idx = list(group.index)
    rng.shuffle(idx)  # внутри задачи порядок моделей случайный
    rows += idx
ans = ans.loc[rows].reset_index(drop=True)
ans["код"] = [f"R{i + 1:02d}" for i in range(len(ans))]


def run_code(text):
    """Достаёт блок ```python``` из ответа и запускает его. Возвращает (запустился 0/1, хвост вывода)."""
    blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        return 0, "нет блока кода"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "answer.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(blocks[-1])
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=tmp, capture_output=True,
                               text=True, timeout=300, env={**os.environ, "MPLBACKEND": "Agg"})
        except subprocess.TimeoutExpired:
            return 0, "превышено время 300 с"
    out = (p.stdout if p.returncode == 0 else p.stderr).strip()
    return int(p.returncode == 0), out[-400:]


blank = []
for _, r in ans.iterrows():
    task = bank.loc[r["id"]]
    row = {"код": r["код"], "задача": r["id"], "условие": task["text_ru"],
           "эталон": task["answer"] if r["type"] == "C" else f"{task['answer']} (допуск ±{task['tol']})",
           "ответ модели": r["full_text"]}
    if r["type"] == "C":
        row |= {"балл C (0-2)": "", "запустился (авто)": "", "вывод программы": "",
                "D: результат в допуске (0/1)": "", "D: методология (1 нет ошибок / 0 есть)": ""}
    else:
        ok, out = run_code(r["full_text"])
        row |= {"балл C (0-2)": "—", "запустился (авто)": ok, "вывод программы": out,
                "D: результат в допуске (0/1)": "", "D: методология (1 нет ошибок / 0 есть)": ""}
    row["комментарий"] = ""
    blank.append(row)
    print(r["код"], r["id"], row["запустился (авто)"])

os.makedirs("review", exist_ok=True)
blank = pd.DataFrame(blank)
for n in [int(a) for a in sys.argv[1:]] or (1, 2):  # можно пересобрать один бланк: make_review.py 2; sep=';' и utf-8-sig — чтобы русский Excel открыл файл двойным щелчком
    blank.to_csv(f"review/бланк_проверяющий_{n}.csv", sep=";", index=False, encoding="utf-8-sig")
ans[["код", "id", "model"]].to_csv("review/ключ.csv", index=False)
