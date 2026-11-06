"""Полный размер промптов для GigaChat — для расчёта стоимости.

GigaChat в usage не учитывает закешированную часть промпта (системный промпт, примеры E),
поэтому число входных токенов из журнала занижено и «скачет». Здесь каждый промпт считается
заново бесплатным методом tokens_count (запрос к модели не отправляется).
Запуск один раз: python tools/gigachat_tokens.py  ->  results/gigachat_prompt_tokens.csv
"""
import json
import random
import sys

import pandas as pd

sys.path.insert(0, ".")
from llm import _client  # noqa: E402
from prepare_e import TARGET, load, row_to_text  # noqa: E402
from prompts import SYSTEM, SYSTEM_E, e_prompt  # noqa: E402

client = _client("gigachat")


def count(texts):
    res = []
    for i in range(0, len(texts), 50):  # пачками, чтобы не упереться в размер запроса
        res += [r.tokens for r in client.tokens_count(texts[i:i + 50], model="GigaChat-2-Max")]
    return res


rows = []
bank = pd.read_csv("data/bank.csv")
for tag, n in zip("bank:" + bank["id"], count(list(bank["text_ru"]))):
    rows.append({"tag": tag, "system": "SYSTEM", "user_tokens": n})

df = load()
with open("data/e_split.json", encoding="utf-8") as f:
    split = json.load(f)
for shots in (0, 5, 20):  # те же промпты, что в run_e.py
    ids = list(split.get(f"shots{shots}", []))
    random.Random(42).shuffle(ids)
    examples = [(row_to_text(df.loc[i]), int(df.loc[i, TARGET])) for i in ids]
    texts = [e_prompt(row_to_text(df.loc[i]), examples) for i in split["test"]]
    for i, n in zip(split["test"], count(texts)):
        rows.append({"tag": f"E{shots}:{i}", "system": "SYSTEM_E", "user_tokens": n})

out = pd.DataFrame(rows)
sys_tokens = dict(zip(["SYSTEM", "SYSTEM_E"], count([SYSTEM, SYSTEM_E])))
out["system_tokens"] = out["system"].map(sys_tokens)
out["prompt_tokens"] = out["user_tokens"] + out["system_tokens"]
out.to_csv("results/gigachat_prompt_tokens.csv", index=False)
print(out.groupby(out["tag"].str.split(":").str[0])["prompt_tokens"].mean().round())
