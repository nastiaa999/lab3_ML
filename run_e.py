"""Тип E: LLM как классификатор отказов станка (AI4I 2020). Одна тестовая строка = один запрос.

Запуск: python run_e.py [модели...] [число примеров...]  ->  results/e_answers.csv
        например: python run_e.py qwen_local 5 20   (только режимы с 5 и 20 примерами)
                  python run_e.py gpt_oss 20 part2  (только тестовые строки 51–100 — для второго ключа Groq)
Как и run_bank.py, пропускает уже полученные ответы и останавливает модель при дневном лимите.
"""
import json
import os
import random
import sys

import pandas as pd

from check import parse, to_number
from llm import MODELS
from prepare_e import TARGET, load, row_to_text
from prompts import SYSTEM_E, e_prompt
from run_bank import DailyLimit, ask_patiently

OUT = "results/e_answers.csv"


def run_model(model, df, split, done, modes=(0, 5, 20)):
    for shots in modes:
        ids = list(split.get(f"shots{shots}", []))
        random.Random(42).shuffle(ids)  # перемешиваем, чтобы отказы не стояли всегда первыми
        examples = [(row_to_text(df.loc[i]), int(df.loc[i, TARGET])) for i in ids]
        test = split["test"]
        if "part1" in sys.argv:  # делим тест пополам: два ключа Groq — два дневных лимита
            test = test[:50]
        if "part2" in sys.argv:
            test = test[50:]
        for n, i in enumerate(test, 1):
            if (model, shots, i) in done:
                continue
            result = ask_patiently(model, e_prompt(row_to_text(df.loc[i]), examples),
                                   system=SYSTEM_E, tag=f"E{shots}:{i}")
            if result is None:
                continue
            text, tokens, seconds = result
            answer, _ = parse(text)
            pred = to_number(answer) if answer else None
            pred = int(pred) if pred in (0, 1) else None  # не 0 и не 1 -> нарушение формата
            row = {"model": model, "shots": shots, "row_id": i, "y_true": int(df.loc[i, TARGET]),
                   "pred": pred, "tokens": tokens, "seconds": round(seconds, 1), "full_text": text}
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            print(f"{model} E{shots}: строка {n}/{len(test)}, ответ={pred}, {seconds:.0f} c", flush=True)
        print(f"{model}: {shots} примеров — готово")


if __name__ == "__main__":
    df = load()
    with open("data/e_split.json", encoding="utf-8") as f:
        split = json.load(f)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev["model"], prev["shots"], prev["row_id"]))
    models = [a for a in sys.argv[1:] if not a.isdigit() and a not in ("part1", "part2")] or MODELS
    modes = [int(a) for a in sys.argv[1:] if a.isdigit()] or (0, 5, 20)
    for model in models:
        try:
            run_model(model, df, split, done, modes)
        except DailyLimit:
            print(f"{model}: дневной лимит исчерпан — запустите завтра, продолжится с того же места")
