"""Прогон банка задач через все модели с автопроверкой A, B, F.

Запуск: python run_bank.py pilot   (5 пилотных задач)
        python run_bank.py bank    (основной банк)
Каждый ответ сразу дописывается в results/<имя>_answers.csv. Уже полученные ответы при повторном
запуске пропускаются, поэтому прерванный прогон можно просто запустить заново — лишних запросов не будет.
"""
import os
import sys
import time

import pandas as pd

from check import grade, parse
from llm import MODELS, ask
from prompts import SYSTEM

AUTO_TYPES = {"A", "B", "F"}  # C и D проверяются вручную


class DailyLimit(Exception):
    """Дневной лимит бесплатного тарифа исчерпан — продолжать в этот день бессмысленно."""


def ask_patiently(model, prompt, system, tag, retries=3):
    """ask() с паузой при поминутном лимите (ошибка 429). Неудача -> None, дневной лимит -> DailyLimit."""
    for attempt in range(retries + 1):
        try:
            return ask(model, prompt, temperature=0, system=system, tag=tag)
        except RuntimeError as e:
            msg = str(e)
            if "per day" in msg:
                raise DailyLimit(model)
            if "429" in msg and attempt < retries:
                time.sleep(60)
                continue
            print(f"ОШИБКА {model} {tag}: {msg}"[:200])
            return None


def run(name, models=None):
    tasks = pd.read_csv(f"data/{name}.csv")
    out_path = f"results/{name}_answers.csv"
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        done = set(zip(prev["model"], prev["id"]))

    for model in models or MODELS:
        try:
            run_model(model, name, tasks, out_path, done)
        except DailyLimit:
            print(f"{model}: дневной лимит исчерпан — запустите завтра, продолжится с того же места")


def run_model(model, name, tasks, out_path, done):
    """Прогоняет одну модель по всем ещё не решённым задачам."""
    for _, t in tasks.iterrows():
        if (model, t["id"]) in done:
            continue
        result = ask_patiently(model, t["text_ru"], system=SYSTEM, tag=f"{name}:{t['id']}")
        if result is None:  # ошибку не записываем как ответ — при перезапуске запрос повторится
            continue
        text, tokens, seconds = result
        answer, conf = parse(text)
        correct = refusal = None
        if t["type"] in AUTO_TYPES:
            correct, refusal = grade(answer, t["answer"], t["tol"])
        row = {"model": model, "id": t["id"], "type": t["type"], "answer": answer, "confidence": conf,
               "correct": correct, "refusal": refusal, "tokens": tokens, "seconds": round(seconds, 1),
               "full_text": text}
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(f"{model:12s} {t['id']:4s} ответ={str(answer)[:30]!r:34s} верно={correct} {seconds:5.1f}c")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:] or None)
