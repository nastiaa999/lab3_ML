"""Единая функция ask() для всех четырёх моделей и журнал всех вызовов в JSONL."""
import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LOG_PATH = "results/calls.jsonl"

# Короткое имя модели -> как к ней обращаться. Точный ID, который вернул сервер, пишется в журнал.
MODELS = {
    "gigachat": {"provider": "gigachat", "model": "GigaChat-2-Max", "price_in": None, "price_out": None},
    "alice": {"provider": "yandex", "model": "aliceai-llm", "price_in": 0.5, "price_out": 1.2},
    "deepseek": {"provider": "yandex", "model": "deepseek-v4.1-flash", "price_in": 0.3, "price_out": 0.5},
    "qwen_local": {"provider": "ollama", "model": "qwen3:4b", "price_in": 0.0, "price_out": 0.0},
}
# Цены — рубли за 1000 токенов по официальным тарифам Yandex AI Studio (ноябрь 2026).
# Для GigaChat цена считается из пакета в анализе (тариф пакетный, а не за 1000 токенов).

_clients = {}


def _client(provider):
    """Создаёт клиента один раз и переиспользует его."""
    if provider not in _clients:
        if provider == "yandex":
            _clients[provider] = OpenAI(api_key=os.environ["YANDEX_API_KEY"],
                                        base_url="https://llm.api.cloud.yandex.net/v1",
                                        project=os.environ["YANDEX_FOLDER_ID"])
        elif provider == "ollama":
            _clients[provider] = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        elif provider == "gigachat":
            from gigachat import GigaChat
            _clients[provider] = GigaChat(credentials=os.environ["GIGACHAT_CREDENTIALS"],
                                          scope="GIGACHAT_API_PERS", verify_ssl_certs=False)
    return _clients[provider]


def _call(name, messages, temperature):
    """Один запрос к модели. Возвращает (текст, токены_вход, токены_выход, id_модели_от_сервера)."""
    cfg = MODELS[name]
    client = _client(cfg["provider"])
    if cfg["provider"] == "gigachat":
        resp = client.chat({"model": cfg["model"], "messages": messages, "temperature": temperature})
    else:
        model = cfg["model"]
        if cfg["provider"] == "yandex":
            model = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{model}"
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content, resp.usage.prompt_tokens,
            resp.usage.completion_tokens, resp.model)


def ask(model, prompt, temperature=0.0, system=None, history=None, tag=""):
    """Задаёт вопрос модели и сразу записывает вызов в журнал.

    model — короткое имя из MODELS; history — предыдущие сообщения диалога (если нужны);
    tag — метка для журнала (например, id задачи). Возвращает (ответ, токены, время в секундах).
    """
    messages = ([{"role": "system", "content": system}] if system else []) + (history or [])
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    error = None
    try:
        text, tok_in, tok_out, server_model = _call(model, messages, temperature)
    except Exception as e:  # ошибку тоже пишем в журнал, чтобы ничего не терять
        text, tok_in, tok_out, server_model, error = "", 0, 0, None, repr(e)
    seconds = time.perf_counter() - start

    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "tag": tag, "model": model,
              "server_model": server_model, "temperature": temperature, "messages": messages,
              "answer": text, "tokens_in": tok_in, "tokens_out": tok_out,
              "seconds": round(seconds, 2), "error": error}
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if error:
        raise RuntimeError(f"{model}: {error}")
    return text, tok_in + tok_out, seconds


if __name__ == "__main__":
    # Проверка подключения: python llm.py
    for name in MODELS:
        try:
            answer, tokens, sec = ask(name, "Сколько будет 2+2? Ответь одним числом.", tag="smoke")
            print(f"{name:12s} OK  {sec:5.1f} c  {tokens:4d} ток.  {answer.strip()[:60]!r}")
        except Exception as e:
            print(f"{name:12s} ОШИБКА  {e}"[:200])
