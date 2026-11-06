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
    "gigachat_max": {"provider": "gigachat", "model": "GigaChat-2-Max"},
    "alice_flash": {"provider": "yandex", "model": "aliceai-llm-flash/latest"},
    "gpt_oss": {"provider": "groq", "model": "openai/gpt-oss-120b", "max_tokens": 8192},
    "qwen_local": {"provider": "ollama", "model": "qwen3:4b", "max_tokens": 8192},
}
# qwen3:4b: у Ollama окно контекста по умолчанию 4096 — длинные рассуждения в него не помещались,
# модель теряла условие и зацикливалась. Окно увеличено до 16384 (переменная OLLAMA_CONTEXT_LENGTH),
# длина ответа ограничена так же, как у gpt-oss.
# gpt-oss: лимит ответа поднят с 3072 (по умолчанию у Groq) до 8192 — на 20 примерах рассуждения
# не помещались и ответ обрезался. Ни один ответ до этого лимита не доходил, так что сравнение не нарушено.
# Цены для расчёта стоимости правильного ответа задаются в анализе по официальным тарифам
# (GigaChat — пакеты Сбера, Alice и gpt-oss-120b — прайс Yandex AI Studio), хотя мы пользуемся бесплатными лимитами.

_clients = {}


def _client(provider):
    """Создаёт клиента один раз и переиспользует его."""
    if provider not in _clients:
        if provider == "yandex":
            _clients[provider] = OpenAI(api_key=os.environ["YANDEX_API_KEY"],
                                        base_url="https://ai.api.cloud.yandex.net/v1",
                                        project=os.environ["YANDEX_FOLDER_ID"])
        elif provider == "groq":
            _clients[provider] = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                                        base_url="https://api.groq.com/openai/v1")
        elif provider == "ollama":
            # локальная модель медленная: ждём до 40 мин и не повторяем запрос автоматически
            _clients[provider] = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1",
                                        timeout=2400, max_retries=0)
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
        if cfg["provider"] == "yandex":  # у Яндекса полный ID модели включает каталог
            model = f"gpt://{os.environ['YANDEX_FOLDER_ID']}/{model}"
        extra = {"max_tokens": cfg["max_tokens"]} if "max_tokens" in cfg else {}
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **extra)
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
