"""Разбор ответа модели и автоматическая проверка задач A, B, F (и E)."""
import re

CANNOT = "НЕЛЬЗЯ ОПРЕДЕЛИТЬ"


def parse(text):
    """Достаёт из ответа строки «ОТВЕТ:» и «УВЕРЕННОСТЬ:» (берём последние). Нет строки -> None."""
    text = text or ""
    answers = re.findall(r"ОТВЕТ\s*:\s*\**\s*(.+)", text, flags=re.IGNORECASE)
    confs = re.findall(r"УВЕРЕННОСТЬ\s*:\s*\**\s*(\d+(?:[.,]\d+)?)", text, flags=re.IGNORECASE)
    answer = answers[-1].strip().strip("*").strip() if answers else None
    conf = float(confs[-1].replace(",", ".")) if confs else None
    return answer, conf


def to_number(s):
    """'0,16' -> 0.16; '16%' -> 0.16; '0.161' -> 0.161. Не число -> None."""
    m = re.search(r"-?\d+(?:[.,]\d+)?", s.replace(" ", ""))
    if not m:
        return None
    x = float(m.group().replace(",", "."))
    return x / 100 if "%" in s else x


def is_cannot(s):
    s = s.upper()
    return "НЕЛЬЗЯ" in s or "НЕДОСТАТОЧНО" in s or "НЕВОЗМОЖНО" in s


def grade(answer, reference, tol):
    """Возвращает (верно: bool, отказ: bool). Отказ = нет ответа или «нельзя определить», когда ответ есть."""
    if answer is None:
        return False, True
    ref = str(reference)
    if ref == CANNOT:
        return is_cannot(answer), False
    if ref in ("ВЕРНО", "НЕВЕРНО"):
        a = answer.upper()
        verdict = "НЕВЕРНО" if "НЕВЕРНО" in a else ("ВЕРНО" if "ВЕРНО" in a else None)
        return verdict == ref, verdict is None
    if is_cannot(answer):
        return False, True
    x = to_number(answer)
    if x is None:
        return False, True
    ref = float(ref)
    return abs(x - ref) <= max(tol * abs(ref), 1e-3), False  # допуск 1% относительный, для нуля — 0.001


if __name__ == "__main__":
    tests = [("...\nОТВЕТ: 0,0776\nУВЕРЕННОСТЬ: 90", 0.0776, (True, False)),
             ("ОТВЕТ: 7.76%\nУВЕРЕННОСТЬ: 80", 0.0776, (True, False)),
             ("**ОТВЕТ:** НЕВЕРНО\nУВЕРЕННОСТЬ: 95", "НЕВЕРНО", (True, False)),
             ("ОТВЕТ: ВЕРНО", "НЕВЕРНО", (False, False)),
             ("ОТВЕТ: Нельзя определить, нет FP", CANNOT, (True, False)),
             ("ОТВЕТ: НЕЛЬЗЯ ОПРЕДЕЛИТЬ", 0.5, (False, True)),
             ("Ответ без формата", 0.5, (False, True)),
             ("ОТВЕТ: 0", 0.0, (True, False))]
    for text, ref, expected in tests:
        ans, _ = parse(text)
        assert grade(ans, ref, 0.01) == expected, (text, grade(ans, ref, 0.01))
    print("Все проверки check.py пройдены")
