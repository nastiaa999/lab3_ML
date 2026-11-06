"""Собирает банк задач data/bank.csv и пилот data/pilot.csv.

Эталоны задач A и F считаются здесь же формулами, эталоны D — в tools/reference_d.py.
Поля по шаблону задания: id, тип, тема, текст_ru, текст_en, эталон, допуск, источник, сложность.
Дополнительно pair — номер пары «учебная задача ↔ изменённая» для гипотезы H1.
"""
import math

import pandas as pd

CANNOT = "НЕЛЬЗЯ ОПРЕДЕЛИТЬ"


def gini(*counts):
    n = sum(counts)
    return 1 - sum((c / n) ** 2 for c in counts)


def gini_gain(parent, left, right):
    n = sum(parent)
    return gini(*parent) - sum(left) / n * gini(*left) - sum(right) / n * gini(*right)


def entropy(*counts):
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c > 0)


def info_gain(parent, left, right):
    n = sum(parent)
    return entropy(*parent) - sum(left) / n * entropy(*left) - sum(right) / n * entropy(*right)


def bayes(prior, sens, fpr):
    return prior * sens / (prior * sens + (1 - prior) * fpr)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def nb_posterior_no(p_yes, p_no):
    """P(no | x) в наивном Байесе; p_yes и p_no — списки [априорная, P(x1|c), P(x2|c), ...]."""
    v_yes, v_no = math.prod(p_yes), math.prod(p_no)
    return v_no / (v_yes + v_no)


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


TEXTBOOK = "учебник"
OWN = "придумана нами"
FMT_NUM = " Ответ — одно число."
FMT_NUM_EN = " The answer is a single number."

tasks = [
    # ---------- A: расчётные, «учебниковые» (у каждой есть изменённая пара в F) ----------
    dict(id="A1", type="A", topic="Байес", pair=1, source=TEXTBOOK + " (задача Эдди о маммографии)", difficulty=1,
         text_ru="Рак груди встречается у 1% обследуемых женщин. Маммография обнаруживает рак у 80% больных, "
                 "а у 9,6% здоровых даёт ложноположительный результат. Результат положительный. "
                 "Найдите вероятность того, что у женщины рак." + FMT_NUM,
         text_en="Breast cancer occurs in 1% of screened women. A mammogram detects cancer in 80% of sick women "
                 "and gives a false positive in 9.6% of healthy women. The result is positive. "
                 "Find the probability that the woman has cancer." + FMT_NUM_EN,
         answer=bayes(0.01, 0.80, 0.096)),
    dict(id="A2", type="A", topic="Деревья", pair=2, source=TEXTBOOK + " (пример из задания)", difficulty=2,
         text_ru="В узле дерева 40 объектов класса 0 и 10 объектов класса 1. Разбиение даёт левый узел "
                 "(35 объектов класса 0; 2 класса 1) и правый узел (5; 8). Найдите прирост информации "
                 "по критерию Джини (взвешенное уменьшение неопределённости Джини)." + FMT_NUM,
         text_en="A tree node contains 40 objects of class 0 and 10 objects of class 1. A split gives a left node "
                 "(35 of class 0; 2 of class 1) and a right node (5; 8). Find the information gain by the Gini "
                 "criterion (weighted decrease of Gini impurity)." + FMT_NUM_EN,
         answer=gini_gain((40, 10), (35, 2), (5, 8))),
    dict(id="A3", type="A", topic="Метрики", pair=3, source=TEXTBOOK + " (пример из задания)", difficulty=1,
         text_ru="Матрица ошибок: TP = 30, FP = 10, FN = 20, TN = 940. Найдите F1-меру." + FMT_NUM,
         text_en="Confusion matrix: TP = 30, FP = 10, FN = 20, TN = 940. Find the F1 score." + FMT_NUM_EN,
         answer=f1(30, 10, 20)),
    dict(id="A4", type="A", topic="SVM", pair=4, source=TEXTBOOK + " (пример из задания)", difficulty=1,
         text_ru="Для разделяющей гиперплоскости SVM w = (3; 4), b = −2 (канонический вид: на опорных векторах "
                 "|w·x + b| = 1). Найдите полную ширину разделяющей полосы (зазора)." + FMT_NUM,
         text_en="For an SVM separating hyperplane w = (3; 4), b = −2 (canonical form: |w·x + b| = 1 on support "
                 "vectors). Find the full width of the margin." + FMT_NUM_EN,
         answer=2 / math.hypot(3, 4)),
    dict(id="A5", type="A", topic="Деревья", pair=5, source=TEXTBOOK + " (Митчелл, PlayTennis)", difficulty=2,
         text_ru="Выборка из 14 дней: 9 дней «играть = да», 5 дней «нет». По признаку «ветер»: при слабом ветре "
                 "6 «да» и 2 «нет», при сильном 3 «да» и 3 «нет». Найдите информационный прирост "
                 "(по энтропии, логарифм по основанию 2) при разбиении по признаку «ветер»." + FMT_NUM,
         text_en="A sample of 14 days: 9 days \"play = yes\", 5 days \"no\". By the feature \"wind\": with weak wind "
                 "6 yes and 2 no, with strong wind 3 yes and 3 no. Find the information gain (entropy, log base 2) "
                 "of splitting by \"wind\"." + FMT_NUM_EN,
         answer=info_gain((9, 5), (6, 2), (3, 3))),
    dict(id="A6", type="A", topic="Байес", pair=6, source=TEXTBOOK + " (Митчелл, PlayTennis)", difficulty=3,
         text_ru="Наивный байесовский классификатор, 14 дней: 9 «да», 5 «нет». Среди дней «да»: солнечно 2, "
                 "прохладно 3, высокая влажность 3, сильный ветер 3. Среди дней «нет»: солнечно 3, прохладно 1, "
                 "высокая влажность 4, сильный ветер 3. Новый день: солнечно, прохладно, высокая влажность, "
                 "сильный ветер. Найдите апостериорную вероятность класса «нет» (без сглаживания)." + FMT_NUM,
         text_en="Naive Bayes classifier, 14 days: 9 yes, 5 no. Among yes-days: sunny 2, cool 3, high humidity 3, "
                 "strong wind 3. Among no-days: sunny 3, cool 1, high humidity 4, strong wind 3. New day: sunny, "
                 "cool, high humidity, strong wind. Find the posterior probability of class \"no\" "
                 "(no smoothing)." + FMT_NUM_EN,
         answer=nb_posterior_no([9 / 14, 2 / 9, 3 / 9, 3 / 9, 3 / 9], [5 / 14, 3 / 5, 1 / 5, 4 / 5, 3 / 5])),

    # ---------- B: верно / неверно (3 верных, 3 ложных) ----------
    dict(id="B1", type="B", topic="Кластеризация", source=OWN, difficulty=2, answer="НЕВЕРНО",
         text_ru="Утверждение: «Если в k-means увеличить число кластеров k, то инерция (сумма квадратов расстояний "
                 "до центров) оптимального решения может вырасти». Верно или неверно? Обоснуйте.",
         text_en="Statement: \"If the number of clusters k in k-means is increased, the inertia (sum of squared "
                 "distances to centroids) of the optimal solution can increase.\" True or false? Justify."),
    dict(id="B2", type="B", topic="Логистическая регрессия", source=OWN, difficulty=2, answer="ВЕРНО",
         text_ru="Утверждение: «На линейно разделимой выборке логистическая регрессия без регуляризации не имеет "
                 "конечного оптимума: при обучении норма весов неограниченно растёт». Верно или неверно? Обоснуйте.",
         text_en="Statement: \"On a linearly separable sample, logistic regression without regularization has no "
                 "finite optimum: the norm of the weights grows without bound during training.\" True or false? "
                 "Justify."),
    dict(id="B3", type="B", topic="PCA", source=OWN, difficulty=1, answer="НЕВЕРНО",
         text_ru="Утверждение: «Стандартизация признаков не меняет результат PCA, потому что главные компоненты "
                 "инвариантны к масштабу признаков». Верно или неверно? Обоснуйте.",
         text_en="Statement: \"Standardizing features does not change the result of PCA, because principal "
                 "components are invariant to feature scale.\" True or false? Justify."),
    dict(id="B4", type="B", topic="Ансамбли", source=OWN, difficulty=1, answer="ВЕРНО",
         text_ru="Утверждение: «Бэггинг в первую очередь уменьшает разброс (дисперсию) ошибки базовой модели, "
                 "а не её смещение». Верно или неверно? Обоснуйте.",
         text_en="Statement: \"Bagging primarily reduces the variance of the base model's error rather than its "
                 "bias.\" True or false? Justify."),
    dict(id="B5", type="B", topic="SVM", source=OWN, difficulty=2, answer="НЕВЕРНО",
         text_ru="Утверждение: «В SVM с жёстким зазором удаление из обучающей выборки объекта, который не является "
                 "опорным вектором, может сдвинуть разделяющую гиперплоскость». Верно или неверно? Обоснуйте.",
         text_en="Statement: \"In a hard-margin SVM, removing a training object that is not a support vector can "
                 "shift the separating hyperplane.\" True or false? Justify."),
    dict(id="B6", type="B", topic="Байес", source=OWN, difficulty=3, answer="ВЕРНО",
         text_ru="Утверждение: «Если добавить в наивный байесовский классификатор точную копию одного из признаков, "
                 "предсказанный класс для некоторых объектов может измениться». Верно или неверно? Обоснуйте.",
         text_en="Statement: \"If an exact copy of one of the features is added to a naive Bayes classifier, the "
                 "predicted class may change for some objects.\" True or false? Justify."),

    # ---------- C: выбор метода (проверка вручную по рубрике) ----------
    dict(id="C1", type="C", topic="Деревья / лог. регрессия", source=OWN, difficulty=2,
         text_ru="Литейный цех: 2 000 отливок, 12 технологических параметров (температура металла, скорость "
                 "заливки и т. п.), брак у 4% отливок. Технолог хочет понять, какие параметры ведут к браку, и "
                 "получить понятные пороговые правила. Какой метод выбрать, какую метрику и как проверять качество?",
         text_en="A foundry: 2,000 castings, 12 process parameters (metal temperature, pouring speed, etc.), 4% of "
                 "castings are defective. The engineer wants to understand which parameters lead to defects and get "
                 "clear threshold rules. Which method, which metric, and how to validate?",
         answer="Неглубокое дерево решений (или лог. регрессия) — интерпретируемость и пороги; учёт дисбаланса "
                "(class_weight); метрика recall/F1/PR-AUC редкого класса, не accuracy; стратифицированная CV."),
    dict(id="C2", type="C", topic="Линейная регрессия / ансамбли", source=OWN, difficulty=2,
         text_ru="Почасовые данные энергопотребления здания за 2 года и погода. Нужно прогнозировать потребление на "
                 "следующие сутки. Какой метод выбрать и как проверять качество модели?",
         text_en="Hourly building energy consumption for 2 years plus weather. The task is to forecast consumption "
                 "for the next day. Which method should be chosen and how should the model be validated?",
         answer="Регрессия (градиентный бустинг или линейная) на лаговых и календарных признаках; проверка только "
                "по времени (TimeSeriesSplit / отложенный последний период), без перемешивания; MAE/RMSE."),
    dict(id="C3", type="C", topic="Кластеризация", source=OWN, difficulty=2,
         text_ru="300 станков описаны 6 числовыми признаками (наработка, вибрация, температура и т. п.) в разных "
                 "единицах измерения. Меток нет, в данных есть выбросы. Нужно разбить станки на группы для "
                 "планирования обслуживания. Какой метод выбрать и как выбрать число групп?",
         text_en="300 machines are described by 6 numeric features (operating hours, vibration, temperature, etc.) "
                 "in different units. There are no labels and the data contain outliers. The machines must be "
                 "grouped for maintenance planning. Which method and how to choose the number of groups?",
         answer="Обязательно масштабирование; k-means с выбором k по силуэту/локтю (выбросы убрать) или DBSCAN, "
                "который сам помечает выбросы как шум; интерпретация центров кластеров."),
    dict(id="C4", type="C", topic="PCA / линейная регрессия", source=OWN, difficulty=3,
         text_ru="60 образцов сплава, для каждого 1 500 значений спектра; нужно предсказывать прочность (число). "
                 "Какой метод выбрать и как честно оценить качество?",
         text_en="60 alloy samples, each with 1,500 spectral values; the task is to predict strength (a number). "
                 "Which method should be chosen and how to estimate quality honestly?",
         answer="Признаков намного больше, чем объектов: регуляризованная линейная модель (Ridge/Lasso) или PCA/PLS + линейная регрессия; подбор "
                "внутри кросс-валидации (LOO или повторённая k-fold), PCA обучать только на train-фолдах."),
    dict(id="C5", type="C", topic="kNN / лог. регрессия", source=OWN, difficulty=2,
         text_ru="На микроконтроллере с 32 КБ памяти нужно в реальном времени по 3 датчикам определять «норма / "
                 "авария». Размеченных записей 50 000. Какой метод выбрать и почему не kNN?",
         text_en="A microcontroller with 32 KB of memory must classify \"normal / failure\" in real time from 3 "
                 "sensors. There are 50,000 labeled records. Which method should be chosen and why not kNN?",
         answer="Компактная модель: логистическая регрессия или неглубокое дерево (мало параметров, быстрый "
                "предсказатель). kNN хранит всю выборку и медленно предсказывает. Учесть дисбаланс аварий."),

    # ---------- D: генерация кода (запуск + ручная проверка методологии) ----------
    dict(id="D1", type="D", topic="kNN", source=OWN, difficulty=2, tol=0.03,
         text_ru="Напиши код на Python (sklearn): на датасете load_wine отложи 25% в тест (stratify, "
                 "random_state=42), подбери число соседей k от 1 до 15 для kNN 5-кратной кросс-валидацией и "
                 "выведи accuracy на тесте. Последней строкой программа печатает одно число.",
         text_en="Write Python (sklearn) code: on load_wine hold out 25% for test (stratify, random_state=42), tune "
                 "the number of neighbors k from 1 to 15 for kNN with 5-fold cross-validation and print test "
                 "accuracy. The program's last printed line is a single number.",
         answer="0.7333 (без масштабирования) или 1.0 (со StandardScaler) — оба допустимы"),
    dict(id="D2", type="D", topic="Линейная регрессия", source=OWN, difficulty=2, tol=0.03,
         text_ru="Напиши код на Python (sklearn): на датасете load_diabetes (тест 20%, random_state=42) обучи "
                 "Ridge, подобрав alpha из [0.01, 0.1, 1, 10, 100] кросс-валидацией, и выведи R² на тесте. "
                 "Последней строкой программа печатает одно число.",
         text_en="Write Python (sklearn) code: on load_diabetes (20% test, random_state=42) train Ridge, tuning "
                 "alpha from [0.01, 0.1, 1, 10, 100] with cross-validation, and print test R². The program's last "
                 "printed line is a single number.",
         answer="0.4609 (без масштабирования) или 0.4572 (со StandardScaler) — оба допустимы"),
    dict(id="D3", type="D", topic="Ансамбли / метрики", source=OWN, difficulty=3, tol=0.05,
         text_ru="Напиши код на Python (sklearn): сгенерируй make_classification(n_samples=2000, n_features=20, "
                 "n_informative=5, weights=[0.95], random_state=42), обучи случайный лес (random_state=42) и "
                 "оцени, насколько хорошо он находит редкий класс, на отложенной выборке 25% (random_state=42). "
                 "Последней строкой программа печатает одно число — значение подходящей метрики.",
         text_en="Write Python (sklearn) code: generate make_classification(n_samples=2000, n_features=20, "
                 "n_informative=5, weights=[0.95], random_state=42), train a random forest (random_state=42) and "
                 "evaluate how well it detects the rare class on a 25% hold-out set (random_state=42). The "
                 "program's last printed line is a single number — the value of an appropriate metric.",
         answer="любая метрика редкого класса: F1 0.2941, recall 0.1724 или PR-AUC 0.5388"),
    dict(id="D4", type="D", topic="PCA / лог. регрессия", source=OWN, difficulty=2, tol=0.03,
         text_ru="Напиши код на Python (sklearn): на датасете load_digits (тест 25%, stratify, random_state=42) "
                 "понизь размерность с помощью PCA, сохранив 95% дисперсии, обучи логистическую регрессию и выведи "
                 "accuracy на тесте. Последней строкой программа печатает одно число.",
         text_en="Write Python (sklearn) code: on load_digits (25% test, stratify, random_state=42) reduce "
                 "dimensionality with PCA keeping 95% of the variance, train logistic regression and print test "
                 "accuracy. The program's last printed line is a single number.",
         answer="0.9511 (без масштабирования) или 0.96 (со StandardScaler) — оба допустимы"),
    dict(id="D5", type="D", topic="Кластеризация", source=OWN, difficulty=2, tol=0.03,
         text_ru="Напиши код на Python (sklearn): кластеризуй load_iris на 3 кластера методом KMeans "
                 "(random_state=42), не используя метки при обучении, и оцени совпадение кластеров с настоящими "
                 "видами метрикой adjusted_rand_score. Последней строкой программа печатает одно число.",
         text_en="Write Python (sklearn) code: cluster load_iris into 3 clusters with KMeans (random_state=42) "
                 "without using labels for training, and measure agreement with the true species using "
                 "adjusted_rand_score. The program's last printed line is a single number.",
         answer="0.7302 (без масштабирования) или 0.6201 (со StandardScaler) — оба допустимы"),

    # ---------- F: ловушки ----------
    # F1–F6 — изменённые версии A1–A6 (гипотеза H1)
    dict(id="F1", type="F", topic="Байес", pair=1, source=OWN + " (изменённая A1)", difficulty=1,
         text_ru="Дефект встречается у 2% деталей. Датчик срабатывает на 90% дефектных деталей и на 7% исправных. "
                 "Датчик сработал. Найдите вероятность того, что деталь дефектная." + FMT_NUM,
         text_en="A defect occurs in 2% of parts. A sensor fires on 90% of defective parts and on 7% of good parts. "
                 "The sensor fired. Find the probability that the part is defective." + FMT_NUM_EN,
         answer=bayes(0.02, 0.90, 0.07)),
    dict(id="F2", type="F", topic="Деревья", pair=2, source=OWN + " (изменённая A2)", difficulty=2,
         text_ru="В узле дерева 30 объектов класса 0 и 20 объектов класса 1. Разбиение даёт левый узел "
                 "(25 объектов класса 0; 4 класса 1) и правый узел (5; 16). Найдите прирост информации "
                 "по критерию Джини (взвешенное уменьшение неопределённости Джини)." + FMT_NUM,
         text_en="A tree node contains 30 objects of class 0 and 20 objects of class 1. A split gives a left node "
                 "(25 of class 0; 4 of class 1) and a right node (5; 16). Find the information gain by the Gini "
                 "criterion (weighted decrease of Gini impurity)." + FMT_NUM_EN,
         answer=gini_gain((30, 20), (25, 4), (5, 16))),
    dict(id="F3", type="F", topic="Метрики", pair=3, source=OWN + " (изменённая A3)", difficulty=1,
         text_ru="Матрица ошибок: TP = 45, FP = 15, FN = 5, TN = 435. Найдите F1-меру." + FMT_NUM,
         text_en="Confusion matrix: TP = 45, FP = 15, FN = 5, TN = 435. Find the F1 score." + FMT_NUM_EN,
         answer=f1(45, 15, 5)),
    dict(id="F4", type="F", topic="SVM", pair=4, source=OWN + " (изменённая A4)", difficulty=1,
         text_ru="Для разделяющей гиперплоскости SVM w = (5; 12), b = 3 (канонический вид: на опорных векторах "
                 "|w·x + b| = 1). Найдите полную ширину разделяющей полосы (зазора)." + FMT_NUM,
         text_en="For an SVM separating hyperplane w = (5; 12), b = 3 (canonical form: |w·x + b| = 1 on support "
                 "vectors). Find the full width of the margin." + FMT_NUM_EN,
         answer=2 / math.hypot(5, 12)),
    dict(id="F5", type="F", topic="Деревья", pair=5, source=OWN + " (изменённая A5)", difficulty=2,
         text_ru="Выборка из 14 дней: 9 дней «играть = да», 5 дней «нет». По признаку «ветер»: при слабом ветре "
                 "7 «да» и 1 «нет», при сильном 2 «да» и 4 «нет». Найдите информационный прирост "
                 "(по энтропии, логарифм по основанию 2) при разбиении по признаку «ветер»." + FMT_NUM,
         text_en="A sample of 14 days: 9 days \"play = yes\", 5 days \"no\". By the feature \"wind\": with weak wind "
                 "7 yes and 1 no, with strong wind 2 yes and 4 no. Find the information gain (entropy, log base 2) "
                 "of splitting by \"wind\"." + FMT_NUM_EN,
         answer=info_gain((9, 5), (7, 1), (2, 4))),
    dict(id="F6", type="F", topic="Байес", pair=6, source=OWN + " (изменённая A6: нулевая частота)", difficulty=3,
         text_ru="Наивный байесовский классификатор, 14 дней: 9 «да», 5 «нет». Среди дней «да»: пасмурно 4, "
                 "прохладно 3, высокая влажность 3, сильный ветер 3. Среди дней «нет»: пасмурно 0, прохладно 1, "
                 "высокая влажность 4, сильный ветер 3. Новый день: пасмурно, прохладно, высокая влажность, "
                 "сильный ветер. Найдите апостериорную вероятность класса «нет» (без сглаживания)." + FMT_NUM,
         text_en="Naive Bayes classifier, 14 days: 9 yes, 5 no. Among yes-days: overcast 4, cool 3, high humidity 3, "
                 "strong wind 3. Among no-days: overcast 0, cool 1, high humidity 4, strong wind 3. New day: "
                 "overcast, cool, high humidity, strong wind. Find the posterior probability of class \"no\" "
                 "(no smoothing)." + FMT_NUM_EN,
         answer=nb_posterior_no([9 / 14, 4 / 9, 3 / 9, 3 / 9, 3 / 9], [5 / 14, 0 / 5, 1 / 5, 4 / 5, 3 / 5])),
    # F7–F8 — не хватает данных
    dict(id="F7", type="F", topic="Метрики", source=OWN + " (не хватает данных)", difficulty=2, answer=CANNOT,
         text_ru="В тестовой выборке 500 объектов. Классификатор верно нашёл 40 объектов положительного класса "
                 "(TP = 40) и пропустил 10 (FN = 10). Найдите precision." + FMT_NUM,
         text_en="The test set has 500 objects. The classifier correctly found 40 positive objects (TP = 40) and "
                 "missed 10 (FN = 10). Find precision." + FMT_NUM_EN),
    dict(id="F8", type="F", topic="PCA", source=OWN + " (не хватает данных)", difficulty=2, answer=CANNOT,
         text_ru="Данные с 4 признаками. Два наибольших собственных значения ковариационной матрицы равны 5 и 3. "
                 "Найдите долю дисперсии, объяснённую первой главной компонентой." + FMT_NUM,
         text_en="Data with 4 features. The two largest eigenvalues of the covariance matrix are 5 and 3. Find the "
                 "share of variance explained by the first principal component." + FMT_NUM_EN),
    # F9–F10 — лишние данные
    dict(id="F9", type="F", topic="Линейная регрессия", source=OWN + " (лишние данные)", difficulty=1,
         text_ru="Модель обучена на 1 200 объектах 12 марта, R² на обучении 0,83, признак x3 = 17 в модель не "
                 "вошёл. Модель: ŷ = 2 + 0,5·x1 − 1,5·x2. Найдите прогноз для объекта x1 = 4, x2 = 2, x3 = 17."
                 + FMT_NUM,
         text_en="The model was trained on 1,200 objects on March 12, training R² is 0.83, feature x3 = 17 was not "
                 "included. Model: ŷ = 2 + 0.5·x1 − 1.5·x2. Find the prediction for x1 = 4, x2 = 2, x3 = 17."
                 + FMT_NUM_EN,
         answer=2 + 0.5 * 4 - 1.5 * 2),
    dict(id="F10", type="F", topic="kNN", source=OWN + " (лишние данные)", difficulty=2,
         text_ru="kNN с k = 3, евклидово расстояние, один признак. Обучающие точки: 1 (класс 0), 2,5 (класс 0), "
                 "4 (класс 1), 5,5 (класс 1), 7 (класс 1). Точность модели на валидации 91%, обучение заняло "
                 "0,2 с, в исходном датасете было 30 признаков, но использовался один. Какой класс будет "
                 "предсказан для x = 3,6?" + FMT_NUM,
         text_en="kNN with k = 3, Euclidean distance, one feature. Training points: 1 (class 0), 2.5 (class 0), "
                 "4 (class 1), 5.5 (class 1), 7 (class 1). Validation accuracy is 91%, training took 0.2 s, the "
                 "original dataset had 30 features but only one was used. Which class is predicted for x = 3.6?"
                 + FMT_NUM_EN,
         answer=1),
]

pilot = [
    dict(id="P1", type="A", topic="Метрики", source=OWN, difficulty=1, answer=(20 + 65) / 100,
         text_ru="Матрица ошибок: TP = 20, FP = 5, FN = 10, TN = 65. Найдите accuracy." + FMT_NUM,
         text_en="Confusion matrix: TP = 20, FP = 5, FN = 10, TN = 65. Find accuracy." + FMT_NUM_EN),
    dict(id="P2", type="B", topic="kNN", source=OWN, difficulty=1, answer="ВЕРНО",
         text_ru="Утверждение: «kNN — ленивый алгоритм: на этапе обучения он только запоминает выборку». "
                 "Верно или неверно? Обоснуйте.",
         text_en="Statement: \"kNN is a lazy algorithm: at training time it only memorizes the sample.\" "
                 "True or false? Justify."),
    dict(id="P3", type="F", topic="Метрики", source=OWN, difficulty=1, answer=CANNOT,
         text_ru="Модель ошиблась на 12 объектах тестовой выборки. Найдите accuracy." + FMT_NUM,
         text_en="The model made errors on 12 test objects. Find accuracy." + FMT_NUM_EN),
    dict(id="P4", type="B", topic="Деревья", source=OWN, difficulty=1, answer="НЕВЕРНО",
         text_ru="Утверждение: «Глубокое дерево решений без ограничений обычно имеет высокое смещение». "
                 "Верно или неверно? Обоснуйте.",
         text_en="Statement: \"A deep unrestricted decision tree usually has high bias.\" True or false? Justify."),
    dict(id="P5", type="A", topic="Логистическая регрессия", source=OWN, difficulty=1,
         answer=sigmoid(1 * 2 - 2 * 1 + 0.5),
         text_ru="Логистическая регрессия: w = (1; −2), b = 0,5. Найдите вероятность класса 1 для x = (2; 1)."
                 + FMT_NUM,
         text_en="Logistic regression: w = (1; −2), b = 0.5. Find the probability of class 1 for x = (2; 1)."
                 + FMT_NUM_EN),
]

COLUMNS = ["id", "type", "topic", "text_ru", "text_en", "answer", "tol", "source", "difficulty", "pair"]


def to_frame(rows):
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    numeric = df["answer"].map(lambda a: isinstance(a, (int, float)))
    df.loc[numeric, "answer"] = df.loc[numeric, "answer"].map(lambda a: round(a, 4))
    # допуск: для A/F — 1% относительный (записываем 0.01), для D — абсолютный из задачи
    df["pair"] = df["pair"].astype("Int64")
    df["tol"] = df["tol"].where(df["tol"].notna(), df["type"].map({"A": 0.01, "F": 0.01}))
    return df


if __name__ == "__main__":
    bank, pil = to_frame(tasks), to_frame(pilot)
    bank.to_csv("data/bank.csv", index=False, encoding="utf-8")
    pil.to_csv("data/pilot.csv", index=False, encoding="utf-8")
    print(bank.groupby("type").size().to_string())
    print(bank[["id", "answer"]].to_string(index=False))
