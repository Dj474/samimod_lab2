"""Прогоны имитационной модели на SimPy.

Календарь событий и продвижение времени по ближайшему событию выполняет
simpy.Environment: каждое событие (simpy.Event, env.timeout, запрос к
ресурсу) попадает в его очередь, env.run() берёт событие с минимальным
временем и возобновляет ожидающие его процессы. Постоянного шага нет.

Предметная модель (см. airport.py) должна реализовать:

  model.start(run)   - зарегистрировать процессы агентов в run.env;
  model.finish(run)  - закрыть статистику в момент окончания прогона;
  model.responses()  - вернуть словарь откликов.

Классы:
  Run    - единичный прогон модели (своё simpy.Environment и датчик);
  Runner - серия независимых прогонов (репликаций) и их статистика.
"""

import math
import random
import statistics

import simpy


class Run:
    """Единичный прогон модели.

    run_time - горизонт поступления заявок. При drain=True после горизонта
    новые заявки не генерируются, а прогон продолжается, пока в календаре
    SimPy есть события (дообслуживание). При drain=False прогон обрывается
    в момент run_time.
    """

    def __init__(self, model, run_time, seed, trace=False, drain=True):
        self.model = model
        self.run_time = run_time
        self.seed = seed
        self.trace = trace
        self.drain = drain
        self.env = simpy.Environment()
        self.rng = random.Random(seed)
        self.responses = None

    @property
    def time(self):
        return self.env.now

    # --------------------------------------------------- случайные величины
    def exp(self, mean):
        return -mean * math.log(1.0 - self.rng.random())

    def uniform(self, a, b):
        return a + self.rng.random() * (b - a)

    # -------------------------------------------------------------- прогон
    def simulate(self):
        self.model.start(self)
        if self.drain:
            self.env.run()
        else:
            self.env.run(until=self.run_time)
        self.model.finish(self)
        self.responses = self.model.responses()
        return self


def t_quantile_95(n):
    """Квантиль t-распределения (двусторонний 95%) для n наблюдений."""
    if n < 2:
        return float("inf")
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
        20: 2.086, 30: 2.042, 40: 2.021, 60: 2.0, 120: 1.98, 1000: 1.96,
    }
    df = n - 1
    if df in table:
        return table[df]
    keys = sorted(table)
    if df > keys[-1]:
        return 1.96
    for a, b in zip(keys, keys[1:]):
        if a <= df <= b:
            f = (df - a) / (b - a)
            return table[a] * (1 - f) + table[b] * f
    return 1.96


def describe(values):
    """Среднее, СКО, минимум, максимум и полуширина 95% ДИ."""
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "ci": t_quantile_95(n) * std / math.sqrt(n) if n > 1 else 0.0,
    }


class Runner:
    """Серия независимых прогонов одной модели с разными seed."""

    def __init__(self, model_factory, cfg, n_runs, first_seed=1):
        self.model_factory = model_factory
        self.cfg = cfg
        self.n_runs = n_runs
        self.first_seed = first_seed
        self.runs = []

    def make_run(self, seed, trace=False):
        return Run(self.model_factory(self.cfg), self.cfg["horizon"], seed,
                   trace=trace, drain=self.cfg.get("drain", True))

    def run_all(self, on_run=None):
        self.runs = []
        for seed in range(self.first_seed, self.first_seed + self.n_runs):
            run = self.make_run(seed).simulate()
            self.runs.append(run)
            if on_run:
                on_run(run)
        return self.runs

    def values(self, key):
        return [run.responses[key] for run in self.runs]

    def summary(self, keys):
        return {key: describe(self.values(key)) for key in keys}
