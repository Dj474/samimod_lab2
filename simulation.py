"""Запуск имитационной модели грузового аэропорта (вариант №15).

  python simulation.py                              # серия прогонов, отклики в терминал
  python simulation.py --config config_overload.json # режим перегрузки
  python simulation.py --runs 20                     # другое число прогонов
  python simulation.py --trace                       # один прогон с трассировкой
  python simulation.py --trace-excerpt               # фрагмент трассы в out/

Прогоны на SimPy (Run, Runner) - engine.py,
агенты и ресурсы аэропорта (AirportModel) - airport.py.
"""

import csv
import io
import json
import os
import sys

from airport import AirportModel, RESPONSE_KEYS, RESPONSE_LABELS
from engine import Runner

CSV_KEYS = ["seed"] + RESPONSE_KEYS + [
    "arrived_tons", "delivered_tons", "stored_tons_end", "in_planes_tons",
    "n_departures_high", "n_containers_loaded", "t_end", "drain_time",
]


class Tee:
    """Дублирование вывода в консоль и в файл (файл всегда в UTF-8)."""

    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except UnicodeEncodeError:
                enc = getattr(st, "encoding", None) or "utf-8"
                st.write(s.encode(enc, "replace").decode(enc, "replace"))

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def default_config():
    return {
        "horizon": 8760.0,
        "arrival_rate": 3.0,
        "weight_min": 10.0,
        "weight_max": 30.0,
        "n_normal": 2,
        "n_high": 1,
        "normal_capacity": 100.0,
        "high_capacity": 300.0,
        "load_time_mean": 0.15,
        "flight_time_mean": 6.0,
        "n_runs": 100,
        "seed_trace": 7,
        "drain": True,
    }


def load_config(path="config.json"):
    cfg = default_config()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def scenario_suffix(config_path):
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return "" if stem == "config" else stem.replace("config", "", 1)


def model_factory(cfg):
    return AirportModel(cfg)


def arg_value(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def print_responses(responses):
    for k, v in responses.items():
        print(f"  {k:22s} = {v:.4f}" if isinstance(v, float) else f"  {k:22s} = {v}")


def print_run_row(run):
    r = run.responses
    print(f"{run.seed:5d} | {r['R1_n_departures']:6d} | {r['R2_share_high']:7.4f} | "
          f"{r['R3_avg_store_count']:9.2f} | {r['R4_avg_wait']:9.3f} | "
          f"{r['drain_time']:8.1f}")


def print_summary(runner):
    n = len(runner.runs)
    print("\n" + "=" * 96)
    print(f"СТАТИСТИКА ОТКЛИКОВ ПО {n} ПРОГОНАМ (95% ДИ = m ± t·s/√N)")
    print("=" * 96)
    print(f"{'Отклик':40s} | {'среднее':>11s} | {'СКО':>9s} | {'мин':>10s} | "
          f"{'макс':>10s} | {'±ДИ':>8s}")
    print("-" * 96)
    for key, st in runner.summary(RESPONSE_KEYS).items():
        print(f"{RESPONSE_LABELS[key]:40s} | {st['mean']:11.4f} | {st['std']:9.4f} | "
              f"{st['min']:10.4f} | {st['max']:10.4f} | {st['ci']:8.4f}")
    errs = [run.model.balance()[2] for run in runner.runs]
    print("-" * 96)
    print(f"Баланс «прибыло = вывезено + склад + в самолётах»: "
          f"макс. расхождение {max(errs):.2e} т")


def run_trace(cfg, suffix):
    path = os.path.join("out", f"trace_full{suffix}.txt")
    runner = Runner(model_factory, cfg, 1)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as fh:
        old = sys.stdout
        sys.stdout = Tee(sys.stdout, fh)
        try:
            run = runner.make_run(cfg["seed_trace"], trace=True).simulate()
            lhs, rhs, err = run.model.balance()
            print("\n=== ОТКЛИКИ ПРОГОНА (трассировка) ===")
            print_responses(run.responses)
            print(f"\nБаланс: прибыло {lhs:.1f} т = вывезено + склад + в самолётах "
                  f"{rhs:.1f} т | расхождение {err:.2e} т")
        finally:
            sys.stdout = old
    print(f"\nПолная трассировка сохранена: {path} (UTF-8)")


def run_trace_excerpt(cfg, suffix, lines_limit=80):
    cfg = dict(cfg, horizon=300.0)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        Runner(model_factory, cfg, 1).make_run(cfg["seed_trace"], trace=True).simulate()
    finally:
        sys.stdout = old
    lines = buf.getvalue().splitlines()
    path = os.path.join("out", f"trace_excerpt{suffix}.txt")
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("\n".join(lines[:lines_limit]) + "\n")
    print(f"Фрагмент трассы ({min(lines_limit, len(lines))} строк из {len(lines)}) -> {path}")


def run_replications(cfg, suffix, n_runs):
    print(f"Сценарий: λ={cfg['arrival_rate']} конт./ч, T={cfg['horizon']:.0f} ч, "
          f"обычных {cfg['n_normal']}×{cfg['normal_capacity']:.0f} т, "
          f"повыш. ГП {cfg['n_high']}×{cfg['high_capacity']:.0f} т, "
          f"дообслуживание={'да' if cfg['drain'] else 'нет'}, прогонов {n_runs}")
    print(f"\n{'seed':>5s} | {'R1':>6s} | {'R2':>7s} | {'R3':>9s} | {'R4, ч':>9s} | "
          f"{'дообсл,ч':>8s}")
    print("-" * 59)
    runner = Runner(model_factory, cfg, n_runs)
    runner.run_all(on_run=print_run_row)
    print_summary(runner)

    path = os.path.join("out", f"results_replications{suffix}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_KEYS, extrasaction="ignore")
        wr.writeheader()
        for run in runner.runs:
            wr.writerow(dict(run.responses, seed=run.seed))
    print(f"Результаты: {path}")
    return runner


def main():
    config_path = arg_value("--config", "config.json")
    cfg = load_config(config_path)
    suffix = scenario_suffix(config_path)
    os.makedirs("out", exist_ok=True)

    if "--trace-excerpt" in sys.argv:
        return run_trace_excerpt(cfg, suffix)
    if "--trace" in sys.argv:
        return run_trace(cfg, suffix)
    return run_replications(cfg, suffix, int(arg_value("--runs", cfg["n_runs"])))


if __name__ == "__main__":
    main()
