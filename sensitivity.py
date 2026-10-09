"""Чувствительность откликов R1-R4 к входным параметрам модели.

Для каждого значения варьируемого параметра (остальные — как в config.json)
выполняется серия независимых прогонов через Runner; на графиках —
среднее по прогонам и 95%-й доверительный интервал.

  python sensitivity.py            # 20 прогонов на точку
  python sensitivity.py --runs 10
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from airport import RESPONSE_KEYS, RESPONSE_LABELS
from engine import Runner
from simulation import Tee, arg_value, load_config, model_factory

OUT = "out"

EXPERIMENTS = [
    ("arrival_rate", "Интенсивность поступления контейнеров λ, конт./ч",
     [1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0], "fig11_sens_arrival_rate.png"),
    ("flight_time_mean", "Среднее время рейса, ч",
     [3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "fig12_sens_flight_time.png"),
    ("n_normal", "Число самолётов обычной грузоподъёмности",
     [1, 2, 3, 4], "fig13_sens_n_normal.png"),
]


def sweep(base_cfg, param, values, n_runs):
    points = []
    for v in values:
        cfg = dict(base_cfg, **{param: v})
        runner = Runner(model_factory, cfg, n_runs)
        runner.run_all()
        st = runner.summary(RESPONSE_KEYS)
        points.append((v, st))
        print(f"{param}={v:<6} | " + " | ".join(
            f"{k.split('_')[0]}={st[k]['mean']:10.3f} ±{st[k]['ci']:.3f}"
            for k in RESPONSE_KEYS))
    return points


def plot_sweep(points, param_label, fname, base_value, n_runs):
    xs = [v for v, _ in points]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, key in zip(axes.reshape(-1), RESPONSE_KEYS):
        means = [st[key]["mean"] for _, st in points]
        cis = [st[key]["ci"] for _, st in points]
        ax.errorbar(xs, means, yerr=cis, marker="o", color="#1f77b4",
                    ecolor="#d62728", capsize=5, lw=2)
        ax.axvline(base_value, color="#888", ls="--", lw=1, label="базовое значение")
        ax.set_title(RESPONSE_LABELS[key], fontsize=12)
        ax.set_xlabel(param_label)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Чувствительность откликов: {param_label}\n"
                 f"(среднее ± 95% ДИ по {n_runs} прогонам на точку)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(" ", path)


def _run(n_runs):
    base = load_config("config.json")
    rows = []
    for param, label, values, fname in EXPERIMENTS:
        print(f"\n=== Варьируется {param} ({label}), прогонов на точку: {n_runs} ===")
        points = sweep(base, param, values, n_runs)
        plot_sweep(points, label, fname, base[param], n_runs)
        for v, st in points:
            row = {"param": param, "value": v}
            for k in RESPONSE_KEYS:
                row[f"{k}_mean"] = st[k]["mean"]
                row[f"{k}_ci"] = st[k]["ci"]
            rows.append(row)
    path = os.path.join(OUT, "sensitivity.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nТаблица: {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    n_runs = int(arg_value("--runs", 20))
    log_path = os.path.join(OUT, "sensitivity_log.txt")
    with open(log_path, "w", encoding="utf-8-sig", newline="\n") as fh:
        old = sys.stdout
        sys.stdout = Tee(sys.stdout, fh)
        try:
            _run(n_runs)
        finally:
            sys.stdout = old


if __name__ == "__main__":
    main()
