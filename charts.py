"""Диаграммы результатов имитации (matplotlib -> PNG в out/).

  fig1-fig8  - базовый сценарий (config.json): распределения откликов,
               ящики, сходимость, доверительные интервалы, Gantt самолётов;
  fig9       - динамика откликов в модельном времени, базовый сценарий;
  fig10      - то же для режима перегрузки (config_overload.json).

Графики чувствительности строит sensitivity.py.
"""

import bisect
import math
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from airport import AirportModel, ST_FLYING, ST_LOADING, ST_NAME, ST_READY
from engine import Runner
from simulation import Tee, load_config, run_replications

OUT = "out"
DPI = 130
COL_ST = {ST_READY: "#2ca02c", ST_LOADING: "#ff7f0e", ST_FLYING: "#1f77b4"}
N_REALIZATIONS = 5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
})


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(" ", path)
    return path


def plot_hist(values, nbins, fname, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=nbins, color="#4C72B0", edgecolor="white", alpha=0.9)
    m = statistics.mean(values)
    ax.axvline(m, color="#d62728", lw=2, label=f"среднее ≈ {m:.2f}")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    return savefig(fig, fname)


def plot_series(fname, history, title):
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot([h[0] / 24 for h in history], [h[2] for h in history],
             color="#1f77b4", lw=1.2, label="Вес груза на складе, т")
    ax1.set_xlabel("Время, сут")
    ax1.set_ylabel("Вес на складе, т", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot([h[0] / 24 for h in history], [h[3] for h in history],
             color="#d62728", lw=1.6, label="Накопленное число вылетов")
    ax2.set_ylabel("Накопленное число вылетов", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(title, fontsize=15)
    ln1, lb1 = ax1.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(ln1 + ln2, lb1 + lb2, loc="upper left")
    fig.tight_layout()
    return savefig(fig, fname)


def plot_boxpanels(fname, panels, title):
    cols = 2
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows))
    axes = list(axes.reshape(-1))
    for ax, (vals, label) in zip(axes, panels):
        bp = ax.boxplot(vals, orientation="vertical", widths=0.5,
                        patch_artist=True, showmeans=True)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][0].set_alpha(0.7)
        bp["medians"][0].set_color("#d62728")
        bp["means"][0].set_marker("D")
        bp["means"][0].set_markerfacecolor("#d62728")
        bp["means"][0].set_markeredgecolor("#d62728")
        ax.set_title(label, fontsize=12)
        ax.set_xticks([])
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout()
    return savefig(fig, fname)


def plot_convergence(fname, series, title):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for vals, col, lab in series:
        acc, run = 0.0, []
        for i, v in enumerate(vals, 1):
            acc += v
            run.append(acc / i)
        ax.plot(range(1, len(vals) + 1), [v / run[-1] for v in run],
                color=col, lw=2, label=lab)
    ax.axhline(1.0, color="#555", lw=1, ls="--")
    ax.set_xlabel("Число прогонов")
    ax.set_ylabel("Скользящее среднее / итоговое среднее")
    ax.set_title(title, fontsize=15)
    ax.legend(loc="best")
    fig.tight_layout()
    return savefig(fig, fname)


def plot_ci_bars(fname, panels, title):
    cols = 2
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows))
    axes = list(axes.reshape(-1))
    for ax, (lab, st, dig) in zip(axes, panels):
        ax.bar([0], [st["mean"]], yerr=[st["ci"]], width=0.55, color="#4C72B0",
               capsize=6, alpha=0.85, error_kw={"ecolor": "#d62728", "lw": 2})
        ax.set_title(lab, fontsize=12)
        ax.set_xticks([])
        ax.text(0, st["mean"], f"{st['mean']:.{dig}f} ±{st['ci']:.{dig}f}",
                ha="center", va="bottom", color="#222", fontsize=10)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout()
    return savefig(fig, fname)


def plot_gantt(fname, plane_series, n_normal, n_high, title, tmax):
    by_plane = {}
    for t, pid, st in plane_series:
        if t <= tmax:
            by_plane.setdefault(pid, []).append((t, st))
    n_planes = n_normal + n_high
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for pid in range(n_planes):
        seq = sorted(by_plane.get(pid, [(0.0, ST_READY)]))
        for (t0, st0), (t1, _) in zip(seq, seq[1:]):
            ax.barh(pid, t1 - t0, left=t0, height=0.7, color=COL_ST[st0], edgecolor="none")
        t0, st0 = seq[-1]
        ax.barh(pid, tmax - t0, left=t0, height=0.7, color=COL_ST[st0], edgecolor="none")
    names = [f"Обычный {i + 1}" for i in range(n_normal)]
    names += [f"Повыш. {i - n_normal + 1}" for i in range(n_normal, n_planes)]
    ax.set_yticks(range(n_planes))
    ax.set_yticklabels(names)
    ax.set_xlim(0, tmax)
    xt = ax.get_xticks()
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{x / 24:.0f}" for x in xt])
    ax.set_xlabel("Время, сутки")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL_ST[s]) for s in COL_ST]
    ax.legend(handles, [ST_NAME[s] for s in COL_ST], loc="upper right")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return savefig(fig, fname)


def sample(history, grid):
    """Значения ступенчатых траекторий истории в точках сетки (только для
    рисования; продвижение времени в модели — по событиям)."""
    times = [h[0] for h in history]
    q, r3, r4 = [], [], []
    for g in grid:
        i = bisect.bisect_right(times, g) - 1
        if i < 0:
            q.append(0.0)
            r3.append(0.0)
            r4.append(0.0)
            continue
        t, n_store, _, _, area, wait_sum, n_loaded = history[i]
        q.append(n_store)
        r3.append((area + n_store * (g - t)) / g if g > 0 else 0.0)
        r4.append(wait_sum / n_loaded if n_loaded else 0.0)
    return q, r3, r4


def plot_dynamics(fname, runs, horizon, title):
    t_end = max(run.time for run in runs)
    grid = [t_end * i / 1500 for i in range(1, 1501)]
    curves = [sample(run.model.history, grid) for run in runs]
    days = [g / 24 for g in grid]
    panels = [
        (0, "Q(t): контейнеров на складе", "контейнеров"),
        (1, "R3(t) = ∫Q dt / t: накопленное среднее", "контейнеров"),
        (2, "R4(t): накопленное среднее ожидание", "часов"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, (k, label, unit) in zip(axes, panels):
        for i, c in enumerate(curves):
            ax.plot(days, c[k], lw=0.8, alpha=0.55,
                    label=f"реализация seed={runs[i].seed}")
        mean = [statistics.mean(c[k][j] for c in curves) for j in range(len(grid))]
        ax.plot(days, mean, color="black", lw=2.2,
                label=f"среднее по {len(runs)} реализациям")
        ax.axvline(horizon / 24, color="#d62728", ls="--", lw=1.4,
                   label="конец поступлений (T)")
        ax.set_title(label, fontsize=13)
        ax.set_ylabel(unit)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[-1].set_xlabel("Модельное время, сут")
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout()
    return savefig(fig, fname)


def recorded_runs(cfg, n):
    runner = Runner(lambda c: AirportModel(c, record=True), cfg, n)
    return runner.run_all()


def _run():
    cfg = load_config("config.json")
    cfg_over = load_config("config_overload.json")

    runner = run_replications(cfg, "", cfg["n_runs"])
    R = {k: runner.values(k) for k in
         ["R1_n_departures", "R2_share_high", "R3_avg_store_count", "R4_avg_wait"]}
    stats = runner.summary(list(R))

    trace_seed = int(cfg["seed_trace"])
    rep = Runner(lambda c: AirportModel(c, record=True), cfg, 1).make_run(trace_seed).simulate()

    print("Рисунки:")
    plot_hist(R["R1_n_departures"], 12, "fig1_hist_R1.png",
              "Распределение числа вылетевших рейсов по повторным прогонам (R1)",
              "Число рейсов за прогон", "Частота")
    plot_hist(R["R2_share_high"], 14, "fig2_hist_R2.png",
              "Распределение доли рейсов повышенной грузоподъёмности (R2)",
              "Доля рейсов повышенной грузоподъёмности", "Частота")
    plot_series("fig3_series.png", rep.model.history,
                f"Динамика накопителя и вылетов (прогон seed={trace_seed})")
    plot_hist(R["R4_avg_wait"], 14, "fig4_hist_R4.png",
              "Распределение среднего времени ожидания контейнера (R4)",
              "Среднее время ожидания, ч", "Частота")
    plot_boxpanels("fig5_boxplots.png",
                   [(R["R1_n_departures"], "R1: число рейсов (адд.)"),
                    (R["R2_share_high"], "R2: доля рейсов повыш. ГП (адд.)"),
                    (R["R3_avg_store_count"], "R3: ср. контейнеров на складе (непр.)"),
                    (R["R4_avg_wait"], "R4: ср. ожидание контейнера, ч (дискр.)")],
                   "Разброс откликов по повторным прогонам (ящик с усами)")
    plot_convergence("fig6_conv.png",
                     [(R["R1_n_departures"], "#1f77b4", "R1 (число рейсов)"),
                      (R["R2_share_high"], "#9467bd", "R2 (доля повыш. ГП)"),
                      (R["R3_avg_store_count"], "#d62728", "R3 (контейнеров на складе)"),
                      (R["R4_avg_wait"], "#2ca02c", "R4 (время ожидания)")],
                     "Сходимость оценок откликов при увеличении числа прогонов")
    plot_ci_bars("fig7_CI.png",
                 [("R1: число рейсов (адд.)", stats["R1_n_departures"], 1),
                  ("R2: доля повыш. ГП (адд.)", stats["R2_share_high"], 4),
                  ("R3: контейнеров на складе (непр.)", stats["R3_avg_store_count"], 2),
                  ("R4: время ожидания, ч (дискр.)", stats["R4_avg_wait"], 3)],
                 "Оценки откликов: среднее по прогонам и 95%-е доверительные интервалы")
    plot_gantt("fig8_gantt.png", rep.model.plane_series, cfg["n_normal"], cfg["n_high"],
               f"Квазипараллельные процессы: состояния самолётов "
               f"(первый месяц, прогон seed={trace_seed})",
               min(cfg["horizon"], 720))

    plot_dynamics("fig9_dynamics_base.png", recorded_runs(cfg, N_REALIZATIONS),
                  cfg["horizon"],
                  f"Динамика откликов, базовый режим (λ={cfg['arrival_rate']} конт./ч): "
                  f"переходный период → стационар")
    plot_dynamics("fig10_dynamics_overload.png", recorded_runs(cfg_over, N_REALIZATIONS),
                  cfg_over["horizon"],
                  f"Динамика откликов, перегрузка (λ={cfg_over['arrival_rate']} конт./ч): "
                  f"стационара нет, рост до T, затем дообслуживание")
    print("Готово.")


def main():
    os.makedirs(OUT, exist_ok=True)
    log_path = os.path.join(OUT, "charts_log.txt")
    with open(log_path, "w", encoding="utf-8-sig", newline="\n") as fh:
        old = sys.stdout
        sys.stdout = Tee(sys.stdout, fh)
        try:
            _run()
        finally:
            sys.stdout = old
    print(f"Лог сохранён: {log_path} (UTF-8)")


if __name__ == "__main__":
    main()
