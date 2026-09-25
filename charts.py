"""Генерация диаграмм результатов имитации (matplotlib -> PNG).

Читает out/results_replications.csv, дополнительно запускает один
представительный прогон (для временного ряда и Gantt-диаграммы) и строит
восемь диаграмм в папке out/.
"""

import csv
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import Simulation, load_config

OUT = "out"
DPI = 130
ST_READY, ST_LOADING, ST_FLYING = 0, 1, 2
ST_NAME = {ST_READY: "ГОТОВ", ST_LOADING: "ЗАГРУЗКА", ST_FLYING: "РЕЙС"}
COL_ST = {ST_READY: "#2ca02c", ST_LOADING: "#ff7f0e", ST_FLYING: "#1f77b4"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
})


def pct(vals, p):
    s = sorted(vals)
    if not s:
        return 0.0
    idx = (len(s) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return s[lo]
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def t_student(n):
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
    if df < keys[0]:
        return table[keys[0]]
    if df > keys[-1]:
        return 1.96
    for a, b in zip(keys, keys[1:]):
        if a <= df <= b:
            f = (df - a) / (b - a)
            return table[a] * (1 - f) + table[b] * f
    return 1.96


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


def plot_series(fname, store_w, cum_dep):
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot([x / 24 for x, _ in store_w], [y for _, y in store_w],
             color="#1f77b4", lw=1.2, label="Вес груза на складе, т")
    ax1.set_xlabel("Время, сут")
    ax1.set_ylabel("Вес на складе, т", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot([x / 24 for x, _ in cum_dep], [y for _, y in cum_dep],
             color="#d62728", lw=1.6, label="Накопленное число вылетов")
    ax2.set_ylabel("Накопленное число вылетов", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Динамика накопителя и вылетов (прогон seed=7)", fontsize=15)
    ln1, lb1 = ax1.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(ln1 + ln2, lb1 + lb2, loc="upper left")
    fig.tight_layout()
    return savefig(fig, fname)


def plot_boxpanels(fname, panels, title):
    n = len(panels)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes = list(axes.reshape(-1))
    for ax, (key, vals, label) in zip(axes, panels):
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


def plot_convergence(fname, series, title, labels):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n = max(len(v) for v, _ in series)
    for (vals, col), lab in zip(series, labels):
        acc = 0.0
        run = []
        for i, v in enumerate(vals, 1):
            acc += v
            run.append(acc / i)
        ax.plot(range(1, n + 1), run, color=col, lw=2, label=lab)
    ax.set_xlabel("Число прогонов")
    ax.set_ylabel("Скользящее среднее отклика")
    ax.set_title(title, fontsize=15)
    ax.legend(loc="best")
    fig.tight_layout()
    return savefig(fig, fname)


def plot_ci_bars(fname, panels, title):
    n = len(panels)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes = list(axes.reshape(-1))
    for ax, (lab, m, ci, dig) in zip(axes, panels):
        ax.bar([0], [m], yerr=[ci], width=0.55, color="#4C72B0",
               capsize=6, alpha=0.85, error_kw={"ecolor": "#d62728", "lw": 2})
        ax.set_title(lab, fontsize=12)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.text(0, m, f"{m:.{dig}f} ±{ci:.{dig}f}",
                ha="center", va="bottom", color="#222", fontsize=10)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout()
    return savefig(fig, fname)


def plot_gantt(fname, plane_series, n_normal, n_high, title, tmax):
    by_plane = {}
    for t, pid, st in plane_series:
        if t > tmax:
            continue
        by_plane.setdefault(pid, []).append((t, st))
    n_planes = n_normal + n_high
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for pid in range(n_planes):
        seq = sorted(by_plane.get(pid, [(0.0, ST_READY)]))
        for (t0, st0), (t1, st1) in zip(seq, seq[1:]):
            ax.barh(pid, t1 - t0, left=t0, height=0.7, color=COL_ST[st0],
                    edgecolor="none")
        t0, st0 = seq[-1]
        ax.barh(pid, tmax - t0, left=t0, height=0.7, color=COL_ST[st0],
                edgecolor="none")
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


def read_results():
    path = os.path.join(OUT, "results_replications.csv")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    from simulation import main as sim_main
    sim_main()

    rows = read_results()
    n = len(rows)

    R1 = [float(r["R1_n_departures"]) for r in rows]
    R2 = [float(r["R2_share_high"]) for r in rows]
    R3 = [float(r["R3_avg_store_count"]) for r in rows]
    R4 = [float(r["R4_avg_wait"]) for r in rows]
    R5 = [float(r["R5_avg_loading_ops"]) for r in rows]
    R6 = [float(r["R6_delivered_tons"]) for r in rows]

    rep = Simulation(cfg, seed=int(cfg.get("seed_trace", 7)), record=True)
    rep.run()

    print("Рисунки:")
    plot_hist(R1, 12, "fig1_hist_R1.png",
              "Распределение числа вылетевших рейсов по повторным прогонам (R1)",
              "Число рейсов за горизонт", "Частота")
    plot_hist(R2, 14, "fig2_hist_R2.png",
              "Распределение доли рейсов повышенной грузоподъёмности (R2)",
              "Доля рейсов повышенной грузоподъёмности", "Частота")
    plot_series("fig3_series.png",
                [(t, w) for t, c, w, d in rep.history],
                [(t, d) for t, c, w, d in rep.history])
    plot_hist(R4, 14, "fig4_hist_R4.png",
              "Распределение среднего времени ожидания контейнера (R4)",
              "Среднее время ожидания, ч", "Частота")
    plot_boxpanels("fig5_boxplots.png",
                   [("R1", R1, "R1: число рейсов (дискр.)"),
                    ("R2", R2, "R2: доля рейсов повыш. ГП (дискр.)"),
                    ("R3", R3, "R3: ср. число контейнеров на складе (непр.)"),
                    ("R4", R4, "R4: ср. время ожидания контейнера, ч (непр.)"),
                    ("R5", R5, "R5: ср. число параллельных загрузок (непр.)"),
                    ("R6", R6, "R6: вывезено груза, тыс. т (непр.)")],
                   "Разброс откликов по повторным прогонам (ящик с усами)")
    plot_convergence("fig6_conv.png",
                     [(R1, "#1f77b4"), (R3, "#d62728"), (R4, "#2ca02c")],
                     "Сходимость оценок откликов при увеличении числа прогонов",
                     ["R1 (число рейсов)", "R3 (контейнеров на складе)",
                      "R4 (время ожидания, ч)"])
    panels = []
    for lab, vals, dig, scale in [
        ("R1: число рейсов", R1, 1, 1),
        ("R2: доля повыш. ГП", R2, 4, 1),
        ("R3: контейнеров на складе", R3, 2, 1),
        ("R4: время ожидания, ч", R4, 3, 1),
        ("R5: параллельных загрузок", R5, 4, 1),
        ("R6: вывезено, тыс. т", R6, 2, 1e-3),
    ]:
        m = statistics.mean(vals) * scale
        s = statistics.stdev(vals) * scale if n > 1 else 0.0
        panels.append((lab, m, t_student(n) * s / math.sqrt(n), dig))
    plot_ci_bars("fig7_CI.png", panels,
                 "Оценки откликов: среднее по прогонам и 95%-е доверительные интервалы")
    plot_gantt("fig8_gantt.png", rep.plane_series, cfg["n_normal"], cfg["n_high"],
               "Квазипараллельные процессы: состояния самолётов "
               "(первый месяц, прогон seed=7)",
               min(cfg["horizon"], 720))
    print("Готово.")


if __name__ == "__main__":
    main()