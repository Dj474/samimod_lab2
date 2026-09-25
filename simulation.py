"""Имитационная модель грузового аэропорта (вариант №15).

Дискретно-событийное моделирование с продвижением времени по ближайшему
событию (next-event). Постоянный временной шаг не используется.

Все случайные величины генерируются единственным датчиком random.Random.

События:
  ARRIVAL    - поступление контейнера
  LOAD_STEP  - загрузка одного контейнера на самолёт
  RETURN     - возврат самолёта из рейса (готов к загрузке)
  END        - конец прогона

Квазипараллельные процессы: CONTAINER_FLOW, PLANE_i, DISPATCHER.
"""

import json
import os
import random
import math
import sys
import csv
from collections import deque
from heapq import heappop, heappush

ST_READY, ST_LOADING, ST_FLYING = 0, 1, 2
ST_NAME = {ST_READY: "ГОТОВ", ST_LOADING: "ЗАГРУЗКА", ST_FLYING: "РЕЙС"}
NORMAL, HIGH = "normal", "high"


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


class Plane:
    __slots__ = ("id", "type", "capacity", "state", "loaded", "n_containers", "scheduled")

    def __init__(self, pid, ptype, capacity):
        self.id = pid
        self.type = ptype
        self.capacity = capacity
        self.state = ST_READY
        self.loaded = 0.0
        self.n_containers = 0
        self.scheduled = False


class Simulation:
    """Один прогон имитационной модели."""

    def __init__(self, cfg, seed, trace=False, record=False):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.trace = trace
        self.record = record

        self.t = 0.0
        self.T = cfg["horizon"]
        self.calendar = []
        self.seq = 0
        self.plane_series = []

        self.planes = []
        for i in range(cfg["n_normal"]):
            self.planes.append(Plane(i, NORMAL, cfg["normal_capacity"]))
        for i in range(cfg["n_high"]):
            self.planes.append(Plane(cfg["n_normal"] + i, HIGH, cfg["high_capacity"]))
        for p in self.planes:
            self.plane_series.append((0.0, p.id, ST_READY))

        self.store = deque()
        self.store_weight = 0.0
        self.arrived_weight = 0.0
        self.loaded_weight = 0.0

        self.n_departures = 0
        self.n_departures_high = 0
        self.wait_times = []
        self.busy_time = 0.0
        self.pending_loads = 0

        self.area_count = 0.0
        self.area_weight = 0.0
        self.t_last_area = 0.0

        self.history = []

        self.rng = random.Random(seed)
        self._schedule(self._exp(self.cfg["arrival_rate"]), "ARRIVAL", None)
        self._schedule(self.T, "END", None)
        self._try_dispatch(reason="start")
        if self.trace:
            self._trace_header()

    # ------------------------------------------------------------ случайные
    def _exp(self, rate):
        if rate <= 0:
            return 0.0
        return -math.log(1.0 - self.rng.random()) / rate

    def _uniform(self, a, b):
        return a + self.rng.random() * (b - a)

    # ------------------------------------------------------------- события
    def _schedule(self, delay, kind, arg):
        heappush(self.calendar, (self.t + delay, self.seq, kind, arg))
        self.seq += 1

    def _next(self):
        self.t, _, kind, arg = heappop(self.calendar)
        return kind, arg

    # --------------------------------------------------------------- учёт
    def _accum_area(self):
        dt = self.t - self.t_last_area
        self.area_count += len(self.store) * dt
        self.area_weight += self.store_weight * dt
        self.t_last_area = self.t

    def _trace(self, proc, msg):
        states = " ".join(
            f"{'N' if p.type == NORMAL else 'H'}{p.id}:{ST_NAME[p.state]}"
            for p in self.planes
        )
        print(
            f"t={self.t:9.3f} | {proc:16s} | {msg:55s} | "
            f"склад[шт]={len(self.store):4d} вес={self.store_weight:8.1f} т | {states}"
        )

    def _trace_header(self):
        print("=" * 120)
        print(
            "КВАЗИПАРАЛЛЕЛЬНАЯ ТРАССА (t | процесс | действие | склад | состояния самолётов)"
        )
        print("=" * 120)
        self._trace("DISPATCHER", "старт прогона, попытка назначения")

    # ------------------------------------------------------- правила логики
    def _ready_normal(self):
        return [p for p in self.planes if p.type == NORMAL and p.state == ST_READY]

    def _ready_high(self):
        return [p for p in self.planes if p.type == HIGH and p.state == ST_READY]

    def _try_dispatch(self, reason=""):
        """Правило управляющего: обычные самолёты в приоритете."""
        norm = self._ready_normal()
        if norm and self.store_weight >= self.cfg["normal_capacity"]:
            p = norm[0]
            p.state = ST_LOADING
            self.plane_series.append((self.t, p.id, ST_LOADING))
            if self.trace:
                self._trace(f"PLANE_{p.id}", f"назначен ({NORMAL}), {p.capacity} т")
            self._advance_loading()
            return
        if not norm:
            high = self._ready_high()
            if high and self.store_weight >= self.cfg["high_capacity"]:
                p = high[0]
                p.state = ST_LOADING
                self.plane_series.append((self.t, p.id, ST_LOADING))
                if self.trace:
                    self._trace(f"PLANE_{p.id}", f"назначен ({HIGH}), {p.capacity} т")
                self._advance_loading()
                return

    def _advance_loading(self):
        """Продвинуть загрузку самолётов, находящихся в состоянии ЗАГРУЗКА.

        Планируется не более одного события LOAD_STEP на самолёт (флаг
        p.scheduled) и не более числа контейнеров, имеющихся на складе на
        момент планирования (self.pending_loads < |склад|), что исключает
        обращение к пустой очереди и двойное планирование.
        """
        cfg = self.cfg
        for p in self.planes:
            if p.state == ST_LOADING and not p.scheduled:
                if p.loaded < p.capacity and self.store and self.pending_loads < len(self.store):
                    dt = self._exp(1.0 / cfg["load_time_mean"])
                    self.busy_time += dt
                    self.pending_loads += 1
                    p.scheduled = True
                    self._schedule(dt, "LOAD_STEP", p)
                    if self.trace:
                        self._trace(
                            f"PLANE_{p.id}",
                            f"взят контейнер на загрузку "
                            f"(загружено {p.loaded:.0f}/{p.capacity} т)",
                        )

    # ------------------------------------------------------------ обработчик
    def run(self):
        while self.calendar:
            kind, arg = self._next()
            if kind == "END":
                self._accum_area()
                break
            elif kind == "ARRIVAL":
                self._on_arrival()
            elif kind == "LOAD_STEP":
                self._on_load_step(arg)
            elif kind == "RETURN":
                self._on_return(arg)
        self._finalize()
        return self

    def _on_arrival(self):
        cfg = self.cfg
        w = self._uniform(cfg["weight_min"], cfg["weight_max"])
        self.store.append((w, self.t))
        self.store_weight += w
        self.arrived_weight += w
        self._accum_area()
        if self.record:
            self.history.append(
                (self.t, len(self.store), self.store_weight, self.n_departures)
            )
        if self.trace:
            self._trace(
                "CONTAINER_FLOW",
                f"поступил контейнер весом {w:.1f} т",
            )
        self._schedule(self._exp(cfg["arrival_rate"]), "ARRIVAL", None)
        self._advance_loading()
        self._try_dispatch(reason="arrival")

    def _on_load_step(self, p):
        p.scheduled = False
        self.pending_loads -= 1
        w, t_arr = self.store.popleft()
        p.loaded += w
        p.n_containers += 1
        self.store_weight -= w
        self._accum_area()
        self.wait_times.append(self.t - t_arr)
        if self.record:
            self.history.append(
                (self.t, len(self.store), self.store_weight, self.n_departures)
            )
        if self.trace:
            self._trace(
                f"PLANE_{p.id}",
                f"загружен контейнер {w:.1f} т, уже {p.loaded:.0f}/{p.capacity} т",
            )
        if p.loaded >= p.capacity:
            self._departure(p)
        else:
            self._advance_loading()
            self._try_dispatch(reason="load")

    def _departure(self, p):
        w = p.loaded
        self.n_departures += 1
        if p.type == HIGH:
            self.n_departures_high += 1
        self.loaded_weight += w
        p.loaded = 0.0
        p.n_containers = 0
        p.state = ST_FLYING
        if self.trace:
            kind = NORMAL if p.type == NORMAL else HIGH
            self._trace(
                f"PLANE_{p.id}",
                f"ВЫЛЕТ (рейс #{self.n_departures}, {kind}, груз {w:.0f} т)",
            )
        self.plane_series.append((self.t, p.id, ST_FLYING))
        self._schedule(self._exp(1.0 / self.cfg["flight_time_mean"]), "RETURN", p)
        self._try_dispatch(reason="departure")

    def _on_return(self, p):
        p.state = ST_READY
        if self.trace:
            self._trace(f"PLANE_{p.id}", "возврат из рейса, готов к загрузке")
        self.plane_series.append((self.t, p.id, ST_READY))
        self._try_dispatch(reason="return")

    # ----------------------------------------------------------- результаты
    def _finalize(self):
        for p in self.planes:
            self.plane_series.append((self.t, p.id, p.state))
        self.in_planes_tons = sum(p.loaded for p in self.planes
                                  if p.state == ST_LOADING)
        self.responses = {
            "R1_n_departures": self.n_departures,
            "R2_share_high": (self.n_departures_high / self.n_departures
                              if self.n_departures else 0.0),
            "R3_avg_store_count": self.area_count / max(self.t, 1e-12),
            "R4_avg_wait": (sum(self.wait_times) / len(self.wait_times)
                            if self.wait_times else 0.0),
            "R5_avg_loading_ops": self.busy_time / max(self.t, 1e-12),
            "R6_delivered_tons": self.loaded_weight,
            "R7_stored_tons_end": self.store_weight,
            "R8_in_planes_tons": self.in_planes_tons,
            "arrived_tons": self.arrived_weight,
            "n_departures_high": self.n_departures_high,
            "n_containers_loaded": len(self.wait_times),
        }
        if self.record:
            self.history = [tuple(h) for h in self.history]

    def consistency_balance(self):
        r = self.responses
        lhs = r["arrived_tons"]
        rhs = (r["R6_delivered_tons"] + r["R7_stored_tons_end"]
               + r.get("R8_in_planes_tons", 0.0))
        return lhs, rhs, abs(lhs - rhs)


# ----------------------------------------------------------------------------
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
    }


def load_config(path="config.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = default_config()
        cfg.update(data)
        return cfg
    return default_config()


def main():
    cfg = load_config()
    os.makedirs("out", exist_ok=True)

    if "--trace-excerpt" in sys.argv:
        import io
        cfg2 = dict(cfg)
        cfg2["horizon"] = 300.0
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            sim = Simulation(cfg2, seed=cfg.get("seed_trace", 7), trace=True)
            sim.run()
        finally:
            sys.stdout = old
        lines = buf.getvalue().splitlines()
        path = os.path.join("out", "trace_excerpt.txt")
        with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
            f.write("\n".join(lines[:80]) + "\n")
        print(f"Фрагмент трассы ({min(80, len(lines))} строк из "
              f"{len(lines)}) -> {path}")
        return sim

    if "--trace" in sys.argv:
        path = os.path.join("out", "trace_full.txt")
        with open(path, "w", encoding="utf-8-sig", newline="\n") as fh:
            old = sys.stdout
            sys.stdout = Tee(sys.stdout, fh)
            try:
                sim = Simulation(cfg, seed=cfg.get("seed_trace", 7),
                                 trace=True, record=True)
                sim.run()
                print("\n=== ИТОГОВЫЙ БАЛАНС ПРОГОНА (трассировка) ===")
                lhs, rhs, err = sim.consistency_balance()
                print(f"прибыло тонн: {lhs:.1f} | "
                      f"вывезено+остаток: {rhs:.1f} | расхождение: {err:.6f}")
                for k, v in sim.responses.items():
                    print(f"{k:26s} = {v:.4f}" if isinstance(v, float)
                          else f"{k:26s} = {v}")
            finally:
                sys.stdout = old
        print(f"\nПолная трассировка сохранена: {path} (UTF-8)")
        return sim

    n = cfg.get("n_runs", 1)
    rows = []
    for run in range(1, n + 1):
        sim = Simulation(cfg, seed=run, record=True)
        sim.run()
        r = sim.responses
        r["seed"] = run
        rows.append(r)
    keys = [
        "seed", "R1_n_departures", "R2_share_high", "R3_avg_store_count",
        "R4_avg_wait", "R5_avg_loading_ops", "R6_delivered_tons",
        "R7_stored_tons_end", "R8_in_planes_tons", "arrived_tons",
        "n_departures_high", "n_containers_loaded",
    ]
    with open(os.path.join("out", "results_replications.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"Выполнено прогонов: {n}. Результаты: out/results_replications.csv")
    # краткая сводка
    means = {k: sum(r[k] for r in rows) / n for k in
             ["R1_n_departures", "R2_share_high", "R3_avg_store_count",
              "R4_avg_wait", "R5_avg_loading_ops"]}
    for k, v in means.items():
        print(f"{k:24s} mean = {v:.4f}")
    return rows


if __name__ == "__main__":
    main()