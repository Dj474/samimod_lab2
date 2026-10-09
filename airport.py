"""Модель грузового аэропорта (вариант №15) на SimPy.

Ресурс:
  Warehouse        - склад-накопитель, simpy.Store (FIFO-очередь контейнеров)
                     с учётом суммарного веса и интеграла Q(t)dt.
Агенты (процессы SimPy):
  ContainerSource  - CONTAINER_FLOW, генератор потока контейнеров;
  Plane            - PLANE_i, самолёт (ГОТОВ -> ЗАГРУЗКА -> РЕЙС -> ГОТОВ);
  Dispatcher       - DISPATCHER, правило управляющего (приоритет обычных)
                     и процесс горизонта поступлений.
Модель:
  AirportModel     - связывает агентов и ресурс, считает отклики.
"""

import simpy

ST_READY, ST_LOADING, ST_FLYING = 0, 1, 2
ST_NAME = {ST_READY: "ГОТОВ", ST_LOADING: "ЗАГРУЗКА", ST_FLYING: "РЕЙС"}
NORMAL, HIGH = "normal", "high"

RESPONSE_KEYS = ["R1_n_departures", "R2_share_high",
                 "R3_avg_store_count", "R4_avg_wait"]
RESPONSE_LABELS = {
    "R1_n_departures": "R1 число рейсов (адд.)",
    "R2_share_high": "R2 доля рейсов повыш. ГП (адд.)",
    "R3_avg_store_count": "R3 ср. контейнеров на складе (непр.)",
    "R4_avg_wait": "R4 ср. время ожидания, ч (дискр.)",
}


class Container:
    __slots__ = ("weight", "t_arrival")

    def __init__(self, weight, t_arrival):
        self.weight = weight
        self.t_arrival = t_arrival


class Warehouse(simpy.Store):
    """Ресурс: склад. Контейнеры выдаются самолётам в порядке FIFO."""

    def __init__(self, env):
        super().__init__(env)
        self.weight = 0.0
        self.area_count = 0.0
        self.t_last = 0.0

    def __len__(self):
        return len(self.items)

    def accumulate(self, t):
        self.area_count += len(self.items) * (t - self.t_last)
        self.t_last = t

    def _do_put(self, event):
        self.accumulate(self._env.now)
        self.weight += event.item.weight
        return super()._do_put(event)

    def _do_get(self, event):
        if self.items:
            self.accumulate(self._env.now)
            remaining = self.weight - self.items[0].weight
            self.weight = remaining if len(self.items) > 1 else 0.0
        return super()._do_get(event)

    def free_items(self):
        """Контейнеры, ещё не обещанные ожидающим самолётам."""
        return len(self.items) - len(self.get_queue)


class ContainerSource:
    """Агент CONTAINER_FLOW: пуассоновский поток контейнеров до момента T."""

    name = "CONTAINER_FLOW"

    def __init__(self, model):
        self.model = model
        self.n_arrived = 0
        self.arrived_weight = 0.0

    def process(self):
        m = self.model
        run = m.run
        while True:
            delay = run.exp(1.0 / m.cfg["arrival_rate"])
            if run.time + delay > run.run_time:
                return
            yield run.env.timeout(delay)
            container = Container(run.uniform(m.cfg["weight_min"], m.cfg["weight_max"]),
                                  run.time)
            yield m.warehouse.put(container)
            self.n_arrived += 1
            self.arrived_weight += container.weight
            m.record()
            m.trace(self, f"поступил контейнер весом {container.weight:.1f} т")
            m.dispatcher.try_dispatch()


class Plane:
    """Агент PLANE_i: самолёт обычной или повышенной грузоподъёмности."""

    def __init__(self, model, pid, kind, capacity):
        self.model = model
        self.id = pid
        self.kind = kind
        self.capacity = capacity
        self.name = f"PLANE_{pid}"
        self.state = ST_READY
        self.loaded = 0.0
        self.in_hand = 0.0
        self.assigned = None

    def set_state(self, state):
        self.state = state
        self.model.plane_series.append((self.model.run.time, self.id, state))

    def assign(self):
        """Вызывается диспетчером: будит процесс самолёта."""
        self.set_state(ST_LOADING)
        self.model.trace(self, f"назначен ({self.kind}), {self.capacity} т")
        self.assigned.succeed()

    def take_container(self):
        """Ждать контейнер со склада. На дообслуживании, если склад пуст,
        вернуть None (грузить больше нечего)."""
        m = self.model
        if m.closed and m.warehouse.free_items() <= 0:
            return None
        get = m.warehouse.get()
        if m.closed:
            return (yield get)
        yield get | m.horizon
        if get.triggered:
            return get.value
        get.cancel()
        return (yield from self.take_container())

    def load(self):
        m = self.model
        run = m.run
        while self.loaded < self.capacity:
            m.trace(self, f"ждёт контейнер (загружено {self.loaded:.0f}/{self.capacity} т)")
            container = yield from self.take_container()
            if container is None:
                return
            wait = run.time - container.t_arrival
            m.wait_times.append(wait)
            m.wait_sum += wait
            self.in_hand = container.weight
            m.record()
            yield run.env.timeout(run.exp(m.cfg["load_time_mean"]))
            self.in_hand = 0.0
            self.loaded += container.weight
            m.trace(self, f"загружен контейнер {container.weight:.1f} т, "
                          f"уже {self.loaded:.0f}/{self.capacity} т")
            if self.loaded < self.capacity:
                m.dispatcher.try_dispatch()

    def depart(self):
        m = self.model
        cargo = self.loaded
        m.n_departures += 1
        if self.kind == HIGH:
            m.n_departures_high += 1
        m.delivered_weight += cargo
        self.loaded = 0.0
        self.set_state(ST_FLYING)
        m.trace(self, f"ВЫЛЕТ (рейс #{m.n_departures}, {self.kind}, груз {cargo:.0f} т)")
        m.dispatcher.try_dispatch()

    def become_ready(self):
        self.assigned = self.model.run.env.event()
        self.set_state(ST_READY)

    def process(self):
        m = self.model
        run = m.run
        self.assigned = run.env.event()
        while True:
            yield self.assigned
            yield from self.load()
            if self.loaded == 0.0:
                self.become_ready()
                continue
            self.depart()
            yield run.env.timeout(run.exp(m.cfg["flight_time_mean"]))
            self.become_ready()
            m.trace(self, "возврат из рейса, готов к загрузке")
            m.dispatcher.try_dispatch()


class Dispatcher:
    """Агент DISPATCHER: обычные самолёты в приоритете.

    1) есть свободный обычный и на складе >= C_n  -> назначить обычный;
    2) свободных обычных нет, на складе >= C_h    -> назначить повышенной ГП;
    3) иначе ждать.
    На дообслуживании (поступления закрыты) порог — любой незакреплённый груз.
    """

    name = "DISPATCHER"

    def __init__(self, model):
        self.model = model

    def horizon_process(self):
        m = self.model
        yield m.run.env.timeout(m.run.run_time)
        m.closed = True
        m.horizon.succeed()
        m.record()
        m.trace(self, "горизонт: поступления закрыты, дообслуживание")
        self.try_dispatch()

    def has_unclaimed_cargo(self):
        m = self.model
        claimed = sum(p.capacity - p.loaded - p.in_hand
                      for p in m.planes if p.state == ST_LOADING)
        return m.warehouse.free_items() > 0 and m.warehouse.weight > claimed

    def try_dispatch(self):
        m = self.model
        closing = m.closed
        ready_normal = [p for p in m.planes if p.kind == NORMAL and p.state == ST_READY]
        if ready_normal:
            enough = (self.has_unclaimed_cargo() if closing
                      else m.warehouse.weight >= m.cfg["normal_capacity"])
            if enough:
                ready_normal[0].assign()
            return
        ready_high = [p for p in m.planes if p.kind == HIGH and p.state == ST_READY]
        if ready_high:
            enough = (self.has_unclaimed_cargo() if closing
                      else m.warehouse.weight >= m.cfg["high_capacity"])
            if enough:
                ready_high[0].assign()


class AirportModel:
    """Грузовой аэропорт: склад + самолёты + диспетчер."""

    def __init__(self, cfg, record=False):
        self.cfg = cfg
        self.record_history = record
        self.run = None
        self.warehouse = None
        self.horizon = None
        self.closed = False
        self.source = ContainerSource(self)
        self.dispatcher = Dispatcher(self)
        self.planes = []
        for i in range(cfg["n_normal"]):
            self.planes.append(Plane(self, i, NORMAL, cfg["normal_capacity"]))
        for i in range(cfg["n_high"]):
            self.planes.append(Plane(self, cfg["n_normal"] + i, HIGH,
                                     cfg["high_capacity"]))
        self.n_departures = 0
        self.n_departures_high = 0
        self.delivered_weight = 0.0
        self.wait_times = []
        self.wait_sum = 0.0
        self.history = []
        self.plane_series = [(0.0, p.id, ST_READY) for p in self.planes]

    # ------------------------------------------------------ интерфейс ядра
    def start(self, run):
        self.run = run
        env = run.env
        self.warehouse = Warehouse(env)
        self.horizon = env.event()
        if run.trace:
            print("=" * 120)
            print("КВАЗИПАРАЛЛЕЛЬНАЯ ТРАССА (t | агент | действие | склад | состояния самолётов)")
            print("=" * 120)
        self.trace(self.dispatcher, "старт прогона")
        for p in self.planes:
            env.process(p.process())
        env.process(self.source.process())
        env.process(self.dispatcher.horizon_process())

    def finish(self, run):
        self.warehouse.accumulate(run.time)
        for p in self.planes:
            self.plane_series.append((run.time, p.id, p.state))

    def responses(self):
        t_end = self.run.time
        n = self.n_departures
        return {
            "R1_n_departures": n,
            "R2_share_high": self.n_departures_high / n if n else 0.0,
            "R3_avg_store_count": self.warehouse.area_count / max(t_end, 1e-12),
            "R4_avg_wait": (sum(self.wait_times) / len(self.wait_times)
                            if self.wait_times else 0.0),
            "arrived_tons": self.source.arrived_weight,
            "delivered_tons": self.delivered_weight,
            "stored_tons_end": self.warehouse.weight,
            "in_planes_tons": sum(p.loaded + p.in_hand for p in self.planes),
            "n_departures_high": self.n_departures_high,
            "n_containers_loaded": len(self.wait_times),
            "t_end": t_end,
            "drain_time": max(0.0, t_end - self.run.run_time),
        }

    # ---------------------------------------------------- общая логика
    def record(self):
        """История для графиков: (t, контейнеров на складе, вес склада,
        вылетов, интеграл Q(t)dt, сумма ожиданий, взято контейнеров)."""
        if self.record_history:
            t = self.run.time
            wh = self.warehouse
            area = wh.area_count + len(wh) * (t - wh.t_last)
            self.history.append((t, len(wh), wh.weight, self.n_departures,
                                 area, self.wait_sum, len(self.wait_times)))

    def trace(self, agent, msg):
        if not self.run.trace:
            return
        states = " ".join(f"{'N' if p.kind == NORMAL else 'H'}{p.id}:{ST_NAME[p.state]}"
                          for p in self.planes)
        print(f"t={self.run.time:9.3f} | {agent.name:16s} | {msg:55s} | "
              f"склад[шт]={len(self.warehouse):4d} вес={self.warehouse.weight:8.1f} т | "
              f"{states}")

    def balance(self):
        """Проверка: прибыло = вывезено + остаток на складе + груз в самолётах."""
        r = self.responses()
        lhs = r["arrived_tons"]
        rhs = r["delivered_tons"] + r["stored_tons_end"] + r["in_planes_tons"]
        return lhs, rhs, abs(lhs - rhs)
