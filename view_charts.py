"""Просмотр диаграмм лабораторной в окне с вкладками.

  python view_charts.py
  python view_charts.py --dir out

PNG из папки out/ показываются на экране (не только как файлы).
"""

from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

OUT = "out"

# (файл, короткая вкладка, подсказка внизу)
TABS = [
    ("fig1_hist_R1.png", "1. R1 рейсы",
     "Распределение числа вылетевших рейсов по 100 прогонам. Среднее ≈ 3295."),
    ("fig2_hist_R2.png", "2. R2 доля ГП",
     "Доля рейсов повышенной грузоподъёмности. Среднее ≈ 24%."),
    ("fig3_series.png", "3. Склад + вылеты",
     "Один прогон: вес на складе и накопленное число вылетов."),
    ("fig4_hist_R4.png", "4. R4 ожидание",
     "Распределение среднего времени ожидания контейнера. Среднее ≈ 7.2 ч."),
    ("fig5_boxplots.png", "5. Ящики",
     "Разброс откликов R1–R4 по повторным прогонам."),
    ("fig6_conv.png", "6. Сходимость",
     "Скользящее среднее / итог: после ~80 прогонов оценки стабильны."),
    ("fig7_CI.png", "7. 95% ДИ",
     "Средние отклики со 95%-ми доверительными интервалами."),
    ("fig8_gantt.png", "8. Gantt",
     "Состояния самолётов в первый месяц: ГОТОВ / ЗАГРУЗКА / РЕЙС."),
    ("fig9_dynamics_base.png", "9. База λ=3",
     "Стационар: после переходного периода склад колеблется около средней."),
    ("fig10_dynamics_overload.png", "10. Перегрузка λ=4",
     "Рост склада по прямой до T (красная линия), затем спад на дообслуживании."),
    ("fig11_sens_arrival_rate.png", "11. Чувств. λ",
     "Чувствительность к интенсивности потока. Порог перегрузки ≈ 3.4."),
    ("fig12_sens_flight_time.png", "12. Чувств. рейс",
     "Чувствительность к среднему времени рейса."),
    ("fig13_sens_n_normal.png", "13. Чувств. Nобыч",
     "Чувствительность к числу обычных самолётов."),
]


def load_photo(path: str, max_w: int, max_h: int) -> tk.PhotoImage:
    """Загрузить PNG с подгонкой под размер окна."""
    try:
        from PIL import Image, ImageTk
        img = Image.open(path)
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except ImportError:
        photo = tk.PhotoImage(file=path)
        # грубое уменьшение без Pillow
        w, h = photo.width(), photo.height()
        fx = max(1, (w + max_w - 1) // max_w)
        fy = max(1, (h + max_h - 1) // max_h)
        f = max(fx, fy)
        if f > 1:
            photo = photo.subsample(f, f)
        return photo


class ChartViewer(tk.Tk):
    def __init__(self, out_dir: str):
        super().__init__()
        self.title("Лаба 2 — диаграммы имитации (вкладки)")
        self.geometry("1280x820")
        self.minsize(900, 600)
        self.out_dir = out_dir
        self._photos: list[tk.PhotoImage] = []

        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TNotebook.Tab", padding=(10, 4), font=("Segoe UI", 10))

        top = ttk.Frame(self, padding=6)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text=f"Папка: {os.path.abspath(out_dir)}  |  "
                 f"← → или Ctrl+Tab — соседняя вкладка",
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT)

        self.hint = ttk.Label(self, text="", font=("Segoe UI", 10),
                              foreground="#333", padding=(8, 4))
        self.hint.pack(side=tk.BOTTOM, fill=tk.X)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        max_w, max_h = max(900, sw - 80), max(600, sh - 180)

        missing = []
        for fname, title, tip in TABS:
            path = os.path.join(out_dir, fname)
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            frame._tip = tip  # type: ignore[attr-defined]

            if not os.path.isfile(path):
                missing.append(fname)
                ttk.Label(
                    frame,
                    text=f"Файл не найден:\n{path}\n\n"
                         f"Сначала запустите:\n  python charts.py\n  "
                         f"python sensitivity.py",
                    font=("Segoe UI", 12),
                    justify=tk.CENTER,
                ).pack(expand=True)
                continue

            canvas = tk.Canvas(frame, background="#f4f4f4", highlightthickness=0)
            vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
            hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
            canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            hsb.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            photo = load_photo(path, max_w, max_h)
            self._photos.append(photo)  # не дать GC удалить
            inner = ttk.Frame(canvas)
            ttk.Label(inner, image=photo).pack(padx=8, pady=8)
            win = canvas.create_window((0, 0), window=inner, anchor="nw")

            def _sync(event=None, c=canvas, i=inner, w=win):
                c.configure(scrollregion=c.bbox("all"))
                c.itemconfigure(w, width=max(i.winfo_reqwidth(), c.winfo_width()))

            inner.bind("<Configure>", _sync)
            canvas.bind("<Configure>", _sync)
            canvas.bind("<MouseWheel>",
                        lambda e, c=canvas: c.yview_scroll(-int(e.delta / 120), "units"))

        self.bind("<Left>", lambda e: self._step(-1))
        self.bind("<Right>", lambda e: self._step(1))
        self.bind("<Control-Tab>", lambda e: self._step(1))
        self.bind("<Control-Shift-Tab>", lambda e: self._step(-1))
        self._on_tab()

        if missing:
            print("Нет файлов:", ", ".join(missing), file=sys.stderr)

    def _on_tab(self, _event=None):
        idx = self.nb.index(self.nb.select())
        frame = self.nb.nametowidget(self.nb.select())
        tip = getattr(frame, "_tip", "")
        self.hint.configure(text=tip)

    def _step(self, delta: int):
        n = self.nb.index("end")
        if n <= 0:
            return
        i = (self.nb.index(self.nb.select()) + delta) % n
        self.nb.select(i)


def main():
    parser = argparse.ArgumentParser(description="Просмотр графиков лабы во вкладках")
    parser.add_argument("--dir", default=OUT, help="папка с fig*.png (по умолчанию out)")
    args = parser.parse_args()
    if not os.path.isdir(args.dir):
        print(f"Папка не найдена: {args.dir}", file=sys.stderr)
        print("Сначала: python charts.py && python sensitivity.py", file=sys.stderr)
        sys.exit(1)
    app = ChartViewer(args.dir)
    app.mainloop()


if __name__ == "__main__":
    main()
