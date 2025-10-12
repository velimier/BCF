from __future__ import annotations
import json, os, random, statistics, threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Tuple

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QHBoxLayout, QComboBox, QGroupBox
)
from PySide6.QtCore import Signal, QObject, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter

from ..engine.models import RockMass, JointSet, SpacingDist, CaveFace, Defaults, SecondaryRun, PrimaryBlock
from ..engine.primary import generate_primary_blocks
from ..engine.secondary import run_secondary, average_scatter_deg_from_jointsets
from ..engine.hangup import orepass_hangups, kear_hangups
from ..engine.io_formats import distributions_from_blocks, write_prm, write_sec


def randomize_value(
    value: float,
    variation_pct: float,
    minimum: float | None = None,
    maximum: float | None = None,
    rng: random.Random | None = None,
) -> float:
    rng = rng or random
    if variation_pct <= 0:
        new_val = value
    else:
        delta = variation_pct / 100.0
        new_val = value * (1.0 + rng.uniform(-delta, delta))
    if minimum is not None:
        new_val = max(minimum, new_val)
    if maximum is not None:
        new_val = min(maximum, new_val)
    return new_val


def randomize_joint_set(js: JointSet, variation_pct: float, rng: random.Random | None = None) -> JointSet:
    rng = rng or random
    spacing_vals = [
        randomize_value(js.spacing.min, variation_pct, 0.01, rng=rng),
        randomize_value(js.spacing.mean, variation_pct, 0.02, rng=rng),
        randomize_value(js.spacing.max_or_90pct, variation_pct, 0.05, rng=rng),
    ]
    spacing_vals.sort()
    spacing = SpacingDist(
        js.spacing.type,
        spacing_vals[0],
        max(spacing_vals[0], spacing_vals[1]),
        max(spacing_vals[1], spacing_vals[2])
    )
    return JointSet(
        name=js.name,
        mean_dip=randomize_value(js.mean_dip, variation_pct, 0.0, 90.0, rng=rng),
        dip_range=randomize_value(js.dip_range, variation_pct, 0.0, 90.0, rng=rng),
        mean_dip_dir=randomize_value(js.mean_dip_dir, variation_pct, 0.0, 360.0, rng=rng),
        dip_dir_range=randomize_value(js.dip_dir_range, variation_pct, 0.0, 180.0, rng=rng),
        spacing=spacing,
        JC=int(round(randomize_value(js.JC, variation_pct, 0, 40, rng=rng)))
    )


MC_COLOR_OPTIONS: List[Tuple[str, str]] = [
    ("Blue", "#1f77b4"),
    ("Orange", "#ff7f0e"),
    ("Green", "#2ca02c"),
    ("Red", "#d62728"),
    ("Purple", "#9467bd"),
    ("Brown", "#8c564b"),
    ("Pink", "#e377c2"),
    ("Gray", "#7f7f7f"),
    ("Olive", "#bcbd22"),
    ("Cyan", "#17becf"),
    ("Black", "#000000"),
]

LINE_STYLE_OPTIONS: List[Tuple[str, str]] = [
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dotted", ":"),
    ("Dash-dot", "-.")
]


def _monte_carlo_worker(
    seed: int,
    nblocks: int,
    variation: float,
    base_inputs: Dict[str, object],
) -> Tuple[dict, dict]:
    rng = random.Random(seed)

    rock_dict = base_inputs.get("rock", {})
    rock = RockMass(
        rock_type=rock_dict.get("rock_type", "Unknown"),
        MRMR=randomize_value(rock_dict.get("MRMR", 65.0), variation, 0.0, 100.0, rng=rng),
        IRS=randomize_value(rock_dict.get("IRS", 120.0), variation, 1.0, 500.0, rng=rng),
        IBS=rock_dict.get("IBS"),
        mi=randomize_value(rock_dict.get("mi", 17.0), variation, 1.0, 50.0, rng=rng),
        frac_freq=randomize_value(rock_dict.get("frac_freq", 0.0), variation, 0.0, 20.0, rng=rng),
        frac_condition=int(round(randomize_value(rock_dict.get("frac_condition", 20), variation, 0, 40, rng=rng))),
        density=randomize_value(rock_dict.get("density", 3200.0), variation, 1500.0, 4500.0, rng=rng),
    )

    joint_sets = []
    for js_dict in base_inputs.get("joint_sets", []):
        spacing_dict = js_dict.get("spacing", {})
        spacing = SpacingDist(
            spacing_dict.get("type", "trunc_exp"),
            spacing_dict.get("min", 0.3),
            spacing_dict.get("mean", 1.0),
            spacing_dict.get("max_or_90pct", 3.0),
            spacing_dict.get("max_obs"),
        )
        js = JointSet(
            name=js_dict.get("name", "Set"),
            mean_dip=js_dict.get("mean_dip", 45.0),
            dip_range=js_dict.get("dip_range", 10.0),
            mean_dip_dir=js_dict.get("mean_dip_dir", 0.0),
            dip_dir_range=js_dict.get("dip_dir_range", 20.0),
            spacing=spacing,
            JC=js_dict.get("JC", 20),
        )
        joint_sets.append(randomize_joint_set(js, variation, rng=rng))

    cave_dict = base_inputs.get("cave", {})
    cave = CaveFace(
        dip=cave_dict.get("dip", 45.0),
        dip_dir=cave_dict.get("dip_dir", 0.0),
        stress_dip=cave_dict.get("stress_dip", 5.0),
        stress_strike=cave_dict.get("stress_strike", 5.0),
        stress_normal=cave_dict.get("stress_normal", 0.0),
        allow_stress_fractures=cave_dict.get("allow_stress_fractures", True),
        spalling_pct=randomize_value(cave_dict.get("spalling_pct", 0.0), variation, 0.0, 100.0, rng=rng),
    )

    defaults_dict = base_inputs.get("defaults", {})
    defaults = Defaults(
        LHD_cutoff_m3=randomize_value(defaults_dict.get("LHD_cutoff_m3", 2.0), variation, 0.1, 50.0, rng=rng),
        seed=rng.randint(0, 10**9),
        arching_pct=randomize_value(defaults_dict.get("arching_pct", 0.12), variation, 0.0, 1.0, rng=rng),
        arch_stress_conc=randomize_value(defaults_dict.get("arch_stress_conc", 25.0), variation, 1.0, 100.0, rng=rng),
        stress_frac_trace_min=defaults_dict.get("stress_frac_trace_min", 10.0),
        stress_frac_trace_mean=defaults_dict.get("stress_frac_trace_mean", 20.0),
        stress_frac_trace_max=defaults_dict.get("stress_frac_trace_max", 30.0),
        tension_factor=defaults_dict.get("tension_factor", 0.0),
    )

    secondary_dict = base_inputs.get("secondary", {})
    secondary_params = SecondaryRun(
        draw_height=randomize_value(secondary_dict.get("draw_height", 150.0), variation, 1.0, 2000.0, rng=rng),
        max_caving_height=randomize_value(secondary_dict.get("max_caving_height", 300.0), variation, 1.0, 5000.0, rng=rng),
        swell_factor=randomize_value(secondary_dict.get("swell_factor", 1.2), variation, 1.0, 3.0, rng=rng),
        active_draw_width=randomize_value(secondary_dict.get("active_draw_width", 45.0), variation, 1.0, 200.0, rng=rng),
        added_fines_pct=randomize_value(secondary_dict.get("added_fines_pct", 0.0), variation, 0.0, 80.0, rng=rng),
        rate_cm_day=randomize_value(secondary_dict.get("rate_cm_day", 20.0), variation, 0.0, 100.0, rng=rng),
        drawbell_upper_width=randomize_value(secondary_dict.get("drawbell_upper_width", 8.0), variation, 1.0, 50.0, rng=rng),
        drawbell_lower_width=randomize_value(secondary_dict.get("drawbell_lower_width", 6.0), variation, 1.0, 50.0, rng=rng),
    )

    blocks = generate_primary_blocks(nblocks, rock, joint_sets, cave, defaults, seed=defaults.seed)
    primary_stats = distributions_from_blocks(blocks)
    mu = average_scatter_deg_from_jointsets(joint_sets)
    primary_fines_ratio = cave.spalling_pct / 100.0
    sec_blocks, _ = run_secondary(
        blocks,
        rock,
        secondary_params,
        defaults,
        mu_scatter_deg=mu,
        primary_fines_ratio=primary_fines_ratio,
    )
    secondary_stats = distributions_from_blocks(sec_blocks)
    return primary_stats, secondary_stats


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        lay = QVBoxLayout()
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        self.setLayout(lay)
        self.canvas.setMinimumHeight(320)
        self.has_data = False

    def plot_distributions(self, prim_stats: dict, sec_stats: dict | None = None, title: str = ""):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        xs = [lo for lo,hi in prim_stats["bins"]]
        ax.plot(xs, prim_stats["cum_mass"], label="Primary")
        if sec_stats is not None:
            ax.plot(xs, sec_stats["cum_mass"], label="Secondary")
        ax.set_xscale("log")
        ax.set_xlabel("Block volume (m³)")
        ax.set_ylabel("Cumulative mass (%)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
        self.canvas.draw_idle()
        self.has_data = True

    def plot_lines(
        self,
        xs: List[float] | List[List[float]],
        ys_list: List[List[float]],
        labels: List[str] | None = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        logx: bool = False,
        styles: List[Dict[str, object]] | None = None,
        x_formatter=None,
        y_formatter=None,
    ):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if ys_list:
            if isinstance(xs, list) and xs and isinstance(xs[0], (list, tuple)):
                xs_list = list(xs)
            else:
                xs_list = [xs] * len(ys_list)
        else:
            xs_list = []
        for i, ys in enumerate(ys_list):
            label = labels[i] if labels and i < len(labels) else None
            cur_xs = xs_list[i] if i < len(xs_list) else xs_list[0] if xs_list else []
            style = styles[i] if styles and i < len(styles) else {}
            display_label = style.get("label") if isinstance(style, dict) and style.get("label") else label
            color = style.get("color") if isinstance(style, dict) else None
            linestyle = style.get("linestyle") if isinstance(style, dict) and style.get("linestyle") else "-"
            linewidth = style.get("linewidth") if isinstance(style, dict) and style.get("linewidth") else 1.5
            ax.plot(cur_xs, ys, label=display_label, color=color, linestyle=linestyle, linewidth=linewidth)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":")
        if labels or (styles and any(isinstance(style, dict) and style.get("label") for style in (styles or []))):
            ax.legend()
        if x_formatter is not None:
            ax.xaxis.set_major_formatter(x_formatter)
        if y_formatter is not None:
            ax.yaxis.set_major_formatter(y_formatter)
        self.canvas.draw_idle()
        self.has_data = True

    def save_dialog(self, parent: QWidget, suggested_name: str = "chart.png"):
        if not self.has_data:
            QMessageBox.warning(parent, "No chart", "There is no chart to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(parent, "Save chart", suggested_name,
                                             "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)")
        if path:
            try:
                self.fig.savefig(path, dpi=300, bbox_inches="tight")
            except Exception as exc:  # pragma: no cover - UI feedback
                QMessageBox.critical(parent, "Save failed", f"Could not save chart:\n{exc}")
            else:
                QMessageBox.information(parent, "Chart saved", f"Chart saved to:\n{path}")

class AppSignals(QObject):
    log = Signal(str)
    done_primary = Signal(list, float, str)
    done_secondary = Signal(list, float, str)
    done_hangup = Signal(dict)
    done_monte_carlo = Signal(dict)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCF-Style Fragmentation (Python, PySide6)")
        self.resize(1100, 760)
        self.sig = AppSignals()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.rock = RockMass()
        self.joint_sets: List[JointSet] = [
            JointSet("Set1", 60, 20, 90, 20, SpacingDist("trunc_exp",0.3,1.0,3.0), 20),
            JointSet("Set2", 45, 20, 0,  20, SpacingDist("trunc_exp",0.3,1.5,4.0), 20),
            JointSet("Set3", 30, 20, 180,20, SpacingDist("trunc_exp",0.4,2.0,5.0), 20),
        ]
        self.cave = CaveFace()
        self.defaults = Defaults()
        self.secondary = SecondaryRun()

        self.primary_blocks: List[PrimaryBlock] = []
        self.primary_fines_ratio = 0.0
        self.secondary_blocks = []
        self._last_monte_carlo_result: dict | None = None
        self._last_monte_carlo_inputs: dict | None = None
        self._last_monte_carlo_settings: dict | None = None

        self._build_tabs()
        self._connect_signals()

    def _set_uniform_input_width(self, widget):
        widget.setMinimumWidth(140)
        widget.setMaximumWidth(220)

    def _build_tabs(self):
        self.tabs.addTab(self._build_geology_tab(), "Geology")
        self.tabs.addTab(self._build_cave_tab(), "Cave")
        self.tabs.addTab(self._build_primary_tab(), "Primary run")
        self.tabs.addTab(self._build_secondary_tab(), "Secondary & hang-up")
        self.tabs.addTab(self._build_monte_carlo_tab(), "Monte Carlo")
        self.tabs.addTab(self._build_defaults_tab(), "Defaults")

    def _build_geology_tab(self):
        w = QWidget(); g = QGridLayout(w)
        row = 0; g.addWidget(QLabel("<b>Rock mass</b>"), row,0,1,2); row+=1
        self.rock_type = QLineEdit(self.rock.rock_type); self._set_uniform_input_width(self.rock_type); g.addWidget(QLabel("Rock type"), row,0); g.addWidget(self.rock_type,row,1); row+=1
        self.mrmr = QDoubleSpinBox(); self.mrmr.setRange(0,100); self.mrmr.setValue(self.rock.MRMR); self._set_uniform_input_width(self.mrmr); g.addWidget(QLabel("MRMR"), row,0); g.addWidget(self.mrmr,row,1); row+=1
        self.irs = QDoubleSpinBox(); self.irs.setRange(1,500); self.irs.setValue(self.rock.IRS); self._set_uniform_input_width(self.irs); g.addWidget(QLabel("IRS (MPa)"), row,0); g.addWidget(self.irs,row,1); row+=1
        self.mi = QDoubleSpinBox(); self.mi.setRange(1,50); self.mi.setValue(self.rock.mi); self._set_uniform_input_width(self.mi); g.addWidget(QLabel("mi (Hoek–Brown)"), row,0); g.addWidget(self.mi,row,1); row+=1
        self.ff = QDoubleSpinBox(); self.ff.setRange(0,20); self.ff.setDecimals(2); self.ff.setValue(self.rock.frac_freq); self._set_uniform_input_width(self.ff); g.addWidget(QLabel("Fracture/veinlet freq (1/m)"), row,0); g.addWidget(self.ff,row,1); row+=1
        self.fc = QSpinBox(); self.fc.setRange(0,40); self.fc.setValue(self.rock.frac_condition); self._set_uniform_input_width(self.fc); g.addWidget(QLabel("Fracture/veinlet condition (0–40)"), row,0); g.addWidget(self.fc,row,1); row+=1
        self.density = QDoubleSpinBox(); self.density.setRange(1500,4500); self.density.setValue(self.rock.density); self._set_uniform_input_width(self.density); g.addWidget(QLabel("Density (kg/m³)"), row,0); g.addWidget(self.density,row,1); row+=1

        row+=1; g.addWidget(QLabel("<b>Joint sets</b>"), row,0,1,2); row+=1
        self.joint_widgets = []
        for i,js in enumerate(self.joint_sets):
            g.addWidget(QLabel(f"<u>{js.name}</u>"), row,0,1,2); row+=1
            dip = QDoubleSpinBox(); dip.setRange(0,90); dip.setValue(js.mean_dip); self._set_uniform_input_width(dip)
            dipr= QDoubleSpinBox(); dipr.setRange(0,90); dipr.setValue(js.dip_range); self._set_uniform_input_width(dipr)
            dd  = QDoubleSpinBox(); dd.setRange(0,360); dd.setValue(js.mean_dip_dir); self._set_uniform_input_width(dd)
            ddr = QDoubleSpinBox(); ddr.setRange(0,180); ddr.setValue(js.dip_dir_range); self._set_uniform_input_width(ddr)
            jc  = QSpinBox(); jc.setRange(0,40); jc.setValue(js.JC); self._set_uniform_input_width(jc)
            s_min=QDoubleSpinBox(); s_min.setRange(0.01,50); s_min.setDecimals(2); s_min.setValue(js.spacing.min); self._set_uniform_input_width(s_min)
            s_mean=QDoubleSpinBox(); s_mean.setRange(0.02,50); s_mean.setDecimals(2); s_mean.setValue(js.spacing.mean); self._set_uniform_input_width(s_mean)
            s_max=QDoubleSpinBox(); s_max.setRange(0.03,200); s_max.setDecimals(2); s_max.setValue(js.spacing.max_or_90pct); self._set_uniform_input_width(s_max)

            g.addWidget(QLabel("Dip / Range"), row,0); g.addWidget(dip,row,1); g.addWidget(dipr,row,2); row+=1
            g.addWidget(QLabel("Dip dir / Range"), row,0); g.addWidget(dd,row,1); g.addWidget(ddr,row,2); row+=1
            g.addWidget(QLabel("JC (0–40)"), row,0); g.addWidget(jc,row,1); row+=1
            g.addWidget(QLabel("Spacing min / mean / max"), row,0); g.addWidget(s_min,row,1); g.addWidget(s_mean,row,2); g.addWidget(s_max,row,3); row+=1

            self.joint_widgets.append((dip,dipr,dd,ddr,jc,s_min,s_mean,s_max))

        w.setLayout(g)
        return w

    def _build_cave_tab(self):
        w = QWidget(); g = QGridLayout(w)
        row=0; g.addWidget(QLabel("<b>Cave face & stresses</b>"), row,0,1,2); row+=1
        self.cave_dip = QDoubleSpinBox(); self.cave_dip.setRange(0,90); self.cave_dip.setValue(self.cave.dip); self._set_uniform_input_width(self.cave_dip); g.addWidget(QLabel("Face dip (°)"), row,0); g.addWidget(self.cave_dip,row,1); row+=1
        self.cave_ddir = QDoubleSpinBox(); self.cave_ddir.setRange(0,360); self.cave_ddir.setValue(self.cave.dip_dir); self._set_uniform_input_width(self.cave_ddir); g.addWidget(QLabel("Face dip direction (°)"), row,0); g.addWidget(self.cave_ddir,row,1); row+=1
        self.st_dip = QDoubleSpinBox(); self.st_dip.setRange(0,100); self.st_dip.setValue(self.cave.stress_dip); self._set_uniform_input_width(self.st_dip); g.addWidget(QLabel("Dip stress (MPa)"), row,0); g.addWidget(self.st_dip,row,1); row+=1
        self.st_strike = QDoubleSpinBox(); self.st_strike.setRange(0,100); self.st_strike.setValue(self.cave.stress_strike); self._set_uniform_input_width(self.st_strike); g.addWidget(QLabel("Strike stress (MPa)"), row,0); g.addWidget(self.st_strike,row,1); row+=1
        self.st_norm = QDoubleSpinBox(); self.st_norm.setRange(0,100); self.st_norm.setValue(self.cave.stress_normal); self._set_uniform_input_width(self.st_norm); g.addWidget(QLabel("Normal stress (MPa)"), row,0); g.addWidget(self.st_norm,row,1); row+=1
        self.allow_sf = QCheckBox("Allow stress fractures"); self.allow_sf.setChecked(self.cave.allow_stress_fractures); g.addWidget(self.allow_sf,row,0,1,2); row+=1
        self.spalling = QDoubleSpinBox(); self.spalling.setRange(0,100); self.spalling.setDecimals(1); self.spalling.setValue(self.cave.spalling_pct); self._set_uniform_input_width(self.spalling); g.addWidget(QLabel("% spalling as fines"), row,0); g.addWidget(self.spalling,row,1); row+=1
        return w

    def _build_primary_tab(self):
        w = QWidget()
        layout = QHBoxLayout(w)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("<b>Primary run</b>"))
        self.nblocks = QSpinBox(); self.nblocks.setRange(100,200000); self.nblocks.setValue(20000)
        controls.addWidget(QLabel("Blocks to generate"))
        controls.addWidget(self.nblocks)
        self.btn_run_primary = QPushButton("Run primary")
        self.btn_save_prm = QPushButton("Save .PRM…"); self.btn_save_prm.setEnabled(False)
        hl = QHBoxLayout(); hl.addWidget(self.btn_run_primary); hl.addWidget(self.btn_save_prm)
        controls.addLayout(hl)
        controls.addStretch(1)

        layout.addLayout(controls, 0)

        plot_layout = QVBoxLayout()
        self.plot_primary = PlotWidget()
        plot_layout.addWidget(self.plot_primary)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_save_primary_plot = QPushButton("Save chart…")
        self.btn_save_primary_plot.setEnabled(False)
        btn_row.addWidget(self.btn_save_primary_plot)
        plot_layout.addLayout(btn_row)
        layout.addLayout(plot_layout, 1)

        return w

    def _build_secondary_tab(self):
        w = QWidget()
        layout = QHBoxLayout(w)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("<b>Secondary run & Hang-up</b>"))
        self.draw_height = QDoubleSpinBox(); self.draw_height.setRange(1,2000); self.draw_height.setValue(self.secondary.draw_height)
        controls.addWidget(QLabel("Draw height (m)")); controls.addWidget(self.draw_height)
        self.max_caving = QDoubleSpinBox(); self.max_caving.setRange(1,5000); self.max_caving.setValue(self.secondary.max_caving_height)
        controls.addWidget(QLabel("Max caving height (m)")); controls.addWidget(self.max_caving)
        self.swell = QDoubleSpinBox(); self.swell.setRange(1.0,3.0); self.swell.setSingleStep(0.05); self.swell.setValue(self.secondary.swell_factor)
        controls.addWidget(QLabel("Swell factor")); controls.addWidget(self.swell)
        self.draw_width = QDoubleSpinBox(); self.draw_width.setRange(1,200); self.draw_width.setValue(self.secondary.active_draw_width)
        controls.addWidget(QLabel("Active draw width (m)")); controls.addWidget(self.draw_width)
        self.add_fines = QDoubleSpinBox(); self.add_fines.setRange(0,80); self.add_fines.setValue(self.secondary.added_fines_pct)
        controls.addWidget(QLabel("Added fines % (dilution)")); controls.addWidget(self.add_fines)
        self.rate = QDoubleSpinBox(); self.rate.setRange(0,100); self.rate.setValue(self.secondary.rate_cm_day)
        controls.addWidget(QLabel("Draw rate (cm/day)")); controls.addWidget(self.rate)
        self.upper_w = QDoubleSpinBox(); self.upper_w.setRange(1,50); self.upper_w.setValue(self.secondary.drawbell_upper_width)
        controls.addWidget(QLabel("Drawbell upper width (m)")); controls.addWidget(self.upper_w)
        self.lower_w = QDoubleSpinBox(); self.lower_w.setRange(1,50); self.lower_w.setValue(self.secondary.drawbell_lower_width)
        controls.addWidget(QLabel("Drawbell lower width (m)")); controls.addWidget(self.lower_w)
        self.hang_method = QComboBox(); self.hang_method.addItems(["Ore-pass rules (width)","Kear method (area)"])
        controls.addWidget(QLabel("Hang-up method")); controls.addWidget(self.hang_method)
        self.btn_run_secondary = QPushButton("Run secondary + hang-up")
        self.btn_save_sec = QPushButton("Save .SEC…"); self.btn_save_sec.setEnabled(False)
        hl = QHBoxLayout(); hl.addWidget(self.btn_run_secondary); hl.addWidget(self.btn_save_sec)
        controls.addLayout(hl)
        controls.addStretch(1)

        layout.addLayout(controls, 0)

        plot_layout = QVBoxLayout()
        self.plot_secondary = PlotWidget()
        plot_layout.addWidget(self.plot_secondary)
        btn_row = QHBoxLayout(); btn_row.addStretch(1)
        self.btn_save_secondary_plot = QPushButton("Save chart…")
        self.btn_save_secondary_plot.setEnabled(False)
        btn_row.addWidget(self.btn_save_secondary_plot)
        plot_layout.addLayout(btn_row)
        self.lbl_hang = QLabel("Hang-up: -")
        self.lbl_hang.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        plot_layout.addWidget(self.lbl_hang)
        layout.addLayout(plot_layout, 1)

        return w

    def _build_monte_carlo_tab(self):
        w = QWidget()
        layout = QHBoxLayout(w)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("<b>Monte Carlo simulation</b>"))
        self.mc_runs = QSpinBox(); self.mc_runs.setRange(1, 1000); self.mc_runs.setValue(10)
        controls.addWidget(QLabel("Number of runs")); controls.addWidget(self.mc_runs)
        self.mc_blocks = QSpinBox(); self.mc_blocks.setRange(100, 50000); self.mc_blocks.setValue(5000)
        controls.addWidget(QLabel("Blocks per run")); controls.addWidget(self.mc_blocks)
        self.mc_variation = QDoubleSpinBox(); self.mc_variation.setRange(0.0, 100.0); self.mc_variation.setSingleStep(1.0); self.mc_variation.setValue(15.0)
        self.mc_variation.setSuffix(" %")
        controls.addWidget(QLabel("Parameter variation (±%)")); controls.addWidget(self.mc_variation)
        self.btn_run_monte_carlo = QPushButton("Run Monte Carlo")
        controls.addWidget(self.btn_run_monte_carlo)
        controls.addStretch(1)

        layout.addLayout(controls, 0)

        plot_layout = QVBoxLayout()
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(QLabel("Series visibility:"))
        self.chk_show_primary = QCheckBox("Primary")
        self.chk_show_primary.setChecked(True)
        self.chk_show_secondary = QCheckBox("Secondary")
        self.chk_show_secondary.setChecked(True)
        toggle_row.addWidget(self.chk_show_primary)
        toggle_row.addWidget(self.chk_show_secondary)
        toggle_row.addStretch(1)
        plot_layout.addLayout(toggle_row)

        style_group = QGroupBox("Chart styling")
        style_layout = QVBoxLayout(style_group)
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title"))
        self.mc_title_edit = QLineEdit("Monte Carlo cumulative mass envelopes")
        title_row.addWidget(self.mc_title_edit)
        style_layout.addLayout(title_row)

        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("X-axis format"))
        self.mc_xformat_combo = QComboBox()
        self.mc_xformat_combo.addItems(["Auto", "0", "0.0", "0.00", "Scientific (1eX)"])
        axis_row.addWidget(self.mc_xformat_combo)
        axis_row.addSpacing(12)
        axis_row.addWidget(QLabel("Y-axis format"))
        self.mc_yformat_combo = QComboBox()
        self.mc_yformat_combo.addItems(["Auto", "0", "0.0", "0.00", "Scientific (1eX)"])
        axis_row.addWidget(self.mc_yformat_combo)
        axis_row.addStretch(1)
        style_layout.addLayout(axis_row)

        style_layout.addWidget(QLabel("Series styling"))
        self.mc_series_container = QVBoxLayout()
        style_layout.addLayout(self.mc_series_container)
        style_layout.addStretch(1)
        plot_layout.addWidget(style_group)

        self.plot_monte_carlo = PlotWidget()
        plot_layout.addWidget(self.plot_monte_carlo)
        btn_row = QHBoxLayout(); btn_row.addStretch(1)
        self.btn_save_mc_report = QPushButton("Save report…")
        self.btn_save_mc_report.setEnabled(False)
        btn_row.addWidget(self.btn_save_mc_report)
        self.btn_save_mc_plot = QPushButton("Save chart…")
        self.btn_save_mc_plot.setEnabled(False)
        btn_row.addWidget(self.btn_save_mc_plot)
        plot_layout.addLayout(btn_row)
        self.lbl_mc_stats = QLabel("Monte Carlo: -")
        plot_layout.addWidget(self.lbl_mc_stats)
        layout.addLayout(plot_layout, 1)

        self._mc_style_widgets: Dict[str, Dict[str, QWidget]] = {}
        self._mc_color_cursor = 0

        return w

    def _build_defaults_tab(self):
        w = QWidget(); g = QGridLayout(w)
        row=0; g.addWidget(QLabel("<b>Defaults</b>"), row,0,1,2); row+=1
        self.lhd_cutoff = QDoubleSpinBox(); self.lhd_cutoff.setRange(0.1,50); self.lhd_cutoff.setValue(self.defaults.LHD_cutoff_m3); self._set_uniform_input_width(self.lhd_cutoff); g.addWidget(QLabel("LHD bucket cutoff (m³)"), row,0); g.addWidget(self.lhd_cutoff,row,1); row+=1
        self.seed = QSpinBox(); self.seed.setRange(0,10**9); self.seed.setValue(self.defaults.seed or 1234); self._set_uniform_input_width(self.seed); g.addWidget(QLabel("Random seed"), row,0); g.addWidget(self.seed,row,1); row+=1
        self.arching_pct = QDoubleSpinBox(); self.arching_pct.setRange(0,1); self.arching_pct.setSingleStep(0.01); self.arching_pct.setValue(self.defaults.arching_pct); self._set_uniform_input_width(self.arching_pct); g.addWidget(QLabel("Arching split fraction (0–1)"), row,0); g.addWidget(self.arching_pct,row,1); row+=1
        self.arch_factor = QDoubleSpinBox(); self.arch_factor.setRange(1,100); self.arch_factor.setValue(self.defaults.arch_stress_conc); self._set_uniform_input_width(self.arch_factor); g.addWidget(QLabel("Arch stress factor (× cave pressure)"), row,0); g.addWidget(self.arch_factor,row,1); row+=1
        return w

    def _connect_signals(self):
        self.btn_run_primary.clicked.connect(self.on_run_primary)
        self.btn_save_prm.clicked.connect(self.on_save_prm)
        self.btn_run_secondary.clicked.connect(self.on_run_secondary)
        self.btn_save_sec.clicked.connect(self.on_save_sec)
        self.btn_save_primary_plot.clicked.connect(lambda: self.plot_primary.save_dialog(self, "primary_distribution.png"))
        self.btn_save_secondary_plot.clicked.connect(lambda: self.plot_secondary.save_dialog(self, "secondary_distribution.png"))
        self.btn_run_monte_carlo.clicked.connect(self.on_run_monte_carlo)
        self.btn_save_mc_plot.clicked.connect(lambda: self.plot_monte_carlo.save_dialog(self, "monte_carlo.png"))
        self.btn_save_mc_report.clicked.connect(self.on_save_monte_carlo_report)
        self.chk_show_primary.toggled.connect(self._refresh_monte_carlo_plot)
        self.chk_show_secondary.toggled.connect(self._refresh_monte_carlo_plot)
        self.mc_title_edit.textChanged.connect(self._refresh_monte_carlo_plot)
        self.mc_xformat_combo.currentIndexChanged.connect(self._refresh_monte_carlo_plot)
        self.mc_yformat_combo.currentIndexChanged.connect(self._refresh_monte_carlo_plot)
        self.sig.done_primary.connect(self.on_done_primary)
        self.sig.done_secondary.connect(self.on_done_secondary)
        self.sig.done_hangup.connect(self.on_done_hangup)
        self.sig.done_monte_carlo.connect(self.on_done_monte_carlo)

    def update_models_from_ui(self):
        self.rock = RockMass(
            rock_type=self.rock_type.text() or "Unknown",
            MRMR=self.mrmr.value(),
            IRS=self.irs.value(),
            mi=self.mi.value(),
            frac_freq=self.ff.value(),
            frac_condition=self.fc.value(),
            density=self.density.value(),
        )
        sets = []
        for i,(dip,dipr,dd,ddr,jc,s_min,s_mean,s_max) in enumerate(self.joint_widgets):
            sets.append(JointSet(
                name=f"Set{i+1}",
                mean_dip=dip.value(),
                dip_range=dipr.value(),
                mean_dip_dir=dd.value(),
                dip_dir_range=ddr.value(),
                spacing=SpacingDist("trunc_exp", s_min.value(), s_mean.value(), s_max.value()),
                JC=jc.value()
            ))
        self.joint_sets = sets
        self.cave = CaveFace(
            dip=self.cave_dip.value(),
            dip_dir=self.cave_ddir.value(),
            stress_dip=self.st_dip.value(),
            stress_strike=self.st_strike.value(),
            stress_normal=self.st_norm.value(),
            allow_stress_fractures=self.allow_sf.isChecked(),
            spalling_pct=self.spalling.value(),
        )
        self.defaults = Defaults(
            LHD_cutoff_m3=self.lhd_cutoff.value(),
            seed=int(self.seed.value()),
            arching_pct=self.arching_pct.value(),
            arch_stress_conc=self.arch_factor.value()
        )
        self.secondary = SecondaryRun(
            draw_height=self.draw_height.value(),
            max_caving_height=self.max_caving.value(),
            swell_factor=self.swell.value(),
            active_draw_width=self.draw_width.value(),
            added_fines_pct=self.add_fines.value(),
            rate_cm_day=self.rate.value(),
            drawbell_upper_width=self.upper_w.value(),
            drawbell_lower_width=self.lower_w.value(),
        )

    def on_run_primary(self):
        self.update_models_from_ui()
        n = int(self.nblocks.value())
        def work():
            blocks = generate_primary_blocks(n, self.rock, self.joint_sets, self.cave, self.defaults, seed=self.defaults.seed or 1234)
            primary_fines_ratio = self.cave.spalling_pct/100.0
            out_path = os.path.join(os.getcwd(), "primary_output.prm")
            write_prm(out_path, self.rock, self.cave, blocks, primary_fines_ratio)
            self.sig.done_primary.emit(blocks, primary_fines_ratio, out_path)
        threading.Thread(target=work, daemon=True).start()

    def on_done_primary(self, blocks, fines_ratio, path):
        self.primary_blocks = blocks
        self.primary_fines_ratio = fines_ratio
        stats = distributions_from_blocks(blocks)
        self.plot_primary.plot_distributions(stats, None, title="Primary fragmentation (cum. mass)")
        self.btn_save_prm.setEnabled(True)
        self.btn_save_primary_plot.setEnabled(True)
        QMessageBox.information(self, "Primary run complete", f"Generated {len(blocks)} blocks.\nTemporary .PRM written to:\n{path}")

    def on_save_prm(self):
        if not self.primary_blocks:
            QMessageBox.warning(self, "No primary run", "Run primary first.")
            return
        path,_ = QFileDialog.getSaveFileName(self, "Save .PRM", "run.prm", "PRM files (*.prm)")
        if path:
            write_prm(path, self.rock, self.cave, self.primary_blocks, self.primary_fines_ratio)
            QMessageBox.information(self, "Saved", f"Wrote {path}")

    def on_run_secondary(self):
        if not self.primary_blocks:
            QMessageBox.warning(self, "No primary blocks", "Run primary first.")
            return
        self.update_models_from_ui()
        def work():
            mu = average_scatter_deg_from_jointsets(self.joint_sets)
            sec_blocks, sec_fines_ratio = run_secondary(self.primary_blocks, self.rock, self.secondary, self.defaults, mu_scatter_deg=mu, primary_fines_ratio=self.primary_fines_ratio)
            out_path = os.path.join(os.getcwd(), "secondary_output.sec")
            write_sec(out_path, self.rock, self.cave, sec_blocks, self.primary_fines_ratio)
            self.sig.done_secondary.emit(sec_blocks, sec_fines_ratio, out_path)
        threading.Thread(target=work, daemon=True).start()

    def on_done_secondary(self, sec_blocks, sec_fines_ratio, path):
        self.secondary_blocks = sec_blocks
        prim_stats = distributions_from_blocks(self.primary_blocks)
        class P: 
            def __init__(self,b): self.V,self.Omega,self.joints_inside=b.V,b.Omega,b.joints_inside
        sec_stats = distributions_from_blocks([P(b) for b in sec_blocks])
        self.plot_secondary.plot_distributions(prim_stats, sec_stats, title="Primary vs Secondary (cum. mass)")
        self.btn_save_sec.setEnabled(True)
        self.btn_save_secondary_plot.setEnabled(True)

        if self.hang_method.currentText().startswith("Ore-pass"):
            stats = orepass_hangups(sec_blocks, bell_width=self.secondary.drawbell_lower_width, seed=self.defaults.seed or 1234)
            self.sig.done_hangup.emit(stats)
        else:
            area = self.secondary.drawbell_lower_width * self.secondary.drawbell_upper_width
            stats = kear_hangups(sec_blocks, bell_area=area, seed=self.defaults.seed or 1234)
            self.sig.done_hangup.emit(stats)

        QMessageBox.information(self, "Secondary complete", f"Generated {len(sec_blocks)} secondary blocks.\nTemporary .SEC written to:\n{path}")

    def on_save_sec(self):
        if not self.secondary_blocks:
            QMessageBox.warning(self, "No secondary run", "Run secondary first.")
            return
        path,_ = QFileDialog.getSaveFileName(self, "Save .SEC", "run.sec", "SEC files (*.sec)")
        if path:
            write_sec(path, self.rock, self.cave, self.secondary_blocks, self.primary_fines_ratio)
            QMessageBox.information(self, "Saved", f"Wrote {path}")

    def on_done_hangup(self, stats: dict):
        self.lbl_hang.setText(f"Hang-ups → High: {stats['n_high']}  Low: {stats['n_low']}  Total hang-up tons (proxy): {stats['total_hangup_tons']:.1f} t")

    def on_run_monte_carlo(self):
        self.update_models_from_ui()
        runs = int(self.mc_runs.value())
        nblocks = int(self.mc_blocks.value())
        variation = float(self.mc_variation.value())

        self._last_monte_carlo_result = None
        self._last_monte_carlo_inputs = {
            "rock": asdict(self.rock),
            "joint_sets": [asdict(js) for js in self.joint_sets],
            "cave": asdict(self.cave),
            "defaults": asdict(self.defaults),
            "secondary": asdict(self.secondary),
        }
        self._last_monte_carlo_settings = {
            "runs": runs,
            "blocks_per_run": nblocks,
            "variation_pct": variation,
        }
        self.btn_save_mc_plot.setEnabled(False)
        self.btn_save_mc_report.setEnabled(False)
        self.lbl_mc_stats.setText("Monte Carlo: running…")
        self._refresh_monte_carlo_plot()

        def work():
            primary_results = []
            primary_avg_volumes = []
            secondary_results = []
            secondary_avg_volumes = []
            max_workers = max(1, min(runs, os.cpu_count() or 4))
            base_inputs = dict(self._last_monte_carlo_inputs or {})

            def consume_result(primary_stats: dict, secondary_stats: dict):
                primary_results.append(primary_stats)
                primary_avg_volumes.append(primary_stats.get("avg_volume", 0.0))
                secondary_results.append(secondary_stats)
                secondary_avg_volumes.append(secondary_stats.get("avg_volume", 0.0))

            seeds = [random.randint(0, 10**9) for _ in range(runs)]

            if max_workers == 1:
                for seed in seeds:
                    primary_stats, sec_stats = _monte_carlo_worker(seed, nblocks, variation, base_inputs)
                    consume_result(primary_stats, sec_stats)
            else:
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(_monte_carlo_worker, seed, nblocks, variation, base_inputs)
                            for seed in seeds
                        ]
                        for fut in as_completed(futures):
                            primary_stats, sec_stats = fut.result()
                            consume_result(primary_stats, sec_stats)
                except Exception as exc:
                    try:
                        self.sig.log.emit(f"Process pool failed ({exc}); falling back to threads.")
                    except Exception:  # pragma: no cover - logging is best effort
                        pass
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(_monte_carlo_worker, seed, nblocks, variation, base_inputs)
                            for seed in seeds
                        ]
                        for fut in as_completed(futures):
                            primary_stats, sec_stats = fut.result()
                            consume_result(primary_stats, sec_stats)

            if not primary_results:
                return

            def envelope(stats_list):
                xs_local = [lo for lo, _ in stats_list[0]["bins"]]
                cum_mass_runs = [stats["cum_mass"] for stats in stats_list]
                avg_line_local = [statistics.mean(vals) for vals in zip(*cum_mass_runs)]
                min_line_local = [min(vals) for vals in zip(*cum_mass_runs)]
                max_line_local = [max(vals) for vals in zip(*cum_mass_runs)]
                return xs_local, avg_line_local, min_line_local, max_line_local

            primary_xs, p_avg, p_min, p_max = envelope(primary_results)
            if secondary_results:
                secondary_xs, s_avg, s_min, s_max = envelope(secondary_results)
            else:
                secondary_xs, s_avg, s_min, s_max = [], [], [], []

            info = {
                "runs": runs,
                "variation_pct": variation,
                "primary": {
                    "mean_avg_volume": statistics.mean(primary_avg_volumes) if primary_avg_volumes else 0.0,
                    "min_avg_volume": min(primary_avg_volumes) if primary_avg_volumes else 0.0,
                    "max_avg_volume": max(primary_avg_volumes) if primary_avg_volumes else 0.0,
                },
                "secondary": {
                    "mean_avg_volume": statistics.mean(secondary_avg_volumes) if secondary_avg_volumes else 0.0,
                    "min_avg_volume": min(secondary_avg_volumes) if secondary_avg_volumes else 0.0,
                    "max_avg_volume": max(secondary_avg_volumes) if secondary_avg_volumes else 0.0,
                },
            }
            self.sig.done_monte_carlo.emit(
                {
                    "primary": {
                        "xs": primary_xs,
                        "series": [
                            ("Primary cumulative mass (average)", p_avg),
                            ("Primary cumulative mass (minimum)", p_min),
                            ("Primary cumulative mass (maximum)", p_max),
                        ],
                    },
                    "secondary": {
                        "xs": secondary_xs,
                        "series": [
                            ("Secondary cumulative mass (average)", s_avg),
                            ("Secondary cumulative mass (minimum)", s_min),
                            ("Secondary cumulative mass (maximum)", s_max),
                        ],
                    } if secondary_xs else None,
                    "info": info,
                }
            )

        threading.Thread(target=work, daemon=True).start()

    def on_done_monte_carlo(self, result: dict):
        self._last_monte_carlo_result = result
        self._ensure_monte_carlo_style_controls()
        self._refresh_monte_carlo_plot()
        info = result.get("info", {})
        p_info = info.get("primary", {})
        s_info = info.get("secondary", {})
        self.lbl_mc_stats.setText(
            "Monte Carlo runs: {runs} | Primary avg volume {p_mean:.2f} m³ (min {p_min:.2f}, max {p_max:.2f}) | "
            "Secondary avg volume {s_mean:.2f} m³ (min {s_min:.2f}, max {s_max:.2f}) | Variation ±{var:.1f}%".format(
                runs=info.get("runs", 0),
                p_mean=p_info.get("mean_avg_volume", 0.0),
                p_min=p_info.get("min_avg_volume", 0.0),
                p_max=p_info.get("max_avg_volume", 0.0),
                s_mean=s_info.get("mean_avg_volume", 0.0),
                s_min=s_info.get("min_avg_volume", 0.0),
                s_max=s_info.get("max_avg_volume", 0.0),
                var=info.get("variation_pct", 0.0),
            )
        )
        self.btn_save_mc_report.setEnabled(True)

    def _refresh_monte_carlo_plot(self):
        result = self._last_monte_carlo_result
        xs_list: List[List[float]] = []
        lines: List[List[float]] = []
        labels: List[str] = []
        styles: List[Dict[str, object]] = []

        if result:
            self._ensure_monte_carlo_style_controls()

        if result and self.chk_show_primary.isChecked():
            primary = result.get("primary") or {}
            xs = primary.get("xs")
            series = primary.get("series") or []
            if xs and series:
                for label, values in series:
                    xs_list.append(xs)
                    lines.append(values)
                    labels.append(label)
                    styles.append(self._collect_series_style(label))

        if result and self.chk_show_secondary.isChecked():
            secondary = result.get("secondary") or {}
            xs = secondary.get("xs")
            series = secondary.get("series") or []
            if xs and series:
                for label, values in series:
                    xs_list.append(xs)
                    lines.append(values)
                    labels.append(label)
                    styles.append(self._collect_series_style(label))

        if lines:
            title_text = self.mc_title_edit.text().strip() if hasattr(self, "mc_title_edit") else ""
            if not title_text:
                title_text = "Monte Carlo cumulative mass envelopes"
            self.plot_monte_carlo.plot_lines(
                xs_list,
                lines,
                labels=labels,
                title=title_text,
                xlabel="Block volume (m³)",
                ylabel="Cumulative mass (%)",
                logx=True,
                styles=styles,
                x_formatter=self._axis_formatter_from_choice(self.mc_xformat_combo.currentText() if hasattr(self, "mc_xformat_combo") else "Auto"),
                y_formatter=self._axis_formatter_from_choice(self.mc_yformat_combo.currentText() if hasattr(self, "mc_yformat_combo") else "Auto"),
            )
            self.btn_save_mc_plot.setEnabled(True)
        else:
            self.plot_monte_carlo.fig.clear()
            self.plot_monte_carlo.canvas.draw_idle()
            self.plot_monte_carlo.has_data = False
            self.btn_save_mc_plot.setEnabled(False)

    def on_save_monte_carlo_report(self):
        if not self._last_monte_carlo_result or not self._last_monte_carlo_settings:
            QMessageBox.warning(self, "No Monte Carlo run", "Run Monte Carlo before saving a report.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Monte Carlo report",
            "monte_carlo_report.docx",
            "Word document (*.docx);;All files (*)",
        )
        if not path:
            return

        if not path.lower().endswith(".docx"):
            path += ".docx"

        try:
            from docx import Document
            from docx.enum.text import WD_ALIGN_PARAGRAPH
        except ImportError:
            QMessageBox.critical(
                self,
                "Missing dependency",
                "Saving Word reports requires the 'python-docx' package.\n"
                "Install it with 'pip install python-docx' and try again.",
            )
            return

        info = self._last_monte_carlo_result.get("info", {})
        primary = self._last_monte_carlo_result.get("primary") or {}
        secondary = self._last_monte_carlo_result.get("secondary") or {}
        doc = Document()
        doc.add_heading("Monte Carlo Fragmentation Report", 0)
        subtitle = doc.add_paragraph(datetime.now().strftime("Generated on %Y-%m-%d at %H:%M"))
        subtitle.alignment = WD_ALIGN_PARAGRAPH.LEFT

        doc.add_heading("Simulation summary", level=1)
        summary_table = doc.add_table(rows=3, cols=4)
        header_cells = summary_table.rows[0].cells
        header_cells[0].text = "Stage"
        header_cells[1].text = "Average volume (m³)"
        header_cells[2].text = "Minimum (m³)"
        header_cells[3].text = "Maximum (m³)"

        primary_row = summary_table.rows[1].cells
        primary_row[0].text = "Primary"
        primary_row[1].text = f"{info.get('primary', {}).get('mean_avg_volume', 0.0):.3f}"
        primary_row[2].text = f"{info.get('primary', {}).get('min_avg_volume', 0.0):.3f}"
        primary_row[3].text = f"{info.get('primary', {}).get('max_avg_volume', 0.0):.3f}"

        secondary_row = summary_table.rows[2].cells
        secondary_row[0].text = "Secondary"
        secondary_row[1].text = f"{info.get('secondary', {}).get('mean_avg_volume', 0.0):.3f}"
        secondary_row[2].text = f"{info.get('secondary', {}).get('min_avg_volume', 0.0):.3f}"
        secondary_row[3].text = f"{info.get('secondary', {}).get('max_avg_volume', 0.0):.3f}"

        doc.add_paragraph(
            "Runs performed: {runs}  |  Blocks per run: {blocks}  |  Variation: ±{var:.1f}%".format(
                runs=self._last_monte_carlo_settings.get("runs", 0),
                blocks=self._last_monte_carlo_settings.get("blocks_per_run", 0),
                var=self._last_monte_carlo_settings.get("variation_pct", 0.0),
            )
        )

        doc.add_heading("Simulation settings", level=1)
        settings_table = doc.add_table(rows=len(self._last_monte_carlo_settings) + 1, cols=2)
        settings_table.rows[0].cells[0].text = "Setting"
        settings_table.rows[0].cells[1].text = "Value"
        for idx, (key, value) in enumerate(self._last_monte_carlo_settings.items(), start=1):
            settings_table.rows[idx].cells[0].text = key.replace("_", " ").title()
            settings_table.rows[idx].cells[1].text = str(value)

        doc.add_heading("Base input snapshot", level=1)

        def add_dict_table(title: str, payload: dict):
            if not payload:
                doc.add_paragraph(f"{title}: (not available)")
                return
            doc.add_paragraph(title, style="List Bullet")
            table = doc.add_table(rows=len(payload) + 1, cols=2)
            table.rows[0].cells[0].text = "Parameter"
            table.rows[0].cells[1].text = "Value"
            for idx, (key, value) in enumerate(payload.items(), start=1):
                table.rows[idx].cells[0].text = key.replace("_", " ")
                table.rows[idx].cells[1].text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

        base_inputs = self._last_monte_carlo_inputs or {}
        add_dict_table("Rock mass", base_inputs.get("rock", {}))
        add_dict_table("Cave face", base_inputs.get("cave", {}))
        add_dict_table("Defaults", base_inputs.get("defaults", {}))
        add_dict_table("Secondary fragmentation", base_inputs.get("secondary", {}))

        joint_sets = base_inputs.get("joint_sets") or []
        if joint_sets:
            doc.add_heading("Joint sets", level=2)
            for idx, js in enumerate(joint_sets, start=1):
                doc.add_paragraph(f"Joint set {idx}: {js.get('name', 'Set')}", style="List Number")
                js_table = doc.add_table(rows=len(js) + 1, cols=2)
                js_table.rows[0].cells[0].text = "Parameter"
                js_table.rows[0].cells[1].text = "Value"
                for row_idx, (key, value) in enumerate(js.items(), start=1):
                    js_table.rows[row_idx].cells[0].text = key.replace("_", " ")
                    if isinstance(value, dict):
                        js_table.rows[row_idx].cells[1].text = json.dumps(value, ensure_ascii=False)
                    else:
                        js_table.rows[row_idx].cells[1].text = str(value)

        def add_envelope_section(title: str, xs: List[float], series: List[tuple[str, List[float]]]):
            doc.add_heading(title, level=1)
            if not xs or not series:
                doc.add_paragraph("No data available.")
                return
            table = doc.add_table(rows=len(xs) + 1, cols=len(series) + 1)
            table.rows[0].cells[0].text = "Block volume (m³)"
            for idx, (label, _) in enumerate(series, start=1):
                table.rows[0].cells[idx].text = label
            for row_idx, vol in enumerate(xs, start=1):
                table.rows[row_idx].cells[0].text = f"{vol:.4f}"
                for col_idx, (_, values) in enumerate(series, start=1):
                    try:
                        table.rows[row_idx].cells[col_idx].text = f"{values[row_idx-1]:.3f}"
                    except IndexError:
                        table.rows[row_idx].cells[col_idx].text = "-"

        add_envelope_section(
            "Primary cumulative-mass envelope",
            primary.get("xs") or [],
            primary.get("series") or [],
        )
        add_envelope_section(
            "Secondary cumulative-mass envelope",
            secondary.get("xs") or [],
            secondary.get("series") or [],
        )

        try:
            doc.save(path)
        except Exception as exc:  # pragma: no cover - UI feedback
            QMessageBox.critical(self, "Save failed", f"Could not save report:\n{exc}")
        else:
            QMessageBox.information(self, "Report saved", f"Monte Carlo report saved to:\n{path}")

    def _ensure_monte_carlo_style_controls(self):
        if not hasattr(self, "mc_series_container"):
            return
        result = self._last_monte_carlo_result or {}
        labels = []
        for key in ("primary", "secondary"):
            series = (result.get(key) or {}).get("series") or []
            labels.extend([label for label, _ in series])
        existing = set(self._mc_style_widgets.keys())
        for label in labels:
            if label not in existing:
                self._add_monte_carlo_style_row(label)
        for label, widgets in self._mc_style_widgets.items():
            container = widgets.get("container")
            if container is not None:
                container.setVisible(label in labels)

    def _add_monte_carlo_style_row(self, label: str):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        base_label = QLabel(label)
        base_label.setMinimumWidth(180)
        layout.addWidget(base_label)

        name_edit = QLineEdit(label)
        name_edit.setPlaceholderText("Legend label")
        layout.addWidget(name_edit)

        color_combo = QComboBox()
        for color_name, color_value in MC_COLOR_OPTIONS:
            color_combo.addItem(color_name, color_value)
        default_color_index = self._mc_color_cursor % len(MC_COLOR_OPTIONS)
        color_combo.setCurrentIndex(default_color_index)
        self._mc_color_cursor += 1
        layout.addWidget(color_combo)

        dash_combo = QComboBox()
        for text, pattern in LINE_STYLE_OPTIONS:
            dash_combo.addItem(text, pattern)
        dash_combo.setCurrentIndex(self._index_for_dash(self._default_dash_for_label(label)))
        layout.addWidget(dash_combo)

        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.5, 8.0)
        width_spin.setSingleStep(0.1)
        width_spin.setValue(self._default_width_for_label(label))
        layout.addWidget(width_spin)

        layout.addStretch(1)
        self.mc_series_container.addWidget(row)
        self._mc_style_widgets[label] = {
            "container": row,
            "name": name_edit,
            "color": color_combo,
            "dash": dash_combo,
            "width": width_spin,
        }
        name_edit.textChanged.connect(self._refresh_monte_carlo_plot)
        color_combo.currentIndexChanged.connect(self._refresh_monte_carlo_plot)
        dash_combo.currentIndexChanged.connect(self._refresh_monte_carlo_plot)
        width_spin.valueChanged.connect(self._refresh_monte_carlo_plot)

    def _collect_series_style(self, label: str) -> Dict[str, object]:
        widgets = self._mc_style_widgets.get(label)
        if not widgets:
            return {"label": label}
        name_widget = widgets.get("name")
        color_combo = widgets.get("color")
        dash_combo = widgets.get("dash")
        width_spin = widgets.get("width")
        custom_label = name_widget.text().strip() if isinstance(name_widget, QLineEdit) else label
        if not custom_label:
            custom_label = label
        color = color_combo.currentData() if isinstance(color_combo, QComboBox) else None
        linestyle = dash_combo.currentData() if isinstance(dash_combo, QComboBox) else "-"
        linewidth = width_spin.value() if isinstance(width_spin, QDoubleSpinBox) else 1.5
        return {
            "label": custom_label,
            "color": color,
            "linestyle": linestyle or "-",
            "linewidth": linewidth,
        }

    def _default_dash_for_label(self, label: str) -> str:
        text = label.lower()
        if "minimum" in text:
            return "--"
        if "maximum" in text:
            return "-."
        return "-"

    def _default_width_for_label(self, label: str) -> float:
        text = label.lower()
        if "average" in text:
            return 2.0
        if any(word in text for word in ("minimum", "maximum")):
            return 1.3
        return 1.5

    def _index_for_dash(self, dash: str) -> int:
        for idx, (_, pattern) in enumerate(LINE_STYLE_OPTIONS):
            if pattern == dash:
                return idx
        return 0

    def _axis_formatter_from_choice(self, choice: str):
        option = (choice or "Auto").strip()
        if option == "Auto":
            return None
        if option == "0":
            return StrMethodFormatter("{x:.0f}")
        if option == "0.0":
            return StrMethodFormatter("{x:.1f}")
        if option == "0.00":
            return StrMethodFormatter("{x:.2f}")
        if option.startswith("Scientific"):
            formatter = ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 4))
            return formatter
        return None

def launch():
    app = QApplication([])
    mw = MainWindow()
    mw.show()
    app.exec()
