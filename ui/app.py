from __future__ import annotations
import os, random, statistics, threading
from dataclasses import replace
from typing import List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QHBoxLayout, QComboBox
)
from PySide6.QtCore import Signal, QObject, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..engine.models import RockMass, JointSet, SpacingDist, CaveFace, Defaults, SecondaryRun, PrimaryBlock
from ..engine.primary import generate_primary_blocks
from ..engine.secondary import run_secondary, average_scatter_deg_from_jointsets
from ..engine.hangup import orepass_hangups, kear_hangups
from ..engine.io_formats import distributions_from_blocks, write_prm, write_sec

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

    def plot_lines(self, xs: List[float] | List[List[float]], ys_list: List[List[float]], labels: List[str] | None = None,
                   title: str = "", xlabel: str = "", ylabel: str = "", logx: bool = False):
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
            ax.plot(cur_xs, ys, label=label)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":")
        if labels:
            ax.legend()
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
        self.plot_monte_carlo = PlotWidget()
        plot_layout.addWidget(self.plot_monte_carlo)
        btn_row = QHBoxLayout(); btn_row.addStretch(1)
        self.btn_save_mc_plot = QPushButton("Save chart…")
        self.btn_save_mc_plot.setEnabled(False)
        btn_row.addWidget(self.btn_save_mc_plot)
        plot_layout.addLayout(btn_row)
        self.lbl_mc_stats = QLabel("Monte Carlo: -")
        plot_layout.addWidget(self.lbl_mc_stats)
        layout.addLayout(plot_layout, 1)

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

    def _randomize_value(self, value: float, variation_pct: float, minimum: float | None = None, maximum: float | None = None):
        if variation_pct <= 0:
            new_val = value
        else:
            delta = variation_pct / 100.0
            new_val = value * (1.0 + random.uniform(-delta, delta))
        if minimum is not None:
            new_val = max(minimum, new_val)
        if maximum is not None:
            new_val = min(maximum, new_val)
        return new_val

    def _randomize_joint_set(self, js: JointSet, variation_pct: float) -> JointSet:
        spacing_vals = [
            self._randomize_value(js.spacing.min, variation_pct, 0.01),
            self._randomize_value(js.spacing.mean, variation_pct, 0.02),
            self._randomize_value(js.spacing.max_or_90pct, variation_pct, 0.05),
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
            mean_dip=self._randomize_value(js.mean_dip, variation_pct, 0.0, 90.0),
            dip_range=self._randomize_value(js.dip_range, variation_pct, 0.0, 90.0),
            mean_dip_dir=self._randomize_value(js.mean_dip_dir, variation_pct, 0.0, 360.0),
            dip_dir_range=self._randomize_value(js.dip_dir_range, variation_pct, 0.0, 180.0),
            spacing=spacing,
            JC=int(round(self._randomize_value(js.JC, variation_pct, 0, 40)))
        )

    def on_run_monte_carlo(self):
        self.update_models_from_ui()
        runs = int(self.mc_runs.value())
        nblocks = int(self.mc_blocks.value())
        variation = float(self.mc_variation.value())

        def work():
            primary_results = []
            primary_avg_volumes = []
            secondary_results = []
            secondary_avg_volumes = []
            for _ in range(runs):
                rock = RockMass(
                    rock_type=self.rock.rock_type,
                    MRMR=self._randomize_value(self.rock.MRMR, variation, 0.0, 100.0),
                    IRS=self._randomize_value(self.rock.IRS, variation, 1.0, 500.0),
                    IBS=self.rock.IBS,
                    mi=self._randomize_value(self.rock.mi, variation, 1.0, 50.0),
                    frac_freq=self._randomize_value(self.rock.frac_freq, variation, 0.0, 20.0),
                    frac_condition=int(round(self._randomize_value(self.rock.frac_condition, variation, 0, 40))),
                    density=self._randomize_value(self.rock.density, variation, 1500.0, 4500.0),
                )
                joint_sets = [self._randomize_joint_set(js, variation) for js in self.joint_sets]
                cave = replace(self.cave, spalling_pct=self._randomize_value(self.cave.spalling_pct, variation, 0.0, 100.0))
                defaults = Defaults(
                    LHD_cutoff_m3=self._randomize_value(self.defaults.LHD_cutoff_m3, variation, 0.1, 50.0),
                    seed=random.randint(0, 10**9),
                    arching_pct=self._randomize_value(self.defaults.arching_pct, variation, 0.0, 1.0),
                    arch_stress_conc=self._randomize_value(self.defaults.arch_stress_conc, variation, 1.0, 100.0),
                )
                secondary_params = SecondaryRun(
                    draw_height=self._randomize_value(self.secondary.draw_height, variation, 1.0, 2000.0),
                    max_caving_height=self._randomize_value(self.secondary.max_caving_height, variation, 1.0, 5000.0),
                    swell_factor=self._randomize_value(self.secondary.swell_factor, variation, 1.0, 3.0),
                    active_draw_width=self._randomize_value(self.secondary.active_draw_width, variation, 1.0, 200.0),
                    added_fines_pct=self._randomize_value(self.secondary.added_fines_pct, variation, 0.0, 80.0),
                    rate_cm_day=self._randomize_value(self.secondary.rate_cm_day, variation, 0.0, 100.0),
                    drawbell_upper_width=self._randomize_value(self.secondary.drawbell_upper_width, variation, 1.0, 50.0),
                    drawbell_lower_width=self._randomize_value(self.secondary.drawbell_lower_width, variation, 1.0, 50.0),
                )
                blocks = generate_primary_blocks(nblocks, rock, joint_sets, cave, defaults, seed=defaults.seed)
                primary_stats = distributions_from_blocks(blocks)
                primary_results.append(primary_stats)
                primary_avg_volumes.append(primary_stats["avg_volume"])

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
                class P:
                    def __init__(self, b):
                        self.V, self.Omega, self.joints_inside = b.V, b.Omega, b.joints_inside

                sec_stats = distributions_from_blocks([P(b) for b in sec_blocks])
                secondary_results.append(sec_stats)
                secondary_avg_volumes.append(sec_stats["avg_volume"])

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
                        "xs": [primary_xs, primary_xs, primary_xs],
                        "lines": [p_avg, p_min, p_max],
                        "labels": [
                            "Primary cumulative mass (average)",
                            "Primary cumulative mass (minimum)",
                            "Primary cumulative mass (maximum)",
                        ],
                    },
                    "secondary": {
                        "xs": [secondary_xs, secondary_xs, secondary_xs] if secondary_xs else [],
                        "lines": [s_avg, s_min, s_max] if secondary_xs else [],
                        "labels": [
                            "Secondary cumulative mass (average)",
                            "Secondary cumulative mass (minimum)",
                            "Secondary cumulative mass (maximum)",
                        ] if secondary_xs else [],
                    },
                    "info": info,
                }
            )

        threading.Thread(target=work, daemon=True).start()

    def on_done_monte_carlo(self, result: dict):
        primary = result.get("primary", {})
        secondary = result.get("secondary", {})
        xs = []
        lines = []
        labels = []
        for payload in (primary, secondary):
            xs.extend(payload.get("xs", []))
            lines.extend(payload.get("lines", []))
            labels.extend(payload.get("labels", []))

        if lines:
            self.plot_monte_carlo.plot_lines(
                xs,
                lines,
                labels=labels,
                title="Monte Carlo cumulative mass envelopes",
                xlabel="Block volume (m³)",
                ylabel="Cumulative mass (%)",
                logx=True,
            )
            self.btn_save_mc_plot.setEnabled(True)

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

def launch():
    app = QApplication([])
    mw = MainWindow()
    mw.show()
    app.exec()
