from __future__ import annotations
import os, threading
from typing import List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QHBoxLayout, QComboBox
)
from PySide6.QtCore import Signal, QObject

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..engine.models import RockMass, JointSet, SpacingDist, CaveFace, Defaults, SecondaryRun, PrimaryBlock
from ..engine.primary import generate_primary_blocks
from ..engine.secondary import run_secondary, average_scatter_deg_from_jointsets
from ..engine.hangup import orepass_hangups, kear_hangups
from ..engine.io_formats import distributions_from_blocks, write_prm, write_sec

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6,4))
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout()
        lay.addWidget(self.canvas)
        self.setLayout(lay)

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

class AppSignals(QObject):
    log = Signal(str)
    done_primary = Signal(list, float, str)
    done_secondary = Signal(list, float, str)
    done_hangup = Signal(dict)

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

    def _build_tabs(self):
        self.tabs.addTab(self._build_geology_tab(), "Geology")
        self.tabs.addTab(self._build_cave_tab(), "Cave")
        self.tabs.addTab(self._build_primary_tab(), "Primary run")
        self.tabs.addTab(self._build_secondary_tab(), "Secondary & hang-up")
        self.tabs.addTab(self._build_defaults_tab(), "Defaults")

    def _build_geology_tab(self):
        w = QWidget(); g = QGridLayout(w)
        row = 0; g.addWidget(QLabel("<b>Rock mass</b>"), row,0,1,2); row+=1
        self.rock_type = QLineEdit(self.rock.rock_type); g.addWidget(QLabel("Rock type"), row,0); g.addWidget(self.rock_type,row,1); row+=1
        self.mrmr = QDoubleSpinBox(); self.mrmr.setRange(0,100); self.mrmr.setValue(self.rock.MRMR); g.addWidget(QLabel("MRMR"), row,0); g.addWidget(self.mrmr,row,1); row+=1
        self.irs = QDoubleSpinBox(); self.irs.setRange(1,500); self.irs.setValue(self.rock.IRS); g.addWidget(QLabel("IRS (MPa)"), row,0); g.addWidget(self.irs,row,1); row+=1
        self.mi = QDoubleSpinBox(); self.mi.setRange(1,50); self.mi.setValue(self.rock.mi); g.addWidget(QLabel("mi (Hoek–Brown)"), row,0); g.addWidget(self.mi,row,1); row+=1
        self.ff = QDoubleSpinBox(); self.ff.setRange(0,20); self.ff.setDecimals(2); self.ff.setValue(self.rock.frac_freq); g.addWidget(QLabel("Fracture/veinlet freq (1/m)"), row,0); g.addWidget(self.ff,row,1); row+=1
        self.fc = QSpinBox(); self.fc.setRange(0,40); self.fc.setValue(self.rock.frac_condition); g.addWidget(QLabel("Fracture/veinlet condition (0–40)"), row,0); g.addWidget(self.fc,row,1); row+=1
        self.density = QDoubleSpinBox(); self.density.setRange(1500,4500); self.density.setValue(self.rock.density); g.addWidget(QLabel("Density (kg/m³)"), row,0); g.addWidget(self.density,row,1); row+=1

        row+=1; g.addWidget(QLabel("<b>Joint sets</b>"), row,0,1,2); row+=1
        self.joint_widgets = []
        for i,js in enumerate(self.joint_sets):
            g.addWidget(QLabel(f"<u>{js.name}</u>"), row,0,1,2); row+=1
            dip = QDoubleSpinBox(); dip.setRange(0,90); dip.setValue(js.mean_dip)
            dipr= QDoubleSpinBox(); dipr.setRange(0,90); dipr.setValue(js.dip_range)
            dd  = QDoubleSpinBox(); dd.setRange(0,360); dd.setValue(js.mean_dip_dir)
            ddr = QDoubleSpinBox(); ddr.setRange(0,180); ddr.setValue(js.dip_dir_range)
            jc  = QSpinBox(); jc.setRange(0,40); jc.setValue(js.JC)
            s_min=QDoubleSpinBox(); s_min.setRange(0.01,50); s_min.setDecimals(2); s_min.setValue(js.spacing.min)
            s_mean=QDoubleSpinBox(); s_mean.setRange(0.02,50); s_mean.setDecimals(2); s_mean.setValue(js.spacing.mean)
            s_max=QDoubleSpinBox(); s_max.setRange(0.03,200); s_max.setDecimals(2); s_max.setValue(js.spacing.max_or_90pct)

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
        self.cave_dip = QDoubleSpinBox(); self.cave_dip.setRange(0,90); self.cave_dip.setValue(self.cave.dip); g.addWidget(QLabel("Face dip (°)"), row,0); g.addWidget(self.cave_dip,row,1); row+=1
        self.cave_ddir = QDoubleSpinBox(); self.cave_ddir.setRange(0,360); self.cave_ddir.setValue(self.cave.dip_dir); g.addWidget(QLabel("Face dip direction (°)"), row,0); g.addWidget(self.cave_ddir,row,1); row+=1
        self.st_dip = QDoubleSpinBox(); self.st_dip.setRange(0,100); self.st_dip.setValue(self.cave.stress_dip); g.addWidget(QLabel("Dip stress (MPa)"), row,0); g.addWidget(self.st_dip,row,1); row+=1
        self.st_strike = QDoubleSpinBox(); self.st_strike.setRange(0,100); self.st_strike.setValue(self.cave.stress_strike); g.addWidget(QLabel("Strike stress (MPa)"), row,0); g.addWidget(self.st_strike,row,1); row+=1
        self.st_norm = QDoubleSpinBox(); self.st_norm.setRange(0,100); self.st_norm.setValue(self.cave.stress_normal); g.addWidget(QLabel("Normal stress (MPa)"), row,0); g.addWidget(self.st_norm,row,1); row+=1
        self.allow_sf = QCheckBox("Allow stress fractures"); self.allow_sf.setChecked(self.cave.allow_stress_fractures); g.addWidget(self.allow_sf,row,0,1,2); row+=1
        self.spalling = QDoubleSpinBox(); self.spalling.setRange(0,100); self.spalling.setDecimals(1); self.spalling.setValue(self.cave.spalling_pct); g.addWidget(QLabel("% spalling as fines"), row,0); g.addWidget(self.spalling,row,1); row+=1
        return w

    def _build_primary_tab(self):
        w = QWidget(); g = QGridLayout(w)
        row=0; g.addWidget(QLabel("<b>Primary run</b>"), row,0,1,2); row+=1
        self.nblocks = QSpinBox(); self.nblocks.setRange(100,200000); self.nblocks.setValue(20000); g.addWidget(QLabel("Blocks to generate"), row,0); g.addWidget(self.nblocks,row,1); row+=1
        self.btn_run_primary = QPushButton("Run primary")
        self.btn_save_prm = QPushButton("Save .PRM…"); self.btn_save_prm.setEnabled(False)
        hl = QHBoxLayout(); hl.addWidget(self.btn_run_primary); hl.addWidget(self.btn_save_prm)
        g.addLayout(hl, row,0,1,3); row+=1
        self.plot_primary = PlotWidget()
        g.addWidget(self.plot_primary, row,0,1,3); row+=1
        w.setLayout(g)
        return w

    def _build_secondary_tab(self):
        w = QWidget(); g = QGridLayout(w)
        row=0; g.addWidget(QLabel("<b>Secondary run & Hang-up</b>"), row,0,1,2); row+=1
        self.draw_height = QDoubleSpinBox(); self.draw_height.setRange(1,2000); self.draw_height.setValue(self.secondary.draw_height); g.addWidget(QLabel("Draw height (m)"), row,0); g.addWidget(self.draw_height,row,1); row+=1
        self.max_caving = QDoubleSpinBox(); self.max_caving.setRange(1,5000); self.max_caving.setValue(self.secondary.max_caving_height); g.addWidget(QLabel("Max caving height (m)"), row,0); g.addWidget(self.max_caving,row,1); row+=1
        self.swell = QDoubleSpinBox(); self.swell.setRange(1.0,3.0); self.swell.setSingleStep(0.05); self.swell.setValue(self.secondary.swell_factor); g.addWidget(QLabel("Swell factor"), row,0); g.addWidget(self.swell,row,1); row+=1
        self.draw_width = QDoubleSpinBox(); self.draw_width.setRange(1,200); self.draw_width.setValue(self.secondary.active_draw_width); g.addWidget(QLabel("Active draw width (m)"), row,0); g.addWidget(self.draw_width,row,1); row+=1
        self.add_fines = QDoubleSpinBox(); self.add_fines.setRange(0,80); self.add_fines.setValue(self.secondary.added_fines_pct); g.addWidget(QLabel("Added fines % (dilution)"), row,0); g.addWidget(self.add_fines,row,1); row+=1
        self.rate = QDoubleSpinBox(); self.rate.setRange(0,100); self.rate.setValue(self.secondary.rate_cm_day); g.addWidget(QLabel("Draw rate (cm/day)"), row,0); g.addWidget(self.rate,row,1); row+=1
        self.upper_w = QDoubleSpinBox(); self.upper_w.setRange(1,50); self.upper_w.setValue(self.secondary.drawbell_upper_width); g.addWidget(QLabel("Drawbell upper width (m)"), row,0); g.addWidget(self.upper_w,row,1); row+=1
        self.lower_w = QDoubleSpinBox(); self.lower_w.setRange(1,50); self.lower_w.setValue(self.secondary.drawbell_lower_width); g.addWidget(QLabel("Drawbell lower width (m)"), row,0); g.addWidget(self.lower_w,row,1); row+=1
        self.hang_method = QComboBox(); self.hang_method.addItems(["Ore-pass rules (width)","Kear method (area)"]); g.addWidget(QLabel("Hang-up method"), row,0); g.addWidget(self.hang_method,row,1); row+=1

        self.btn_run_secondary = QPushButton("Run secondary + hang-up")
        self.btn_save_sec = QPushButton("Save .SEC…"); self.btn_save_sec.setEnabled(False)
        hl = QHBoxLayout(); hl.addWidget(self.btn_run_secondary); hl.addWidget(self.btn_save_sec)
        g.addLayout(hl, row,0,1,3); row+=1
        self.plot_secondary = PlotWidget()
        g.addWidget(self.plot_secondary, row,0,1,3); row+=1
        self.lbl_hang = QLabel("Hang-up: -")
        g.addWidget(self.lbl_hang, row,0,1,3); row+=1

        w.setLayout(g)
        return w

    def _build_defaults_tab(self):
        w = QWidget(); g = QGridLayout(w)
        row=0; g.addWidget(QLabel("<b>Defaults</b>"), row,0,1,2); row+=1
        self.lhd_cutoff = QDoubleSpinBox(); self.lhd_cutoff.setRange(0.1,50); self.lhd_cutoff.setValue(self.defaults.LHD_cutoff_m3); g.addWidget(QLabel("LHD bucket cutoff (m³)"), row,0); g.addWidget(self.lhd_cutoff,row,1); row+=1
        self.seed = QSpinBox(); self.seed.setRange(0,10**9); self.seed.setValue(self.defaults.seed or 1234); g.addWidget(QLabel("Random seed"), row,0); g.addWidget(self.seed,row,1); row+=1
        self.arching_pct = QDoubleSpinBox(); self.arching_pct.setRange(0,1); self.arching_pct.setSingleStep(0.01); self.arching_pct.setValue(self.defaults.arching_pct); g.addWidget(QLabel("Arching split fraction (0–1)"), row,0); g.addWidget(self.arching_pct,row,1); row+=1
        self.arch_factor = QDoubleSpinBox(); self.arch_factor.setRange(1,100); self.arch_factor.setValue(self.defaults.arch_stress_conc); g.addWidget(QLabel("Arch stress factor (× cave pressure)"), row,0); g.addWidget(self.arch_factor,row,1); row+=1
        return w

    def _connect_signals(self):
        self.btn_run_primary.clicked.connect(self.on_run_primary)
        self.btn_save_prm.clicked.connect(self.on_save_prm)
        self.btn_run_secondary.clicked.connect(self.on_run_secondary)
        self.btn_save_sec.clicked.connect(self.on_save_sec)
        self.sig.done_primary.connect(self.on_done_primary)
        self.sig.done_secondary.connect(self.on_done_secondary)
        self.sig.done_hangup.connect(self.on_done_hangup)

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

def launch():
    app = QApplication([])
    mw = MainWindow()
    mw.show()
    app.exec()
