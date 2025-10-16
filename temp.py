import sys, os, platform, shutil, time, threading, queue
from collections import deque
import psutil
import platform
import pynvml
from ping3 import ping
import wmi

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QScrollArea, QGridLayout, QGroupBox, QSizePolicy
)
from PySide6.QtGui import QPainter, QPen, QColor
USING_PYSIDE = True


NVML_OK = False
try:
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

APP_TITLE = "System Monitor (CPU/RAM/Disk/VRAM/Net/Temps)"

class ThresholdProgressBar(QProgressBar):
    def __init__(self, threshold=80, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QPainter(self)
        pen = QPen(QColor("red"))
        pen.setWidth(2)
        qp.setPen(pen)
        x = int(self.width() * (self.threshold / 100.0))
        qp.drawLine(x, 0, x, self.height())
        qp.end()

def format_bytes(n: float) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"

def get_cpu_temp_psutil():
    try:
        # 1) LibreHardwareMonitor / OpenHardwareMonitor (실행 중이어야 함)
        for ns in (r"root\LibreHardwareMonitor", r"root\OpenHardwareMonitor"):
            try:
                c = wmi.WMI(namespace=ns)
                vals = []
                for s in c.Sensor():
                    # SensorType == 'Temperature', Name에 'CPU' or 'Package' 포함되는 값 우선
                    st = getattr(s, 'SensorType', '')
                    name = (getattr(s, 'Name', '') or '')
                    if st == 'Temperature' and ('CPU' in name or 'Package' in name):
                        v = getattr(s, 'Value', None)
                        if v is not None:
                            vals.append(float(v))
                if vals:
                    return max(vals)  # °C
            except Exception:
                pass
        # 2) ACPI (정밀도 낮은 경우 많음, 단위: 1/10 Kelvin)
        try:
            c = wmi.WMI(namespace=r"root\wmi")
            vals = []
            for t in c.MSAcpi_ThermalZoneTemperature():
                cur = getattr(t, 'CurrentTemperature', None)
                if cur is not None:
                    vals.append((cur / 10.0) - 273.15)  # Kelvin→C
            if vals:
                return max(vals)
        except Exception:
            pass
        return None

    except Exception:
        return None


def get_gpu_infos_nvml():
    if not NVML_OK:
        return []
    infos = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(raw_name, bytes):
                name = raw_name.decode("utf-8", errors="ignore")
            else:
                name = str(raw_name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            infos.append({
                "index": i,
                "name": name,
                "mem_total": mem.total,
                "mem_used": mem.used,
                "util": util.gpu,   # %
                "temp": temp        # C
            })
    except Exception:
        return []
    return infos

def get_all_disks_usage():
    disks = []
    try:
        parts = psutil.disk_partitions(all=False)
    except Exception:
        parts = []
    for p in parts:
        if isinstance(p.opts, str) and "cdrom" in p.opts.lower():
            continue
        try:
            du = psutil.disk_usage(p.mountpoint)
        except Exception:
            continue
        if platform.system().lower().startswith("win"):
            name = p.device or p.mountpoint
        else:
            name = p.mountpoint
        disks.append({
            "name": name,
            "mount": p.mountpoint,
            "pct": du.percent,
            "total": du.total,
            "used": du.used,
        })
    seen = set()
    unique = []
    for d in disks:
        if d["mount"] in seen:
            continue
        seen.add(d["mount"])
        unique.append(d)
    return unique

def ping_once(host: str, timeout_ms: int = 1000):
    try:
        latency = ping(host, timeout=timeout_ms/1000, unit="ms")
        return latency
    except Exception:
        return None

class MetricsWorker(QObject):
    data_ready = Signal(dict)

    def __init__(self, net_host="8.8.8.8", net_window=20, parent=None):
        super().__init__(parent)
        self._stop = False
        self.net_host = net_host
        self.net_samples = deque(maxlen=net_window)
        self.net_latencies = deque(maxlen=net_window)
        self.queue = queue.Queue()
        self._prev_disk = None
        self._prev_disk_ts = None
        try:
            self._prev_disk = psutil.disk_io_counters()
            self._prev_disk_ts = time.time()
        except Exception:
            self._prev_disk = None
            self._prev_disk_ts = None

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        self.ping_thread = threading.Thread(target=self._run_ping_loop, daemon=True)
        self.ping_thread.start()

    def stop(self):
        self._stop = True

    def _run_ping_loop(self):
        while not self._stop:
            latency = ping_once(self.net_host, timeout_ms=1000)
            success = latency is not None
            self.net_samples.append(success)
            if success:
                self.net_latencies.append(latency)
            time.sleep(2.0)

    def _sample_disk_busy_pct(self):
        try:
            now = psutil.disk_io_counters()
            now_ts = time.time()
        except Exception:
            return None

        if self._prev_disk is None or self._prev_disk_ts is None:
            self._prev_disk = now
            self._prev_disk_ts = now_ts
            return None
        if not hasattr(now, "busy_time") or not hasattr(self._prev_disk, "busy_time"):
            self._prev_disk = now
            self._prev_disk_ts = now_ts
            return None

        dt_ms = (now_ts - self._prev_disk_ts) * 1000.0
        if dt_ms <= 1e-6:
            return None

        d_busy_ms = float(now.busy_time - self._prev_disk.busy_time)
        pct = max(0.0, min(100.0, (d_busy_ms / dt_ms) * 100.0))

        self._prev_disk = now
        self._prev_disk_ts = now_ts
        return pct

    def _sample_proc_thread_stats(self):
        total = 0
        running = 0
        threads = 0
        try:
            for p in psutil.process_iter(attrs=['status', 'num_threads']):
                total += 1
                info = p.info
                try:
                    if info.get('status') == psutil.STATUS_RUNNING:
                        running += 1
                    nt = info.get('num_threads')
                    if nt is None:
                        nt = p.num_threads()
                    threads += int(nt)
                except Exception:
                    pass
        except Exception:
            pass
        background = max(0, total - running)
        return total, running, background, threads

    def _run(self):
        disks = get_all_disks_usage()

        while not self._stop:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                cpu_freq = psutil.cpu_freq()
                cpu_freq_mhz = cpu_freq.current if cpu_freq else None
                cpu_temp = get_cpu_temp_psutil()

                vm = psutil.virtual_memory()
                mem_pct = vm.percent
                net1 = psutil.net_io_counters()
                time.sleep(0.3)
                net2 = psutil.net_io_counters()
                up_bps = (net2.bytes_sent - net1.bytes_sent) * (1/0.3)
                down_bps = (net2.bytes_recv - net1.bytes_recv) * (1/0.3)
                if len(self.net_samples) > 0:
                    loss_pct = 100.0 * (1.0 - (sum(self.net_samples) / len(self.net_samples)))
                else:
                    loss_pct = None
                avg_lat = sum(self.net_latencies) / len(self.net_latencies) if self.net_latencies else None
                gpus = get_gpu_infos_nvml() if NVML_OK else []
                disk_busy_pct = self._sample_disk_busy_pct()
                procs_total, procs_running, procs_background, threads_total = self._sample_proc_thread_stats()

                payload = {
                    "cpu_pct": cpu_pct,
                    "cpu_freq_mhz": cpu_freq_mhz,
                    "cpu_temp": cpu_temp,
                    "mem_pct": mem_pct,
                    "mem_total": vm.total,
                    "mem_used": vm.used,
                    "disks": disks,
                    "disk_busy_pct": disk_busy_pct,
                    "net_up_bps": up_bps,
                    "net_down_bps": down_bps,
                    "net_loss_pct": loss_pct,
                    "net_avg_lat_ms": avg_lat,
                    "gpus": gpus,
                    "procs_total": procs_total,
                    "procs_running": procs_running,
                    "procs_background": procs_background,
                    "threads_total": threads_total,
                }
                self.data_ready.emit(payload)
            except Exception:
                pass
            time.sleep(0.7)

class MonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(900, 740)
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)
        container = QWidget()
        scroll.setWidget(container)
        self.layout = QVBoxLayout(container)
        self.system_group = self._make_system_group()
        self.cpu_group = self._make_cpu_group()
        self.mem_group = self._make_mem_group()
        self.disk_group = self._make_disks_group()
        self.net_group = self._make_net_group()
        self.gpu_group = self._make_gpu_group()

        for g in (self.system_group, self.cpu_group, self.mem_group, self.disk_group, self.net_group, self.gpu_group):
            self.layout.addWidget(g)

        self.layout.addStretch(1)

        self.worker = MetricsWorker(net_host="8.8.8.8", net_window=30)
        self.worker.data_ready.connect(self.update_ui)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(lambda: None)
        self.timer.start()

    def _make_labeled_bar(self, title: str):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        bar = ThresholdProgressBar(threshold=80) # 내용에 따라 threshold 조절하기 
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        lab = QLabel("—")
        lab.setAlignment(Qt.AlignRight)
        lay.addWidget(bar)
        lay.addWidget(lab)
        return box, bar, lab

    def _make_system_group(self):
        g = QGroupBox("System")
        grid = QGridLayout(g)
        self.proc_total_lab = QLabel("—")
        self.proc_run_lab = QLabel("—")
        self.proc_bg_lab = QLabel("—")
        self.thread_total_lab = QLabel("—")

        grid.addWidget(QLabel("Processes (total):"), 0, 0)
        grid.addWidget(self.proc_total_lab, 0, 1)
        grid.addWidget(QLabel("Running:"), 0, 2)
        grid.addWidget(self.proc_run_lab, 0, 3)

        grid.addWidget(QLabel("Background:"), 1, 0)
        grid.addWidget(self.proc_bg_lab, 1, 1)
        grid.addWidget(QLabel("Threads (total):"), 1, 2)
        grid.addWidget(self.thread_total_lab, 1, 3)
        return g

    def _make_cpu_group(self):
        g, bar, lab = self._make_labeled_bar("CPU Usage")
        self.cpu_bar, self.cpu_lab = bar, lab

        grid = QGridLayout()
        freq_lab_title = QLabel("Clock:")
        self.cpu_freq_lab = QLabel("—")
        temp_lab_title = QLabel("Temp:")
        self.cpu_temp_lab = QLabel("—")
        grid.addWidget(freq_lab_title, 0, 0)
        grid.addWidget(self.cpu_freq_lab, 0, 1)
        grid.addWidget(temp_lab_title, 1, 0)
        grid.addWidget(self.cpu_temp_lab, 1, 1)
        ((g.layout()).addLayout(grid))
        return g

    def _make_mem_group(self):
        g, bar, lab = self._make_labeled_bar("Memory Usage")
        self.mem_bar, self.mem_lab = bar, lab
        return g

    def _make_disks_group(self):
        g = QGroupBox("Disks")
        lay = QVBoxLayout(g)

        act_box, act_bar, act_lab = self._make_labeled_bar("Disk Activity (I/O Busy %)")
        self.disk_activity_bar = act_bar
        self.disk_activity_lab = act_lab
        lay.addWidget(act_box)

        self.disk_items = {}
        self.disk_layout = lay
        return g

    def _ensure_disk_item(self, key, title):
        if key in self.disk_items:
            return self.disk_items[key]
        box, bar, lab = self._make_labeled_bar(title)
        self.disk_layout.addWidget(box)
        self.disk_items[key] = {"box": box, "bar": bar, "lab": lab}
        return self.disk_items[key]
    
    def _make_net_group(self):
        g = QGroupBox("Network")
        lay = QGridLayout(g)
        self.net_up = QLabel("Up: —")
        self.net_down = QLabel("Down: —")
        self.net_lat = QLabel("Avg Latency: —")
        self.net_loss = QLabel("Loss: —")
        lay.addWidget(self.net_up, 0, 0)
        lay.addWidget(self.net_down, 0, 1)
        lay.addWidget(self.net_lat, 1, 0)
        lay.addWidget(self.net_loss, 1, 1)
        return g

    def _make_gpu_group(self):
        self.gpu_box = QGroupBox("GPU(s)")
        self.gpu_layout = QVBoxLayout(self.gpu_box)
        self.gpu_cards = []
        if not NVML_OK:
            info = QLabel("NVIDIA GPU 탐지 실패")
            self.gpu_layout.addWidget(info)
        return self.gpu_box

    def _ensure_gpu_cards(self, n):
        cur = len(self.gpu_cards)
        if n <= cur:
            return
        for i in range(cur, n):
            card = self._make_single_gpu_card(i)
            self.gpu_cards.append(card)
            self.gpu_layout.addWidget(card["group"])

    def _make_single_gpu_card(self, idx):
        group = QGroupBox(f"GPU #{idx}")
        lay = QVBoxLayout(group)

        name_lab = QLabel("Name: —")
        lay.addWidget(name_lab)

        util_box, util_bar, util_lab = self._make_labeled_bar("GPU Utilization")
        mem_box, mem_bar, mem_lab = self._make_labeled_bar("VRAM Usage")
        temp_box, temp_bar, temp_lab = self._make_labeled_bar("GPU Temperature")

        lay.addWidget(util_box)
        lay.addWidget(mem_box)
        lay.addWidget(temp_box)

        temp_bar.setRange(0, 120)  # °C scale

        return {
            "group": group,
            "name_lab": name_lab,
            "util_bar": util_bar,
            "util_lab": util_lab,
            "mem_bar": mem_bar,
            "mem_lab": mem_lab,
            "temp_bar": temp_bar,
            "temp_lab": temp_lab,
        }

    def update_ui(self, d: dict):
        self.proc_total_lab.setText(f"{d.get('procs_total', 0)}")
        self.proc_run_lab.setText(f"{d.get('procs_running', 0)}")
        self.proc_bg_lab.setText(f"{d.get('procs_background', 0)}")
        self.thread_total_lab.setText(f"{d.get('threads_total', 0)}")

        cpu_pct = int(round(d.get("cpu_pct", 0)))
        self.cpu_bar.setValue(cpu_pct)
        self.cpu_lab.setText(f"{cpu_pct}%")

        freq = d.get("cpu_freq_mhz")
        self.cpu_freq_lab.setText(f"{freq:.0f} MHz" if freq else "—")

        ctemp = d.get("cpu_temp")
        self.cpu_temp_lab.setText(f"{ctemp:.1f} °C" if ctemp is not None else "—")

        mem_pct = int(round(d.get("mem_pct", 0)))
        self.mem_bar.setValue(mem_pct)
        mem_used = d.get("mem_used", 0)
        mem_total = d.get("mem_total", 0)
        self.mem_lab.setText(f"{mem_pct}%  ({format_bytes(mem_used)} / {format_bytes(mem_total)})")

        disk_busy = d.get("disk_busy_pct")
        if disk_busy is not None:
            val = int(round(disk_busy))
            self.disk_activity_bar.setValue(val)
            self.disk_activity_lab.setText(f"{val}%")
        else:
            self.disk_activity_bar.setValue(0)
            self.disk_activity_lab.setText("—")

        for dsk in d.get("disks", []):
            key = dsk["mount"]
            title = f"{dsk['name']} ({dsk['mount']})" if dsk["name"] != dsk["mount"] else dsk["mount"]
            item = self._ensure_disk_item(key, title)
            pct = int(round(dsk["pct"])) if dsk.get("pct") is not None else 0
            item["bar"].setValue(pct)
            used = dsk.get("used", 0)
            total = dsk.get("total", 0)
            item["lab"].setText(f"{pct}%  ({format_bytes(used)} / {format_bytes(total)})")

        up = d.get("net_up_bps")
        down = d.get("net_down_bps")
        self.net_up.setText(f"Up: {format_bytes(up)}/s" if up is not None else "Up: —")
        self.net_down.setText(f"Down: {format_bytes(down)}/s" if down is not None else "Down: —")

        lat = d.get("net_avg_lat_ms")
        loss = d.get("net_loss_pct")
        self.net_lat.setText(f"Avg Latency: {lat:.1f} ms" if lat is not None else "Avg Latency: —")
        self.net_loss.setText(f"Loss: {loss:.1f} %" if loss is not None else "Loss: —")

        gpus = d.get("gpus", [])
        if gpus:
            self._ensure_gpu_cards(len(gpus))
            for i, gi in enumerate(gpus):
                card = self.gpu_cards[i]
                name = gi.get("name", f"GPU #{i}")
                card["name_lab"].setText(f"Name: {name}")

                util = gi.get("util")
                u = int(round(util)) if util is not None else 0
                card["util_bar"].setValue(u)
                card["util_lab"].setText(f"{u}%")

                used = gi.get("mem_used", 0)
                total = gi.get("mem_total", 0)
                mem_pct = int(round((used / total) * 100)) if total else 0
                card["mem_bar"].setValue(mem_pct)
                card["mem_lab"].setText(f"{mem_pct}%  ({format_bytes(used)} / {format_bytes(total)})")

                t = gi.get("temp")
                if t is not None:
                    card["temp_bar"].setValue(int(round(t)))
                    card["temp_lab"].setText(f"{t:.0f} °C")
                else:
                    card["temp_bar"].setValue(0)
                    card["temp_lab"].setText("—")

    def closeEvent(self, event):
        try:
            self.worker.stop()
        except Exception:
            pass
        return super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    w = MonitorWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
