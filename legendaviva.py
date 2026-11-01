# -*- coding: utf-8 -*-
"""
LegendaViva - legendas ao vivo Hebraico -> Português para Windows.

Captura o som que está tocando no PC (WASAPI loopback), reconhece a fala
em hebraico com faster-whisper (modelo ivrit-ai, treinado para hebraico),
traduz para português e mostra numa barra de legenda flutuante,
sempre por cima das outras janelas.

Controles da barra:
  - arrastar com o botão esquerdo ......... mover
  - arrastar o canto  ◢ ................... redimensionar
  - roda do mouse ......................... tamanho da letra
  - botão direito ......................... menu (pausar, configurações, sair)
"""

import os
import sys
import json
import time
import queue
import threading
import traceback
from collections import deque

APP = "LegendaViva"
CFG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP)
os.makedirs(CFG_DIR, exist_ok=True)
CFG_PATH = os.path.join(CFG_DIR, "config.json")
LOG_PATH = os.path.join(CFG_DIR, "log.txt")

# No .exe sem console, stdout/stderr são None e várias bibliotecas quebram ao
# tentar escrever neles. Redireciona tudo para um arquivo de log.
if sys.stdout is None or sys.stderr is None:
    _log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import ttk, messagebox  # noqa: E402


def log(*args):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(a) for a in args)
    try:
        print(msg, flush=True)
    except Exception:
        pass


# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------

MODELS = [
    ("ivrit-ai/whisper-large-v3-turbo-ct2", "Hebraico ivrit-ai turbo (melhor qualidade - recomendado)"),
    ("large-v3-turbo", "Whisper large-v3-turbo (genérico)"),
    ("medium", "Whisper medium (mais leve)"),
    ("small", "Whisper small (PC fraco, qualidade menor)"),
]

DEFAULTS = {
    "model": MODELS[0][0],
    "device": "auto",          # auto / cuda / cpu
    "loopback_device": "",     # "" = saída de som padrão do Windows
    "show_original": False,
    "font_size": 26,
    "max_lines": 2,
    "chunk_seconds": 5.0,      # tamanho alvo de cada trecho de áudio
    "opacity": 0.82,
    "clear_after": 12,         # some com a legenda após N s sem fala
    "geometry": "",
}


def load_cfg():
    cfg = dict(DEFAULTS)
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def save_cfg(cfg):
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("erro salvando config:", e)


# --------------------------------------------------------------------------
# Captura do som do PC (WASAPI loopback)
# --------------------------------------------------------------------------

def list_loopback_devices():
    """Retorna lista de nomes de dispositivos loopback (saídas de som)."""
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        return [d["name"] for d in p.get_loopback_device_info_generator()]
    finally:
        p.terminate()


def find_loopback(p, wanted_name):
    import pyaudiowpatch as pyaudio
    loopbacks = list(p.get_loopback_device_info_generator())
    if not loopbacks:
        raise RuntimeError("Nenhum dispositivo de saída de som (loopback WASAPI) encontrado.")
    if wanted_name:
        for d in loopbacks:
            if d["name"] == wanted_name:
                return d
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return speakers
    for d in loopbacks:
        if speakers["name"] in d["name"]:
            return d
    return loopbacks[0]


class AudioCapture(threading.Thread):
    def __init__(self, out_q, device_name, on_status):
        super().__init__(daemon=True)
        self.out_q = out_q
        self.device_name = device_name
        self.on_status = on_status
        self.stop_evt = threading.Event()
        self.rate = None
        self.ready = threading.Event()
        self.error = None

    def run(self):
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        stream = None
        try:
            dev = find_loopback(p, self.device_name)
            self.rate = int(dev["defaultSampleRate"])
            ch = max(1, int(dev["maxInputChannels"]))
            log("capturando de:", dev["name"], self.rate, "Hz", ch, "canais")

            def cb(in_data, frame_count, time_info, status):
                a = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    a = a[: (len(a) // ch) * ch].reshape(-1, ch).mean(axis=1)
                self.out_q.put(a)
                return (None, pyaudio.paContinue)

            stream = p.open(format=pyaudio.paInt16, channels=ch, rate=self.rate,
                            input=True, input_device_index=dev["index"],
                            frames_per_buffer=int(self.rate * 0.05),
                            stream_callback=cb)
            stream.start_stream()
            self.ready.set()
            while not self.stop_evt.is_set():
                time.sleep(0.1)
        except Exception as e:
            self.error = e
            log("erro na captura:", traceback.format_exc())
            self.on_status(f"Erro no áudio: {e}")
            self.ready.set()
        finally:
            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            p.terminate()

    def stop(self):
        self.stop_evt.set()


# --------------------------------------------------------------------------
# Divisão do áudio em trechos (cortando nas pausas da fala)
# --------------------------------------------------------------------------

FRAME_S = 0.02  # 20 ms


def frame_energy(buf, rate):
    frame = int(rate * FRAME_S)
    n = len(buf) // frame
    if n == 0:
        return np.zeros(0), frame
    x = buf[: n * frame].reshape(n, frame)
    return np.sqrt(np.mean(x * x, axis=1)) + 1e-9, frame


def decide_cut(buf, rate, target_s, flush=False):
    """
    Decide onde cortar o buffer. Retorna índice em amostras, ou None.
    Prioriza cortar numa pausa natural da fala para não partir palavras.
    """
    e, frame = frame_energy(buf, rate)
    n = len(e)
    if n == 0:
        return None
    dur = n * FRAME_S
    min_s = 1.2
    max_s = target_s + 3.0

    if flush and dur >= 0.6:
        return n * frame

    # Só silêncio/ruído baixíssimo: descarta tudo
    if dur >= 1.0 and e.max() < 0.003:
        return n * frame

    smooth = np.convolve(e, np.ones(5) / 5, mode="same")
    thr = max(0.002, 0.35 * float(np.percentile(e, 50)))
    quiet = smooth < thr

    # 1) Pausa de >= 400 ms no fim depois de fala suficiente -> corta já
    pause_frames = int(0.4 / FRAME_S)
    if dur >= min_s and n > pause_frames and quiet[-pause_frames:].all():
        return (n - pause_frames // 2) * frame

    # 2) Atingiu o tamanho alvo: corta na pausa mais silenciosa recente
    if dur >= target_s:
        lo = int((target_s * 0.6) / FRAME_S)
        seg = smooth[lo:]
        if len(seg) and seg.min() < thr:
            return (lo + int(np.argmin(seg))) * frame

    # 3) Passou do máximo: força o corte no ponto mais baixo
    if dur >= max_s:
        lo = int(min_s / FRAME_S)
        return (lo + int(np.argmin(smooth[lo:]))) * frame

    return None


def to_16k(x, rate):
    """Converte para 16 kHz mono com um filtro passa-baixa simples."""
    if rate == 16000:
        return x.astype(np.float32)
    k = int(round(rate / 16000))
    if k > 1:
        x = np.convolve(x, np.ones(k) / k, mode="same")
    n_out = int(len(x) * 16000 / rate)
    t_in = np.arange(len(x)) / rate
    t_out = np.arange(n_out) / 16000
    return np.interp(t_out, t_in, x).astype(np.float32)


# --------------------------------------------------------------------------
# Reconhecimento (Whisper) + tradução
# --------------------------------------------------------------------------

# Frases que o Whisper costuma "inventar" em silêncio/música
HALLUCINATIONS = [
    "תודה רבה", "תודה שצפיתם", "תודה על הצפייה", "כתוביות", "תרגום",
    "תורגם", "הירשמו לערוץ", "צפייה מהנה", "amara.org",
]


def is_hallucination(text):
    t = text.strip(" .!?,-\"'")
    if len(t) < 2:
        return True
    for h in HALLUCINATIONS:
        if h in t and len(t) <= len(h) + 12:
            return True
    return False


class Translator:
    def __init__(self):
        from deep_translator import GoogleTranslator
        self.tr = GoogleTranslator(source="iw", target="pt")

    def __call__(self, text):
        for attempt in range(2):
            try:
                out = self.tr.translate(text)
                if out:
                    return out
            except Exception as e:
                log("erro tradução:", e)
                time.sleep(0.5)
        return None


def load_model(name, device_pref, on_status):
    from faster_whisper import WhisperModel
    threads = max(1, (os.cpu_count() or 4) - 1)

    def try_load(dev):
        if dev == "cuda":
            m = WhisperModel(name, device="cuda", compute_type="float16")
        else:
            m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
        # aquecimento: força carregar as DLLs (cuBLAS/cuDNN) agora
        list(m.transcribe(np.zeros(16000, dtype=np.float32), language="he")[0])
        return m

    order = ["cuda", "cpu"] if device_pref == "auto" else [device_pref]
    if device_pref == "cuda":
        order = ["cuda", "cpu"]
    last = None
    for dev in order:
        try:
            on_status(f"Carregando modelo ({'placa de vídeo' if dev == 'cuda' else 'processador'})… "
                      "na 1ª vez ele é baixado (~1,6 GB), aguarde.")
            m = try_load(dev)
            log("modelo carregado em", dev)
            return m, dev
        except Exception as e:
            last = e
            log(f"falha ao carregar em {dev}:", traceback.format_exc())
    raise RuntimeError(f"Não foi possível carregar o modelo: {last}")


class Engine:
    """Junta captura + segmentação + Whisper + tradução numa thread."""

    def __init__(self, cfg, ui_q):
        self.cfg = cfg
        self.ui_q = ui_q
        self.audio_q = queue.Queue()
        self.stop_evt = threading.Event()
        self.paused = threading.Event()
        self.capture = None
        self.worker = threading.Thread(target=self._run, daemon=True)

    def status(self, msg):
        self.ui_q.put(("status", msg))

    def start(self):
        self.worker.start()

    def stop(self):
        self.stop_evt.set()
        if self.capture:
            self.capture.stop()

    def set_paused(self, p):
        if p:
            self.paused.set()
        else:
            self.paused.clear()

    def _run(self):
        try:
            model, dev = load_model(self.cfg["model"], self.cfg["device"], self.status)
        except Exception as e:
            self.status(f"Erro: {e}")
            self.ui_q.put(("error", str(e)))
            return
        if self.stop_evt.is_set():
            return

        try:
            translator = Translator()
        except Exception as e:
            self.status(f"Erro no tradutor: {e}")
            return

        self.capture = AudioCapture(self.audio_q, self.cfg["loopback_device"], self.status)
        self.capture.start()
        self.capture.ready.wait(10)
        if self.capture.error or not self.capture.rate:
            return
        rate = self.capture.rate
        dev_txt = "GPU" if dev == "cuda" else "CPU"
        self.status(f"● Ouvindo ({dev_txt})")

        buf = np.zeros(0, dtype=np.float32)
        last_audio = time.time()
        prev_text = ""
        target = float(self.cfg["chunk_seconds"])

        while not self.stop_evt.is_set():
            got = False
            try:
                blocks = [self.audio_q.get(timeout=0.2)]
                got = True
                while True:
                    blocks.append(self.audio_q.get_nowait())
            except queue.Empty:
                pass
            if got:
                last_audio = time.time()
                if self.paused.is_set():
                    buf = np.zeros(0, dtype=np.float32)
                    continue
                buf = np.concatenate([buf] + blocks)

            # O loopback não manda nada quando o PC está mudo:
            # se parou de chegar áudio, processa o que sobrou.
            flush = (not got) and (time.time() - last_audio > 0.6) and len(buf) > 0

            # PC lento: não deixa o atraso crescer sem limite
            max_keep = int(rate * (target + 3.0) * 3)
            if len(buf) > max_keep:
                buf = buf[-int(rate * (target + 3.0)):]
                self.status(f"● Ouvindo ({dev_txt}) - atrasado, use um modelo mais leve")

            cut = decide_cut(buf, rate, target, flush=flush)
            if cut is None:
                if flush and len(buf) < int(rate * 0.6):
                    buf = np.zeros(0, dtype=np.float32)
                continue
            chunk, buf = buf[:cut], buf[cut:]

            if float(np.sqrt(np.mean(chunk * chunk))) < 0.002:
                continue  # silêncio

            audio16 = to_16k(chunk, rate)
            t0 = time.time()
            try:
                segments, _ = model.transcribe(
                    audio16,
                    language="he",
                    task="transcribe",
                    beam_size=5 if dev == "cuda" else 2,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                    condition_on_previous_text=False,
                    initial_prompt=prev_text[-200:] or None,
                    without_timestamps=True,
                    no_speech_threshold=0.6,
                )
                parts = []
                for s in segments:
                    if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                        continue
                    if s.compression_ratio > 2.4:
                        continue
                    parts.append(s.text.strip())
                he = " ".join(p for p in parts if p).strip()
            except Exception as e:
                log("erro no Whisper:", traceback.format_exc())
                self.status(f"Erro no reconhecimento: {e}")
                continue

            if not he or is_hallucination(he):
                continue
            prev_text = he
            pt = translator(he)
            log(f"[{time.time() - t0:.1f}s] HE: {he} | PT: {pt}")
            self.ui_q.put(("text", he, pt))


# --------------------------------------------------------------------------
# Interface: barra de legenda flutuante
# --------------------------------------------------------------------------

class Overlay(tk.Tk):
    BG = "#000000"

    def __init__(self):
        super().__init__()
        self.cfg = load_cfg()
        self.ui_q = queue.Queue()
        self.engine = None
        self.paused = False
        self.lines = deque(maxlen=int(self.cfg["max_lines"]))
        self.last_text_time = 0

        self.title(APP)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", float(self.cfg["opacity"]))
        self.configure(bg=self.BG)

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        geo = self.cfg.get("geometry") or f"{int(sw * 0.7)}x150+{int(sw * 0.15)}+{int(sh * 0.78)}"
        self.geometry(geo)

        self.status_lbl = tk.Label(self, text="Iniciando…", bg=self.BG, fg="#8a8a8a",
                                   font=("Segoe UI", 9), anchor="w")
        self.status_lbl.pack(fill="x", padx=8, pady=(3, 0))

        self.orig_lbl = tk.Label(self, text="", bg=self.BG, fg="#9fc5e8",
                                 font=("Segoe UI", 13), justify="center")
        self.text_lbl = tk.Label(self, text="", bg=self.BG, fg="#ffffff",
                                 font=("Segoe UI", int(self.cfg["font_size"]), "bold"),
                                 justify="center")
        self._pack_labels()

        self.grip = tk.Label(self, text="◢", bg=self.BG, fg="#555555",
                             font=("Segoe UI", 11), cursor="size_nw_se")
        self.grip.place(relx=1.0, rely=1.0, anchor="se")

        # Arrastar / redimensionar / fonte / menu
        for w in (self, self.status_lbl, self.text_lbl, self.orig_lbl):
            w.bind("<ButtonPress-1>", self._start_move)
            w.bind("<B1-Motion>", self._on_move)
            w.bind("<ButtonRelease-1>", lambda e: self._remember_geometry())
            w.bind("<Button-3>", self._menu)
            w.bind("<MouseWheel>", self._wheel)
        self.grip.bind("<ButtonPress-1>", self._start_resize)
        self.grip.bind("<B1-Motion>", self._on_resize)
        self.grip.bind("<ButtonRelease-1>", lambda e: self._remember_geometry())
        self.bind("<Configure>", lambda e: self._update_wrap())

        self.menu = tk.Menu(self, tearoff=0)
        self.show_orig_var = tk.BooleanVar(value=bool(self.cfg["show_original"]))
        self.menu.add_command(label="Pausar", command=self.toggle_pause)
        self.menu.add_checkbutton(label="Mostrar texto em hebraico", variable=self.show_orig_var,
                                  command=self._toggle_orig)
        self.menu.add_command(label="Limpar legenda", command=self.clear)
        self.menu.add_separator()
        self.menu.add_command(label="Configurações…", command=self.open_settings)
        self.menu.add_command(label="Abrir pasta do log", command=lambda: os.startfile(CFG_DIR))
        self.menu.add_separator()
        self.menu.add_command(label="Sair", command=self.quit_app)

        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.start_engine()
        self.after(100, self._poll)
        self.after(2000, self._keep_on_top)

    # ---- layout --------------------------------------------------------
    def _pack_labels(self):
        self.orig_lbl.pack_forget()
        self.text_lbl.pack_forget()
        if self.cfg["show_original"]:
            self.orig_lbl.pack(fill="x", padx=14)
        self.text_lbl.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        self._update_wrap()

    def _update_wrap(self):
        w = max(200, self.winfo_width() - 40)
        self.text_lbl.configure(wraplength=w)
        self.orig_lbl.configure(wraplength=w)

    # ---- motor ---------------------------------------------------------
    def start_engine(self):
        if self.engine:
            self.engine.stop()
        self.engine = Engine(dict(self.cfg), self.ui_q)
        self.engine.set_paused(self.paused)
        self.engine.start()

    def _poll(self):
        try:
            while True:
                item = self.ui_q.get_nowait()
                kind = item[0]
                if kind == "status":
                    self.status_lbl.configure(text=item[1])
                elif kind == "text":
                    _, he, pt = item
                    self.lines.append((he, pt or f"(sem tradução) {he}"))
                    self.last_text_time = time.time()
                    self._render()
                elif kind == "error":
                    self.text_lbl.configure(text="Não consegui iniciar. Botão direito → Configurações\n"
                                                 "(tente CPU ou um modelo menor).")
        except queue.Empty:
            pass
        clear_after = float(self.cfg.get("clear_after", 0) or 0)
        if clear_after and self.lines and time.time() - self.last_text_time > clear_after:
            self.clear()
        self.after(100, self._poll)

    def _render(self):
        self.text_lbl.configure(text="\n".join(pt for _, pt in self.lines))
        self.orig_lbl.configure(text="\n".join(he for he, _ in self.lines))

    def clear(self):
        self.lines.clear()
        self._render()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.engine:
            self.engine.set_paused(self.paused)
        self.menu.entryconfigure(0, label="Retomar" if self.paused else "Pausar")
        self.status_lbl.configure(text="⏸ Pausado" if self.paused else "● Ouvindo")

    def _toggle_orig(self):
        self.cfg["show_original"] = bool(self.show_orig_var.get())
        save_cfg(self.cfg)
        self._pack_labels()

    def _keep_on_top(self):
        try:
            self.attributes("-topmost", True)
            self.lift()
        except Exception:
            pass
        self.after(2000, self._keep_on_top)

    # ---- mouse ---------------------------------------------------------
    def _start_move(self, e):
        self._mx, self._my = e.x_root - self.winfo_x(), e.y_root - self.winfo_y()

    def _on_move(self, e):
        self.geometry(f"+{e.x_root - self._mx}+{e.y_root - self._my}")

    def _start_resize(self, e):
        self._rx, self._ry = e.x_root, e.y_root
        self._rw, self._rh = self.winfo_width(), self.winfo_height()

    def _on_resize(self, e):
        w = max(300, self._rw + e.x_root - self._rx)
        h = max(70, self._rh + e.y_root - self._ry)
        self.geometry(f"{w}x{h}")

    def _wheel(self, e):
        size = int(self.cfg["font_size"]) + (2 if e.delta > 0 else -2)
        size = min(72, max(12, size))
        self.cfg["font_size"] = size
        self.text_lbl.configure(font=("Segoe UI", size, "bold"))
        save_cfg(self.cfg)

    def _menu(self, e):
        self.menu.tk_popup(e.x_root, e.y_root)

    def _remember_geometry(self):
        self.cfg["geometry"] = self.geometry()
        save_cfg(self.cfg)

    # ---- configurações -------------------------------------------------
    def open_settings(self):
        SettingsWindow(self)

    def apply_settings(self, new_cfg):
        restart = any(new_cfg[k] != self.cfg[k] for k in
                      ("model", "device", "loopback_device", "chunk_seconds"))
        self.cfg.update(new_cfg)
        save_cfg(self.cfg)
        self.attributes("-alpha", float(self.cfg["opacity"]))
        self.lines = deque(self.lines, maxlen=int(self.cfg["max_lines"]))
        self.show_orig_var.set(bool(self.cfg["show_original"]))
        self._pack_labels()
        self._render()
        if restart:
            self.clear()
            self.start_engine()

    def quit_app(self):
        self._remember_geometry()
        if self.engine:
            self.engine.stop()
        self.destroy()
        os._exit(0)


class SettingsWindow(tk.Toplevel):
    def __init__(self, master: Overlay):
        super().__init__(master)
        self.master_ov = master
        cfg = master.cfg
        self.title("LegendaViva - Configurações")
        self.attributes("-topmost", True)
        self.resizable(False, False)
        pad = dict(padx=10, pady=5, sticky="w")

        ttk.Label(self, text="Modelo de reconhecimento:").grid(row=0, column=0, **pad)
        self.model_var = tk.StringVar(value=cfg["model"])
        names = [m for m, _ in MODELS]
        cb = ttk.Combobox(self, textvariable=self.model_var, values=names, width=42)
        cb.grid(row=0, column=1, **pad)
        desc = "\n".join(f"• {m}: {d}" for m, d in MODELS)
        ttk.Label(self, text=desc, foreground="#555").grid(row=1, column=0, columnspan=2, padx=10, sticky="w")

        ttk.Label(self, text="Processar em:").grid(row=2, column=0, **pad)
        self.dev_var = tk.StringVar(value=cfg["device"])
        ttk.Combobox(self, textvariable=self.dev_var, values=["auto", "cuda", "cpu"],
                     state="readonly", width=10).grid(row=2, column=1, **pad)

        ttk.Label(self, text="Saída de som a ouvir:").grid(row=3, column=0, **pad)
        try:
            devs = list_loopback_devices()
        except Exception as e:
            devs = []
            log("erro listando dispositivos:", e)
        self.lb_var = tk.StringVar(value=cfg["loopback_device"] or "(padrão do Windows)")
        ttk.Combobox(self, textvariable=self.lb_var, values=["(padrão do Windows)"] + devs,
                     state="readonly", width=42).grid(row=3, column=1, **pad)

        ttk.Label(self, text="Tamanho do trecho (s):").grid(row=4, column=0, **pad)
        self.chunk_var = tk.DoubleVar(value=float(cfg["chunk_seconds"]))
        ttk.Spinbox(self, from_=3, to=10, increment=0.5, textvariable=self.chunk_var,
                    width=6).grid(row=4, column=1, **pad)
        ttk.Label(self, text="Menor = legenda mais rápida; maior = tradução com mais contexto.",
                  foreground="#555").grid(row=5, column=0, columnspan=2, padx=10, sticky="w")

        ttk.Label(self, text="Linhas na tela:").grid(row=6, column=0, **pad)
        self.lines_var = tk.IntVar(value=int(cfg["max_lines"]))
        ttk.Spinbox(self, from_=1, to=5, textvariable=self.lines_var, width=6).grid(row=6, column=1, **pad)

        ttk.Label(self, text="Opacidade do fundo:").grid(row=7, column=0, **pad)
        self.op_var = tk.DoubleVar(value=float(cfg["opacity"]))
        ttk.Scale(self, from_=0.3, to=1.0, variable=self.op_var, length=200).grid(row=7, column=1, **pad)

        ttk.Label(self, text="Apagar legenda após (s, 0=nunca):").grid(row=8, column=0, **pad)
        self.clear_var = tk.IntVar(value=int(cfg.get("clear_after", 12)))
        ttk.Spinbox(self, from_=0, to=60, textvariable=self.clear_var, width=6).grid(row=8, column=1, **pad)

        self.orig_var = tk.BooleanVar(value=bool(cfg["show_original"]))
        ttk.Checkbutton(self, text="Mostrar também o texto em hebraico",
                        variable=self.orig_var).grid(row=9, column=0, columnspan=2, **pad)

        btns = ttk.Frame(self)
        btns.grid(row=10, column=0, columnspan=2, pady=10)
        ttk.Button(btns, text="Salvar", command=self.save).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancelar", command=self.destroy).pack(side="left", padx=5)

    def save(self):
        lb = self.lb_var.get()
        try:
            new = {
                "model": self.model_var.get().strip() or DEFAULTS["model"],
                "device": self.dev_var.get(),
                "loopback_device": "" if lb.startswith("(padrão") else lb,
                "chunk_seconds": float(self.chunk_var.get()),
                "max_lines": int(self.lines_var.get()),
                "opacity": round(float(self.op_var.get()), 2),
                "clear_after": int(self.clear_var.get()),
                "show_original": bool(self.orig_var.get()),
            }
        except Exception as e:
            messagebox.showerror(APP, f"Valor inválido: {e}", parent=self)
            return
        self.master_ov.apply_settings(new)
        self.destroy()


def main():
    log("==== iniciando", APP)
    try:
        app = Overlay()
        app.mainloop()
    except Exception:
        log("erro fatal:", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
