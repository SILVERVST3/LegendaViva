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
    "translator": "auto",      # auto / online / offline
    "engine": "auto",          # auto (Vulkan se existir) / vulkan / faster-whisper
    "fast_encoder": True,      # whisper.cpp: encoder encurtado COM verificação automática
    "partials": True,          # legenda provisória enquanto a frase é falada
    "partial_step": 0.5,       # s de áudio novo entre passadas provisórias
    "ggml_model_path": "",     # opcional: caminho de um .bin GGML já baixado
    "loopback_device": "",     # "" = saída de som padrão do Windows
    "show_original": False,
    "font_size": 26,
    "max_lines": 2,
    "chunk_seconds": 6.0,      # tamanho máximo alvo de cada frase (com legenda provisória)
    "opacity": 0.82,
    "clear_after": 12,         # some com a legenda após N s sem fala
    "geometry": "",
}


CFG_VERSION = 3


def load_cfg():
    cfg = dict(DEFAULTS)
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    ver = int(cfg.get("cfg_version", 1))
    if ver < 2:
        # v2: legenda provisória -> frases maiores dão tradução melhor sem aumentar o atraso
        cfg["chunk_seconds"] = DEFAULTS["chunk_seconds"]
        cfg["partials"] = True
    if ver < 3:
        # v3: encoder encurtado agora tem verificação automática -> ligado por padrão
        cfg["fast_encoder"] = True
    cfg["cfg_version"] = CFG_VERSION
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
    min_s = 1.0
    max_s = target_s + 3.0

    if flush and dur >= 0.4:
        return n * frame

    # Só silêncio/ruído baixíssimo: descarta tudo
    if dur >= 1.0 and e.max() < 0.0008:
        return n * frame

    smooth = np.convolve(e, np.ones(5) / 5, mode="same")
    thr = max(0.0005, 0.35 * float(np.percentile(e, 50)))
    quiet = smooth < thr

    # 1) Pausa de >= 300 ms no fim depois de fala suficiente -> corta já
    pause_frames = int(0.3 / FRAME_S)
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


def decide_cut_eager(buf, rate, min_s):
    """
    Corte "assim que a placa fica livre": se já há pelo menos min_s segundos de
    fala, procura o ponto mais silencioso (entre palavras) no fim do buffer e
    corta ali, sem esperar uma pausa longa. Evita a placa ficar parada esperando.
    """
    e, frame = frame_energy(buf, rate)
    n = len(e)
    if n * FRAME_S < min_s:
        return None
    smooth = np.convolve(e, np.ones(5) / 5, mode="same")
    lo = int(min_s * 0.7 / FRAME_S)
    hi = n - int(0.08 / FRAME_S)
    if hi <= lo:
        return None
    idx = lo + int(np.argmin(smooth[lo:hi]))
    thr = max(0.0005, 0.45 * float(np.median(e)))
    if smooth[idx] < thr:
        return idx * frame
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


_BIDI = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"), None)


def clean_text(text):
    """Remove marcas invisíveis de direção (RTL) e espaços extras."""
    return " ".join(text.translate(_BIDI).split())


def is_hallucination(text):
    t = text.strip(" .!?,-\"'")
    if len(t) < 2:
        return True
    for h in HALLUCINATIONS:
        if h in t and len(t) <= len(h) + 12:
            return True
    return False


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


_SESSION = None


def http():
    """Sessão HTTP com keep-alive: evita refazer a conexão TLS (~0,2-0,4 s) a cada frase."""
    global _SESSION
    if _SESSION is None:
        import requests
        _SESSION = requests.Session()
        _SESSION.headers["User-Agent"] = UA
    return _SESSION


def tr_google(text):
    r = http().get("https://translate.googleapis.com/translate_a/single",
                     params={"client": "gtx", "sl": "iw", "tl": "pt", "dt": "t", "q": text},
                     timeout=5)
    r.raise_for_status()
    data = r.json()
    return "".join(seg[0] for seg in data[0] if seg and seg[0]).strip()


_SESSION2 = None
_POOL = None
_ALT_FAILS = 0


def _http2():
    global _SESSION2
    if _SESSION2 is None:
        import requests
        _SESSION2 = requests.Session()
        _SESSION2.headers["User-Agent"] = UA
    return _SESSION2


def tr_google_alt(text):
    """Segundo endereço do Google Tradutor (o usado por extensões do Chrome)."""
    r = _http2().get("https://clients5.google.com/translate_a/t",
                     params={"client": "dict-chrome-ex", "sl": "iw", "tl": "pt", "q": text},
                     timeout=5)
    r.raise_for_status()
    j = r.json()
    # formatos possíveis: ["tradução"], [["tradução","iw"]], {"sentences":[{"trans":..}]}
    if isinstance(j, dict):
        out = "".join(x.get("trans", "") for x in j.get("sentences", []))
    elif isinstance(j, list) and j:
        first = j[0]
        out = first if isinstance(first, str) else (first[0] if isinstance(first, list) and first else "")
    else:
        out = ""
    out = (out or "").strip()
    if not out:
        raise RuntimeError("resposta vazia (Google alt)")
    return out


def tr_google_hedged(text, hedge_after=0.7):
    """
    Pedido "com reserva": manda ao endereço principal; se não responder em
    hedge_after segundos, manda também ao endereço alternativo e usa a
    resposta que chegar primeiro. Corta os atrasos ocasionais de 2-3 s.
    """
    global _POOL, _ALT_FAILS
    import concurrent.futures as cf
    if _POOL is None:
        _POOL = cf.ThreadPoolExecutor(max_workers=3)
    f1 = _POOL.submit(tr_google, text)
    try:
        return f1.result(timeout=hedge_after)
    except cf.TimeoutError:
        pass
    except Exception:
        if _ALT_FAILS >= 5:
            raise
    if _ALT_FAILS >= 5:          # endereço alternativo não funciona nesta rede
        return f1.result(timeout=6)
    f2 = _POOL.submit(tr_google_alt, text)
    pending = {f1, f2}
    last_err = None
    while pending:
        done, pending = cf.wait(pending, timeout=6, return_when=cf.FIRST_COMPLETED)
        if not done:
            break
        for f in done:
            try:
                out = f.result()
                if f is f2:
                    _ALT_FAILS = 0
                return out
            except Exception as e:
                last_err = e
                if f is f2:
                    _ALT_FAILS += 1
                    log("Google alt falhou:", e)
    raise last_err or RuntimeError("Google não respondeu")


def tr_mymemory(text):
    r = http().get("https://api.mymemory.translated.net/get",
                   params={"q": text, "langpair": "he|pt-BR"}, timeout=6)
    r.raise_for_status()
    j = r.json()
    if int(j.get("responseStatus", 0)) != 200:
        raise RuntimeError(j.get("responseDetails", "erro MyMemory"))
    out = (j.get("responseData") or {}).get("translatedText", "").strip()
    if not out or "MYMEMORY WARNING" in out.upper():
        raise RuntimeError("limite diário do MyMemory")
    return out


class OfflineNLLB:
    """Tradução offline hebraico -> português com NLLB-200 (Meta) via CTranslate2."""
    MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    SPM_REPO = "facebook/nllb-200-distilled-600M"

    def __init__(self, on_status):
        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download, hf_hub_download
        on_status("Baixando/carregando tradutor offline (~600 MB na 1ª vez)…")
        path = snapshot_download(self.MODEL_REPO)
        spm_path = os.path.join(path, "sentencepiece.bpe.model")
        if not os.path.exists(spm_path):
            spm_path = hf_hub_download(self.SPM_REPO, "sentencepiece.bpe.model")
        self.sp = spm.SentencePieceProcessor(model_file=spm_path)
        self.tr = ctranslate2.Translator(path, device="cpu", compute_type="int8",
                                         inter_threads=1, intra_threads=2)
        log("tradutor offline carregado")

    def __call__(self, text):
        src = ["heb_Hebr"] + self.sp.encode(text, out_type=str) + ["</s>"]
        res = self.tr.translate_batch([src], target_prefix=[["por_Latn"]],
                                      beam_size=2, max_decoding_length=256)
        toks = res[0].hypotheses[0]
        if toks and toks[0] == "por_Latn":
            toks = toks[1:]
        return self.sp.decode(toks).strip()


class Translator:
    """
    Modos:
      auto    -> Google; se falhar, MyMemory; se falhar, offline (NLLB)
      online  -> só Google/MyMemory
      offline -> só NLLB (não precisa de internet)
    Serviço que falha fica "de castigo" por um tempo e o próximo é usado.
    """

    def __init__(self, mode, on_status):
        self.mode = mode
        self.on_status = on_status
        self.offline = None
        self.offline_failed = False
        self.cooldown = {}
        self.last_used = None
        self.cache = {}
        if mode == "offline":
            self._get_offline()

    def _get_offline(self):
        if self.offline is None and not self.offline_failed:
            try:
                self.offline = OfflineNLLB(self.on_status)
            except Exception:
                self.offline_failed = True
                log("falha no tradutor offline:", traceback.format_exc())
        return self.offline

    def __call__(self, text, partial=False):
        """partial=True: legenda provisória. Usa só serviços rápidos e sem cota
        (Google ou offline já carregado), para não gastar o limite do MyMemory."""
        if text in self.cache:
            self.last_used = "cache"
            return self.cache[text]
        out = self._translate(text, partial)
        if out:
            self.cache[text] = out
            if len(self.cache) > 300:
                self.cache.pop(next(iter(self.cache)))
        return out

    def _translate(self, text, partial):
        providers = []
        if self.mode in ("auto", "online"):
            providers.append(("Google", tr_google_hedged, 120))
            if not partial:
                providers.append(("MyMemory", tr_mymemory, 600))
        if self.mode == "offline" or (self.mode == "auto" and (not partial or self.offline)):
            providers.append(("Offline", None, 0))
        now = time.time()
        for name, fn, penalty in providers:
            if self.cooldown.get(name, 0) > now:
                continue
            try:
                if fn is None:
                    off = self._get_offline()
                    if off is None:
                        continue
                    out = off(text)
                else:
                    out = fn(text)
                if out:
                    self.last_used = name
                    return out
            except Exception as e:
                log(f"erro tradução ({name}):", e)
                self.cooldown[name] = time.time() + penalty
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

    order = ["cpu"]
    if device_pref in ("auto", "cuda"):
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                order = ["cuda", "cpu"]
        except Exception:
            pass
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


# ---- Motor 1: faster-whisper (CPU ou NVIDIA) ------------------------------

class FasterWhisperBackend:
    def __init__(self, cfg, on_status):
        self.model, self.dev = load_model(cfg["model"], cfg["device"], on_status)
        self.label = "GPU NVIDIA" if self.dev == "cuda" else "CPU"
        self.slow = self.dev != "cuda"
        self.final_beam = 1 if self.slow else 3

    def transcribe(self, audio16, prompt, final=True):
        segments, _ = self.model.transcribe(
            audio16, language="he", task="transcribe",
            beam_size=(self.final_beam if final else 1),
            temperature=[0.0, 0.2, 0.4] if final else [0.0],
            vad_filter=final, vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=False, initial_prompt=prompt or None,
            without_timestamps=True, no_speech_threshold=0.6,
        )
        out = []
        for s in segments:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            if s.compression_ratio > 2.4:
                continue
            out.append(s.text.strip())
        return " ".join(t for t in out if t).strip()

    def close(self):
        pass


# ---- Motor 2: whisper.cpp com Vulkan (placas AMD / Intel / NVIDIA) --------

def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def whisper_server_exe():
    p = os.path.join(app_dir(), "whispercpp", "whisper-server.exe")
    return p if os.path.exists(p) else None


# Modelos de hebraico (ivrit.ai) no formato do whisper.cpp, em ordem de preferência
GGML_SOURCES = [
    ("ivrit-ai/whisper-large-v3-turbo-ggml", "ggml-model.bin"),                       # oficial, f16, 1,5 GB
    ("RozzaCreat/whisper-large-v3-turbo-he-ggml", "ivrit-ai-whisper-large-v3-turbo-q4_k.bin"),  # 450 MB
]


def get_ggml_model(cfg, on_status):
    local = (cfg.get("ggml_model_path") or "").strip()
    if local and os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    last = None
    for repo, fname in GGML_SOURCES:
        try:
            on_status("Baixando modelo de hebraico para a placa de vídeo (1ª vez, ~1,5 GB)…")
            return hf_hub_download(repo, fname)
        except Exception as e:
            last = e
            log(f"falha baixando {repo}/{fname}:", e)
    raise RuntimeError(f"Não consegui baixar o modelo GGML: {last}")


def wav_bytes(audio16):
    import io
    import wave
    pcm = (np.clip(audio16, -1, 1) * 32767).astype("<i2").tobytes()
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return bio.getvalue()


class WhisperCppBackend:
    """Roda o whisper-server.exe (compilado com Vulkan) e conversa com ele por HTTP local."""

    def __init__(self, cfg, on_status, on_save=None):
        import requests
        self.requests = requests
        self.session = requests.Session()
        self.exe = whisper_server_exe()
        if not self.exe:
            raise RuntimeError("whisper-server.exe não encontrado")
        self.model = get_ggml_model(cfg, on_status)
        self.cfg = cfg
        self.proc = None
        self.audio_ctx = bool(cfg.get("fast_encoder", True))
        self.final_beam = 1
        self.lp_hist = deque(maxlen=40)
        self.fast_n = 0
        self.fast_redo = 0

        # Calibração (só 1 vez por placa/modelo): mede se o flash attention
        # deixa ESTA placa mais rápida ou mais lenta e guarda a escolha.
        key = f"{os.path.basename(self.model)}|{os.path.getsize(self.model)}|{os.path.getmtime(self.exe):.0f}"
        calib = cfg.get("vk_calib") or {}
        if calib.get("key") == key and "flash_attn" in calib:
            self._start(bool(calib["flash_attn"]), on_status)
        else:
            on_status("Calibrando a placa de vídeo (só na 1ª vez, ~1 min)…")
            self._start(True, on_status)
            t_fa = self.bench()
            log(f"calibração: flash attention LIGADO = {t_fa:.2f}s por passada")
            fa = True
            if not self.slow:
                self.close()
                self._start(False, on_status)
                t_nofa = self.bench()
                log(f"calibração: flash attention DESLIGADO = {t_nofa:.2f}s por passada")
                if t_nofa < t_fa * 0.95:
                    fa = False
                else:
                    self.close()
                    self._start(True, on_status)
            calib = {"key": key, "flash_attn": fa,
                     "t_fa": round(t_fa, 3), "t_nofa": round(t_nofa, 3) if not self.slow else None}
            log("calibração escolhida:", calib)
            if on_save:
                on_save({"vk_calib": calib})
        self.final_beam = 1 if self.slow else 3

    def _start(self, flash_attn, on_status):
        import socket
        import subprocess
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.url = f"http://127.0.0.1:{self.port}"

        self.server_log = os.path.join(CFG_DIR, "whisper-server.log")
        logf = open(self.server_log, "w", encoding="utf-8", errors="replace")
        args = [self.exe, "-m", self.model, "-l", "he", "--host", "127.0.0.1", "--port", str(self.port),
                "-t", str(max(2, min(6, (os.cpu_count() or 4) - 2))), "-nlp", "-sns",
                "-fa" if flash_attn else "-nfa"]
        if self.cfg.get("device") == "cpu":
            args.append("-ng")
        log("iniciando:", " ".join(args))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT,
                                     cwd=os.path.dirname(self.exe), creationflags=flags)

        on_status("Carregando modelo na placa de vídeo (Vulkan)…")
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"whisper-server fechou (código {self.proc.returncode}); veja whisper-server.log")
            try:
                if self.session.get(self.url + "/health", timeout=1).status_code == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > 180:
                self.close()
                raise RuntimeError("whisper-server não respondeu em 3 min")
            time.sleep(0.3)

        self.label = self._detect_gpu()
        self.slow = self.label == "CPU"
        # aquecimento (a 1ª execução no Vulkan compila os shaders)
        on_status(f"Preparando {self.label}…")
        try:
            self._infer(np.zeros(16000 * 2, dtype=np.float32), "", final=False)
        except Exception as e:
            log("aquecimento falhou:", e)

    def bench(self, n=3):
        """Tempo médio de uma passada (5 s de áudio silencioso: mede principalmente o encoder)."""
        a = np.zeros(16000 * 5, dtype=np.float32)
        ts = []
        for _ in range(n):
            t0 = time.time()
            try:
                self._infer(a, "", final=False)
            except Exception as e:
                log("bench falhou:", e)
                return 999.0
            ts.append(time.time() - t0)
        return float(np.median(ts))

    def _detect_gpu(self):
        try:
            with open(self.server_log, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
            for line in txt.splitlines():
                if "ggml_vulkan: 0 =" in line:
                    name = line.split("=", 1)[1].split("(")[0].strip()
                    return f"GPU Vulkan: {name}"
            if "Vulkan" in txt and "no GPU found" not in txt:
                return "GPU Vulkan"
        except Exception:
            pass
        return "CPU"

    def _infer(self, audio16, prompt, final, audio_ctx=None):
        """Uma passada no whisper-server. Retorna (texto, confiança média, n_segmentos)."""
        data = {
            "language": "he",
            "response_format": "verbose_json",
            "temperature": "0.0",
            # parcial: sem re-tentativas com temperatura maior (cada uma repete o trabalho)
            "temperature_inc": "0.2" if final else "0.0",
            "beam_size": str(self.final_beam if final else 1),
            "no_timestamps": "true",
            "no_language_probabilities": "true",
            "suppress_non_speech": "true",
        }
        if prompt:
            data["prompt"] = prompt
        if audio_ctx:
            data["audio_ctx"] = str(int(audio_ctx))
        files = {"file": ("trecho.wav", wav_bytes(audio16), "audio/wav")}
        r = self.session.post(self.url + "/inference", data=data, files=files, timeout=120)
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(j["error"])
        out, lps = [], []
        for seg in j.get("segments", []):
            nsp = float(seg.get("no_speech_prob", 0) or 0)
            alp = float(seg.get("avg_logprob", 0) or 0)
            if nsp > 0.6 and alp < -1.0:
                continue
            t = str(seg.get("text", "")).strip()
            if t:
                out.append(t)
                lps.append(alp)
        if not j.get("segments"):
            out = [str(j.get("text", "")).strip()]
        text = " ".join(t for t in out if t).strip()
        return text, (float(np.mean(lps)) if lps else 0.0), len(out)

    # ---- encoder encurtado com verificação ---------------------------------
    # Pesquisa (FUTO whisper-acft) e testes próprios: encurtar o encoder do
    # modelo turbo deixa 2-4x mais rápido, mas às vezes o decodificador "se perde".
    # Por isso cada resultado rápido é conferido; se parecer ruim, refaz completo.

    def _looks_bad(self, text, lp, secs):
        import zlib
        if not text:
            return False
        b = text.encode("utf-8")
        if len(b) > 40 and len(b) / max(1, len(zlib.compress(b))) > 2.4:
            return True                      # texto repetitivo
        if len(text) / max(secs, 0.5) > 32:
            return True                      # texto demais para a duração do áudio
        hist = self.lp_hist
        ref = float(np.median(hist)) if len(hist) >= 8 else -0.30
        return lp < min(-0.35, ref - 0.20)   # confiança bem abaixo do normal

    def transcribe(self, audio16, prompt, final=True):
        secs = len(audio16) / 16000.0
        if not self.audio_ctx or secs > 26:
            return self._infer(audio16, prompt, final)[0]
        ctx = int(min(1500, max(256, secs * 50 + 128)))
        text, lp, _ = self._infer(audio16, prompt, final, audio_ctx=ctx)
        if not final:
            return text                      # provisória: rápida, sem conferência
        self.fast_n += 1
        if self._looks_bad(text, lp, secs):
            self.fast_redo += 1
            full, lp_full, _ = self._infer(audio16, prompt, final)
            log(f"modo rápido refeito (conf. {lp:.2f} -> {lp_full:.2f}): {text[:60]!r} -> {full[:60]!r}")
            text = full
        else:
            self.lp_hist.append(lp)
        # Se precisar refazer muito, o modo rápido não compensa: desliga sozinho
        if self.fast_n >= 15 and self.fast_redo / self.fast_n > 0.35:
            self.audio_ctx = False
            log(f"modo rápido desligado: {self.fast_redo}/{self.fast_n} trechos refeitos")
        return text

    def close(self):
        if not self.proc:
            return
        try:
            self.proc.terminate()
            self.proc.wait(5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def create_backend(cfg, on_status, on_save=None):
    eng = cfg.get("engine", "auto")
    expected = os.path.join(app_dir(), "whispercpp", "whisper-server.exe")
    log(f"motor: {eng} | whisper-server: {expected} existe={os.path.exists(expected)}")
    if eng in ("auto", "vulkan") and not whisper_server_exe():
        on_status("Placa de vídeo indisponível: falta a pasta 'whispercpp' ao lado do LegendaViva.exe")
        time.sleep(4)
    if eng in ("auto", "vulkan") and whisper_server_exe():
        try:
            return WhisperCppBackend(cfg, on_status, on_save)
        except Exception as e:
            log("falha no motor Vulkan:", traceback.format_exc())
            on_status(f"Placa de vídeo falhou ({e}); usando o processador…")
            time.sleep(2)
    return FasterWhisperBackend(cfg, on_status)


class Engine:
    """Junta captura + segmentação + Whisper + tradução numa thread."""

    def __init__(self, cfg, ui_q):
        self.cfg = cfg
        self.ui_q = ui_q
        self.audio_q = queue.Queue()
        self.stop_evt = threading.Event()
        self.paused = threading.Event()
        self.capture = None
        self.n_chunks = 0
        self.n_silent = 0
        self.tr_lock = threading.Lock()
        self.tr_evt = threading.Event()
        self.tr_finals = deque()
        self.tr_partial = None
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
            backend = create_backend(self.cfg, self.status,
                                     lambda d: self.ui_q.put(("save_cfg", d)))
        except Exception as e:
            log("erro carregando motor:", traceback.format_exc())
            self.status(f"Erro: {e}")
            self.ui_q.put(("error", str(e)))
            return
        self.backend = backend
        if self.stop_evt.is_set():
            backend.close()
            return

        # A tradução roda numa thread separada para não atrasar o reconhecimento
        threading.Thread(target=self._translate_loop, daemon=True).start()

        self.capture = AudioCapture(self.audio_q, self.cfg["loopback_device"], self.status)
        self.capture.start()
        self.capture.ready.wait(10)
        if self.capture.error or not self.capture.rate:
            backend.close()
            return
        rate = self.capture.rate
        dev_txt = backend.label
        self.status(f"● Ouvindo ({dev_txt})")

        try:
            self._loop(backend, rate, dev_txt)
        finally:
            backend.close()

    def _prep16(self, x, rate):
        a = to_16k(x, rate)
        # normaliza o volume (áudio baixo atrapalha o reconhecimento)
        peak = float(np.max(np.abs(a))) if len(a) else 0.0
        if 0 < peak < 0.5:
            a = (a * min(20.0, 0.5 / peak)).astype(np.float32)
        return a

    def _loop(self, backend, rate, dev_txt):
        """
        Estratégia de baixa latência (inspirada no Whisper-Streaming):
          - o áudio da frase atual vai crescendo num buffer;
          - sempre que a placa de vídeo fica livre e chegou áudio novo, o buffer
            inteiro é reconhecido de novo e vira uma LEGENDA PROVISÓRIA (cinza);
          - quando a frase termina (pausa na fala ou tamanho máximo), é feita a
            passada FINAL (branca), com mais qualidade, e o buffer recomeça.
        O ritmo das passadas se ajusta sozinho à velocidade do PC.
        """
        buf = np.zeros(0, dtype=np.float32)
        last_audio = time.time()
        prev_text = ""
        base_target = float(self.cfg["chunk_seconds"])
        target = base_target
        partials = bool(self.cfg.get("partials", True))
        step = float(self.cfg.get("partial_step", 0.5))
        proc_final = None
        proc_part = None
        seg_id = 0
        last_part_len = 0
        last_part_text = ""
        partials_paused = False  # legenda provisória suspensa se a placa for lenta

        while not self.stop_evt.is_set():
            got = False
            try:
                blocks = [self.audio_q.get(timeout=0.05 if len(buf) else 0.2)]
                got = True
                while True:
                    blocks.append(self.audio_q.get_nowait())
            except queue.Empty:
                pass
            if got:
                last_audio = time.time()
                if self.paused.is_set():
                    buf = np.zeros(0, dtype=np.float32)
                    last_part_len = 0
                    continue
                buf = np.concatenate([buf] + blocks)
                self._report_level(blocks)

            flush = (not got) and (time.time() - last_audio > 0.4) and len(buf) > 0

            max_keep = int(rate * (target + 3.0) * 3)
            if len(buf) > max_keep:
                buf = buf[-int(rate * (target + 3.0)):]
                last_part_len = 0
                self.status(f"● Ouvindo ({dev_txt}) - atrasado")

            # ---------- passada FINAL ----------
            cut = decide_cut(buf, rate, target, flush=flush)
            if cut is None and proc_final is not None and (not partials or partials_paused):
                # A placa está livre agora: se já juntou áudio equivalente ao tempo
                # de uma passada, corta entre palavras e processa (computacionalmente consciente).
                eager_min = min(target, max(1.6, proc_final * 0.9))
                cut = decide_cut_eager(buf, rate, eager_min)
            if cut is not None:
                chunk, buf = buf[:cut], buf[cut:]
                last_part_len = 0
                last_part_text = ""
                rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
                if rms < 0.0005:
                    self.n_silent += 1
                    self.ui_q.put(("clear_partial", seg_id))
                    seg_id += 1
                    continue
                self.n_chunks += 1
                t0 = time.time()
                try:
                    he = backend.transcribe(self._prep16(chunk, rate), prev_text[-120:], final=True)
                except Exception as e:
                    log("erro no reconhecimento:", traceback.format_exc())
                    self.status(f"Erro no reconhecimento: {e}")
                    time.sleep(0.5)
                    continue
                proc = time.time() - t0
                proc_final = proc if proc_final is None else 0.7 * proc_final + 0.3 * proc

                # Ajustes automáticos conforme a velocidade da placa/CPU
                if hasattr(backend, "final_beam") and not backend.slow:
                    backend.final_beam = 3 if proc_final < 1.2 else 1
                budget = proc_final + (0 if (not partials or partials_paused) else (proc_part or 0) * 1.5)
                new_target = min(20.0, max(base_target, budget * 1.15))
                if abs(new_target - target) > 0.5:
                    target = new_target
                    extra = "" if target <= base_target + 0.1 else f" - frases de até {target:.0f}s (PC lento)"
                    self.status(f"● Ouvindo ({dev_txt}){extra}")

                he = clean_text(he)
                if not he or is_hallucination(he):
                    log(f"[final {proc:.2f}s] (descartado) {he}")
                    self.ui_q.put(("clear_partial", seg_id))
                    seg_id += 1
                    continue
                prev_text = he
                log(f"[final {proc:.2f}s / áudio {len(chunk) / rate:.1f}s] HE: {he}")
                self._put_translation(("final", seg_id, he, time.time()))
                seg_id += 1
                continue

            # ---------- passada PROVISÓRIA ----------
            if not partials or partials_paused:
                continue
            if len(buf) < int(rate * 0.8) or len(buf) - last_part_len < int(rate * step):
                continue
            tail = buf[-int(rate * 0.4):]
            if float(np.sqrt(np.mean(tail * tail))) < 0.0005 and \
                    float(np.sqrt(np.mean(buf * buf))) < 0.0005:
                continue
            t0 = time.time()
            try:
                he = backend.transcribe(self._prep16(buf, rate), prev_text[-120:], final=False)
            except Exception as e:
                log("erro na legenda provisória:", e)
                last_part_len = len(buf)
                continue
            proc = time.time() - t0
            proc_part = proc if proc_part is None else 0.7 * proc_part + 0.3 * proc
            last_part_len = len(buf)
            if proc_part > 1.6 and not partials_paused:
                # Cada passada demora demais: a legenda provisória só atrasaria a final
                partials_paused = True
                log(f"legenda provisória suspensa (placa leva {proc_part:.1f}s por passada)")
                self.status(f"● Ouvindo ({dev_txt}) - legenda provisória desligada (placa lenta)")
            he = clean_text(he)
            if he and he != last_part_text and not is_hallucination(he):
                last_part_text = he
                log(f"[parcial {proc:.2f}s / áudio {len(buf) / rate:.1f}s] {he}")
                self._put_translation(("partial", seg_id, he, time.time()))

    def _put_translation(self, item):
        with self.tr_lock:
            if item[0] == "final":
                self.tr_finals.append(item)
                # parcial antigo da mesma frase não interessa mais
                if self.tr_partial and self.tr_partial[1] <= item[1]:
                    self.tr_partial = None
            else:
                self.tr_partial = item  # só a mais recente importa
        self.tr_evt.set()

    def _report_level(self, blocks):
        lvl = max(float(np.sqrt(np.mean(b * b))) for b in blocks if len(b))
        self.level_peak = max(getattr(self, "level_peak", 0.0), lvl)
        now = time.time()
        if now - getattr(self, "last_level_ui", 0) > 0.25:
            self.ui_q.put(("level", self.level_peak))
            self.level_peak = 0.0
            self.last_level_ui = now
        if now - getattr(self, "last_level_log", now) > 15:
            log(f"nível de áudio ~{lvl:.4f} | trechos enviados: {self.n_chunks} | silêncio: {self.n_silent}")
            self.last_level_log = now
        elif not hasattr(self, "last_level_log"):
            self.last_level_log = now

    def _translate_loop(self):
        try:
            translator = Translator(self.cfg.get("translator", "auto"), self.status)
        except Exception as e:
            log("erro tradutor:", traceback.format_exc())
            self.status(f"Erro no tradutor: {e}")
            translator = None
        last_partial_call = 0.0
        while not self.stop_evt.is_set():
            self.tr_evt.wait(0.3)
            with self.tr_lock:
                item = None
                if self.tr_finals:
                    item = self.tr_finals.popleft()
                elif self.tr_partial and time.time() - last_partial_call >= 0.6:
                    # limite de ~1,6 traduções provisórias/s para o Google não bloquear
                    item, self.tr_partial = self.tr_partial, None
                if not self.tr_finals and not self.tr_partial:
                    self.tr_evt.clear()
            if item is None:
                if self.tr_partial:
                    time.sleep(0.05)
                continue
            kind, sid, he, t_rec = item
            t0 = time.time()
            if kind == "partial":
                last_partial_call = t0
            pt = translator(he, partial=(kind == "partial")) if translator else None
            dt = time.time() - t0
            if kind == "final":
                log(f"PT ({translator.last_used if translator else '-'}, {dt:.2f}s): {pt}")
                self.ui_q.put(("text", sid, he, pt))
            elif pt:
                with self.tr_lock:
                    stale = any(f[1] >= sid for f in self.tr_finals)
                if not stale:
                    self.ui_q.put(("partial", sid, he, pt))


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
        # legenda provisória (enquanto a frase ainda está sendo falada)
        self.part_lbl = tk.Label(self, text="", bg=self.BG, fg="#b5b5b5",
                                 font=("Segoe UI", int(self.cfg["font_size"]), "bold"),
                                 justify="center")
        self.partial = None   # (seg_id, he, pt)
        self.last_final_sid = -1
        self._pack_labels()

        self.grip = tk.Label(self, text="◢", bg=self.BG, fg="#555555",
                             font=("Segoe UI", 11), cursor="size_nw_se")
        self.grip.place(relx=1.0, rely=1.0, anchor="se")

        # Arrastar / redimensionar / fonte / menu
        for w in (self, self.status_lbl, self.text_lbl, self.orig_lbl, self.part_lbl):
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
        self.part_lbl.pack_forget()
        if self.cfg["show_original"]:
            self.orig_lbl.pack(fill="x", padx=14)
        self.text_lbl.pack(fill="x", padx=14)
        self.part_lbl.pack(fill="x", padx=14, pady=(0, 8))
        self._update_wrap()

    def _update_wrap(self):
        try:
            w = max(200, self.winfo_width() - 40)
            self.text_lbl.configure(wraplength=w)
            self.orig_lbl.configure(wraplength=w)
            self.part_lbl.configure(wraplength=w)
        except tk.TclError:
            pass

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
                    self.status_txt = item[1]
                    self._show_status()
                elif kind == "save_cfg":
                    self.cfg.update(item[1])
                    save_cfg(self.cfg)
                elif kind == "level":
                    self.level = item[1]
                    self._show_status()
                elif kind == "text":
                    _, sid, he, pt = item
                    self.lines.append((he, pt or f"(sem tradução) {he}"))
                    self.last_final_sid = max(self.last_final_sid, sid)
                    if self.partial and self.partial[0] <= sid:
                        self.partial = None
                    self.last_text_time = time.time()
                    self._render()
                elif kind == "partial":
                    _, sid, he, pt = item
                    if sid > self.last_final_sid:
                        self.partial = (sid, he, pt)
                        self.last_text_time = time.time()
                        self._render()
                elif kind == "clear_partial":
                    if self.partial and self.partial[0] <= item[1]:
                        self.partial = None
                        self._render()
                elif kind == "error":
                    self.text_lbl.configure(text="Não consegui iniciar. Botão direito → Configurações\n"
                                                 "(tente CPU ou um modelo menor).")
        except queue.Empty:
            pass
        clear_after = float(self.cfg.get("clear_after", 0) or 0)
        if clear_after and self.lines and time.time() - self.last_text_time > clear_after:
            self.clear()
        self.after(30, self._poll)

    def _show_status(self):
        txt = getattr(self, "status_txt", "")
        lvl = getattr(self, "level", None)
        if lvl is not None:
            # escala em dB: -60 dB = vazio, 0 dB = cheio
            import math
            db = 20 * math.log10(max(lvl, 1e-6))
            n = int(max(0, min(10, (db + 60) / 6)))
            bar = "▮" * n + "▯" * (10 - n)
            aviso = "  (sem som chegando!)" if db < -60 else ""
            txt = f"{txt}    áudio {bar}{aviso}"
        self.status_lbl.configure(text=txt)

    def _render(self):
        n = int(self.cfg["max_lines"])
        finals = list(self.lines)
        if self.partial:
            finals = finals[-(n - 1):] if n > 1 else []
        self.text_lbl.configure(text="\n".join(pt for _, pt in finals))
        self.part_lbl.configure(text=self.partial[2] if self.partial else "")
        he_lines = [he for he, _ in finals] + ([self.partial[1]] if self.partial else [])
        self.orig_lbl.configure(text="\n".join(he_lines))

    def clear(self):
        self.lines.clear()
        self.partial = None
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
        self.part_lbl.configure(font=("Segoe UI", size, "bold"))
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
                      ("model", "device", "loopback_device", "chunk_seconds", "translator", "engine", "fast_encoder", "partials"))
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

        ttk.Label(self, text="Modelo (faster-whisper):").grid(row=0, column=0, **pad)
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

        ttk.Label(self, text="Tamanho máx. da frase (s):").grid(row=4, column=0, **pad)
        self.chunk_var = tk.DoubleVar(value=float(cfg["chunk_seconds"]))
        ttk.Spinbox(self, from_=3, to=10, increment=0.5, textvariable=self.chunk_var,
                    width=6).grid(row=4, column=1, **pad)
        ttk.Label(self, text="Com legenda provisória ligada, 5–8 s é o ideal (o atraso não aumenta).",
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

        ttk.Label(self, text="Tradução:").grid(row=10, column=0, **pad)
        self.tr_labels = {
            "auto": "Automática (Google → MyMemory → offline)",
            "online": "Só online (Google / MyMemory)",
            "offline": "Só offline (NLLB, sem internet)",
        }
        self.tr_var = tk.StringVar(value=self.tr_labels.get(cfg.get("translator", "auto")))
        ttk.Combobox(self, textvariable=self.tr_var, values=list(self.tr_labels.values()),
                     state="readonly", width=42).grid(row=10, column=1, **pad)

        ttk.Label(self, text="Motor de reconhecimento:").grid(row=11, column=0, **pad)
        self.eng_labels = {
            "auto": "Automático (placa de vídeo Vulkan, senão processador)",
            "vulkan": "whisper.cpp + Vulkan (AMD RX / Intel / NVIDIA)",
            "faster-whisper": "faster-whisper (processador ou NVIDIA CUDA)",
        }
        self.eng_var = tk.StringVar(value=self.eng_labels.get(cfg.get("engine", "auto")))
        ttk.Combobox(self, textvariable=self.eng_var, values=list(self.eng_labels.values()),
                     state="readonly", width=42).grid(row=11, column=1, **pad)
        if not whisper_server_exe():
            ttk.Label(self, text="(whisper-server.exe não encontrado nesta instalação)",
                      foreground="#a00").grid(row=12, column=1, padx=10, sticky="w")

        self.fast_var = tk.BooleanVar(value=bool(cfg.get("fast_encoder", False)))
        ttk.Checkbutton(self, text="Modo rápido da placa (encoder encurtado + verificação automática; 2–3× mais rápido)",
                        variable=self.fast_var).grid(row=13, column=0, columnspan=2, **pad)

        self.part_var = tk.BooleanVar(value=bool(cfg.get("partials", True)))
        ttk.Checkbutton(self, text="Legenda provisória (cinza) enquanto a frase é falada — menor atraso",
                        variable=self.part_var).grid(row=14, column=0, columnspan=2, **pad)

        btns = ttk.Frame(self)
        btns.grid(row=15, column=0, columnspan=2, pady=10)
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
                "translator": next((k for k, v in self.tr_labels.items()
                                    if v == self.tr_var.get()), "auto"),
                "engine": next((k for k, v in self.eng_labels.items()
                                if v == self.eng_var.get()), "auto"),
                "fast_encoder": bool(self.fast_var.get()),
                "partials": bool(self.part_var.get()),
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
