"""
Indian Storytelling Ambient Music & Gentle Transition Sound Designer (`src/sfx_generator.py`):
1. Synthesizes a peaceful, warm Indian Tanpura + Bansuri Flute + Santoor inspired storytelling soundtrack
   (`interstellar_bgm.wav`, `kahani_bgm.wav`) suitable for both kids and adults.
2. Generates a soft, non-intrusive scene transition chime & breeze track (`generate_scene_transition_sfx_track`)
   so scene cuts feel warm and magical.
"""
from pathlib import Path
from typing import List
import wave
import numpy as np

SFX_DIR = Path("assets/sfx")
BGM_DIR = Path("assets/bgm")


def save_wav(filename: Path, samples: np.ndarray, sample_rate: int = 44100):
    filename.parent.mkdir(parents=True, exist_ok=True)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    if samples.ndim == 1:
        max_val = float(np.max(np.abs(samples)))
        if max_val > 0:
            samples = samples / max_val * 0.90
        int_samples = (samples * 32767).astype(np.int16)
        stereo = np.column_stack((int_samples, int_samples)).flatten()
    else:
        max_val = float(np.max(np.abs(samples)))
        if max_val > 0:
            samples = samples / max_val * 0.90
        int_samples = (samples * 32767).astype(np.int16)
        stereo = int_samples.flatten()

    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def generate_hook_impact_track(duration_sec: float = 60.0, sr: int = 44100) -> np.ndarray:
    n_samples = int(sr * duration_sec)
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    hit_len = int(1.4 * sr)
    t_hit = np.linspace(0, 1.4, hit_len, endpoint=False)
    # Warm Indian temple bell / santoor chime at 0.0s
    chime_l = (0.35 * np.sin(2 * np.pi * 523.25 * t_hit) + 0.20 * np.sin(2 * np.pi * 783.99 * t_hit)) * np.exp(-2.8 * t_hit)
    chime_r = (0.35 * np.sin(2 * np.pi * 659.25 * t_hit) + 0.20 * np.sin(2 * np.pi * 1046.50 * t_hit)) * np.exp(-2.8 * t_hit)

    left[:hit_len] += chime_l
    right[:hit_len] += chime_r
    return np.column_stack((left, right))


def generate_scene_transition_sfx_track(
    scenes: List[dict], total_duration_sec: float, out_wav: Path, sr: int = 44100
) -> Path:
    dur = max(15.0, float(total_duration_sec) + 1.5)
    n_samples = int(sr * dur)
    left = np.zeros(n_samples, dtype=np.float64)
    right = np.zeros(n_samples, dtype=np.float64)

    # Opening warm bell chime
    hit_len = min(n_samples, int(1.2 * sr))
    t_hit = np.linspace(0, 1.2, hit_len, endpoint=False)
    bell = 0.28 * np.sin(2 * np.pi * 528.0 * t_hit) * np.exp(-3.0 * t_hit)
    left[:hit_len] += bell
    right[:hit_len] += bell

    # Gentle harp/santoor shimmer on scene transitions
    notes = [523.25, 587.33, 659.25, 783.99, 880.00]
    for idx, scene in enumerate(scenes):
        cut_sec = float(scene.get("start_sec", 0.0))
        if idx > 0 and 0.4 < cut_sec < dur - 0.6:
            w_dur = 0.35
            w_len = int(w_dur * sr)
            w_start = max(0, int((cut_sec - 0.05) * sr))
            w_end = min(n_samples, w_start + w_len)
            actual_len = w_end - w_start
            if actual_len > 100:
                tw = np.linspace(0, w_dur, actual_len, endpoint=False)
                freq = notes[idx % len(notes)]
                shimmer = 0.14 * np.sin(2 * np.pi * freq * tw) * np.exp(-6.0 * tw)
                left[w_start:w_end] += shimmer * 0.6
                right[w_start:w_end] += shimmer * 0.4

    stereo = np.column_stack((left, right))
    save_wav(out_wav, stereo, sample_rate=sr)
    return out_wav


def generate_indian_storytelling_bgm(duration_sec: float = 90.0, sr: int = 44100) -> np.ndarray:
    """Synthesizes a peaceful Indian Pentatonic (Raag Bhupali — Sa Re Ga Pa Dha) Tanpura & Bansuri ambient bed."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Soft Tanpura drone (Sa = C3 130.81Hz, Pa = G3 196.00Hz)
    sa = 130.81
    pa = 196.00
    breathe = 0.82 + 0.18 * np.sin(2 * np.pi * 0.18 * t)
    drone_l = (0.14 * np.sin(2 * np.pi * sa * t) + 0.09 * np.sin(2 * np.pi * pa * t)) * breathe
    drone_r = (0.14 * np.sin(2 * np.pi * (sa * 1.001) * t) + 0.09 * np.sin(2 * np.pi * (pa * 0.999) * t)) * breathe

    left = drone_l.copy()
    right = drone_r.copy()

    # Raag Bhupali notes (C4, D4, E4, G4, A4, C5)
    raag_notes = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 440.00, 392.00]
    step_sec = 1.25
    total_steps = int(duration_sec / step_sec)

    for idx in range(total_steps):
        freq = raag_notes[idx % len(raag_notes)]
        s0 = int(idx * step_sec * sr)
        n_len = min(int(step_sec * 1.6 * sr), n_samples - s0)
        if n_len <= 0:
            break
        tn = np.linspace(0, n_len / sr, n_len, endpoint=False)
        env = np.sin(np.pi * tn / (n_len / sr)) ** 1.4
        vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 4.8 * tn)
        flute = (0.12 * np.sin(2 * np.pi * freq * vibrato * tn) + 0.04 * np.sin(2 * np.pi * freq * 2.0 * tn)) * env
        pan = 0.60 if idx % 2 == 0 else 0.40
        left[s0 : s0 + n_len] += flute * pan
        right[s0 : s0 + n_len] += flute * (1.0 - pan)

    fi = int(1.0 * sr)
    fo = int(2.0 * sr)
    left[:fi] *= np.linspace(0.0, 1.0, fi)
    right[:fi] *= np.linspace(0.0, 1.0, fi)
    left[-fo:] *= np.linspace(1.0, 0.0, fo)
    right[-fo:] *= np.linspace(1.0, 0.0, fo)

    return np.column_stack((left, right))


def generate_all_sfx():
    sr = 44100
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    BGM_DIR.mkdir(parents=True, exist_ok=True)

    bgm_stereo = generate_indian_storytelling_bgm(duration_sec=90.0, sr=sr)
    save_wav(SFX_DIR / "interstellar_bgm.wav", bgm_stereo, sr)
    save_wav(SFX_DIR / "kahani_bgm.wav", bgm_stereo, sr)
    save_wav(SFX_DIR / "ambient_bed.wav", bgm_stereo, sr)

    hook_impact = generate_hook_impact_track(duration_sec=60.0, sr=sr)
    save_wav(SFX_DIR / "hook_impact.wav", hook_impact, sr)
    print(f"[SFX] Generated Indian Storytelling Raag Bhupali Score & Chimes -> {SFX_DIR}")


if __name__ == "__main__":
    generate_all_sfx()
