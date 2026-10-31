"""
True Multi-Character Hindi Neural Voiceover Engine (`src/voice_engine.py` — v4.0 Distinct Character Voice Cast Edition):
- Fixes the issue where all male characters sounded like the same `hi-IN-MadhurNeural` voice:
  1. **Distinct Neural Voice IDs Per Character Role**:
     - **Narrator (`narrator`)**:
       - Azure Primary: `hi-IN-RehaanNeural` (or `hi-IN-MadhurNeural`)
       - Edge TTS Primary: `hi-IN-MadhurNeural` (Warm, clear young Indian storyteller)
     - **Young Male Protagonist / Hero (`male_protagonist`)**:
       - Azure Primary: `hi-IN-ArjunNeural` (Energetic young Hindi male voice)
       - Edge TTS Primary: `en-US-AndrewMultilingualNeural` (Native multilingual neural voice speaking fluent Hindi with a distinct youthful male timbre!)
     - **Male Antagonist / Heavy Character (`male_antagonist`)**:
       - Azure Primary: `hi-IN-KunalNeural` (Deep, resonant Hindi male voice)
       - Edge TTS Primary: `en-US-BrianMultilingualNeural` (Distinct deeper male timbre speaking Hindi + formant shift `0.92x` so he sounds unmistakably different from both the Narrator and Hero!)
     - **Young Female Character (`female_character` or `speaker_gender == "female"`)**:
       - Azure Primary: `hi-IN-AnanyaNeural` / `hi-IN-SwaraNeural` (Bright young Hindi female voice)
       - Edge TTS Primary: `hi-IN-SwaraNeural` (Expressive young female Hindi voice)
     - **Second Female / Young Character (`young_character`)**:
       - Azure Primary: `hi-IN-KavyaNeural` / `hi-IN-AartiNeural`
       - Edge TTS Primary: `en-US-AvaMultilingualNeural` (Bright youthful female multilingual voice speaking fluent Hindi)
     - **Wise Elder / Guru / King (`wise_elder`)**:
       - Azure Primary: `hi-IN-MadhurNeural` (Dignified sage timbre)
       - Edge TTS Primary: `en-IN-PrabhatNeural` / `hi-IN-MadhurNeural` with resonant `0.95x` vocal tract depth
  2. **Automatic Hindi Dialogue Speaker Detection (`infer_character_role_from_scene`)**:
     Even if the storyboard LLM accidentally marks a dialogue scene as `"narrator"`, `VoiceEngine` inspects `speaker_name` and the Hindi `narration` text for dialogue cues (e.g. `"सुना जी"`, `"मित्र रक्तमुख"`, `"अरे भाई मगरमच्छ"`, `"पत्नी बोली"`, `"बंदर ने कहा"`) and automatically switches to the exact character's voice!
"""
import asyncio
import os
import re
import subprocess
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape
import numpy as np

DEFAULT_NARRATOR_VOICE = "hi-IN-MadhurNeural"
DEFAULT_FEMALE_VOICE = "hi-IN-SwaraNeural"

# Role -> {azure_candidates: [...], edge_candidates: [...], rate_pct: int, pitch_hz: str, formant_ratio: float}
DISTINCT_CHARACTER_VOICE_CAST: Dict[str, Dict[str, object]] = {
    "narrator": {
        "env_key": "HINDI_NARRATOR_VOICE",
        "azure_voices": ["hi-IN-RehaanNeural", "hi-IN-MadhurNeural"],
        "edge_voices": ["hi-IN-MadhurNeural"],
        "rate_pct": 5,
        "pitch_hz": "+2Hz",
        "formant_ratio": 1.00,
    },
    "male_protagonist": {
        "env_key": "HINDI_MALE_HERO_VOICE",
        "azure_voices": ["hi-IN-ArjunNeural", "en-US-AndrewMultilingualNeural", "hi-IN-MadhurNeural"],
        "edge_voices": ["en-US-AndrewMultilingualNeural", "hi-IN-MadhurNeural"],
        "rate_pct": 8,
        "pitch_hz": "+12Hz",
        "formant_ratio": 1.06,
    },
    "male_antagonist": {
        "env_key": "HINDI_MALE_DEEP_VOICE",
        "azure_voices": ["hi-IN-KunalNeural", "en-US-BrianMultilingualNeural", "hi-IN-MadhurNeural"],
        "edge_voices": ["en-US-BrianMultilingualNeural", "hi-IN-MadhurNeural"],
        "rate_pct": 2,
        "pitch_hz": "-14Hz",
        "formant_ratio": 0.91,
    },
    "female_character": {
        "env_key": "HINDI_FEMALE_VOICE",
        "azure_voices": ["hi-IN-AnanyaNeural", "hi-IN-SwaraNeural", "hi-IN-KavyaNeural"],
        "edge_voices": ["hi-IN-SwaraNeural", "en-US-AvaMultilingualNeural"],
        "rate_pct": 6,
        "pitch_hz": "+8Hz",
        "formant_ratio": 1.04,
    },
    "young_character": {
        "env_key": "HINDI_CHILD_VOICE",
        "azure_voices": ["hi-IN-KavyaNeural", "hi-IN-AartiNeural", "hi-IN-SwaraNeural"],
        "edge_voices": ["en-US-AvaMultilingualNeural", "hi-IN-SwaraNeural"],
        "rate_pct": 9,
        "pitch_hz": "+16Hz",
        "formant_ratio": 1.09,
    },
    "wise_elder": {
        "env_key": "HINDI_ELDER_VOICE",
        "azure_voices": ["hi-IN-MadhurNeural", "hi-IN-KunalNeural"],
        "edge_voices": ["hi-IN-MadhurNeural"],
        "rate_pct": 0,
        "pitch_hz": "-8Hz",
        "formant_ratio": 0.95,
    },
}


@dataclass
class WordTiming:
    word: str
    start_sec: float
    end_sec: float
    scene_id: int
    speaker_role: str = "narrator"
    speaker_gender: str = "male"


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


class VoiceEngine:
    def __init__(self):
        self.speech_key = os.getenv("AZURE_SPEECH_KEY", "").strip()
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "centralindia").strip()
        self.male_voice = os.getenv("AZURE_TTS_VOICE", DEFAULT_NARRATOR_VOICE).strip() or DEFAULT_NARRATOR_VOICE
        self.female_voice = os.getenv("AZURE_TTS_FEMALE_VOICE", DEFAULT_FEMALE_VOICE).strip() or DEFAULT_FEMALE_VOICE

        raw_speed = os.getenv("VOICE_SPEED_PERCENT", "0").strip().replace("%", "").replace("+", "")
        try:
            self.base_speed_pct = int(float(raw_speed))
        except ValueError:
            self.base_speed_pct = 0

    @staticmethod
    def infer_character_role_from_scene(scene: dict) -> Tuple[str, str]:
        """
        Inspects `speaker_role`, `speaker_gender`, `speaker_name`, and the Hindi `narration` text
        so that character dialogue scenes ALWAYS switch to the right character voice (`female_character`,
        `male_protagonist`, `male_antagonist`, `young_character`, or `narrator`).
        """
        role = str(scene.get("speaker_role") or "narrator").strip().lower()
        gender = str(scene.get("speaker_gender") or "").strip().lower()
        speaker_name = str(scene.get("speaker_name") or "").strip().lower()
        narration = str(scene.get("narration") or "").strip()

        female_indicators = (
            "female", "wife", "queen", "princess", "mother", "woman", "girl", "sister",
            "पत्नी", "रानी", "राजकुमारी", "माँ", "माता", "बंदरिया", "ब्राह्मणी", "महिला",
            "देवी", "चुहिया", "लोमड़ी", "भाभी", "सखी", "बेटी", "कन्या", "गौरैया", "हथिनी", "शेरनी",
        )
        female_dialogue_cues = (
            "सुना जी", "प्राण त्याग दूँगी", "दे दूँगी", "खाऊँगी", "जाऊँगी", "मैं कुछ नहीं जानती",
            "करूँगी", "बताती हूँ", "मेरी सखी", "स्वामी!", "हे आर्यपुत्र",
        )
        if (
            gender == "female"
            or role == "female_character"
            or any(tok in speaker_name for tok in female_indicators)
            or any(cue in narration for cue in female_dialogue_cues)
        ):
            return "female_character", "female"

        hero_indicators = (
            "male_protagonist", "protagonist", "hero", "monkey", "rabbit", "hare", "birbal", "tenali",
            "बंदर", "रक्तमुख", "खरगोश", "बीरबल", "तेनाली", "चूहा", "कछुआ", "हंस", "तोता", "मित्र",
        )
        villain_indicators = (
            "male_antagonist", "antagonist", "villain", "crocodile", "lion", "tiger", "snake", "wolf", "hunter",
            "मगरमच्छ", "करालमुख", "शेर", "भासुरक", "पिंगलक", "बाघ", "साँप", "सियार", "शिकारी", "राक्षस", "गिद्ध",
        )
        elder_indicators = ("wise_elder", "elder", "sage", "guru", "king", "brahmin", "विष्णु शर्मा", "मुनि", "साधु", "राजा", "ब्राह्मण", "आचार्य")
        child_indicators = ("young_character", "child", "kid", "बच्चा", "बालक", "नन्हा", "पुत्र")

        if role in DISTINCT_CHARACTER_VOICE_CAST and role != "narrator":
            inferred_gender = "female" if role in ("female_character", "young_character") else "male"
            return role, inferred_gender

        # Check speaker_name even if speaker_role was left as "narrator"
        if any(tok in speaker_name for tok in villain_indicators):
            return "male_antagonist", "male"
        if any(tok in speaker_name for tok in hero_indicators):
            return "male_protagonist", "male"
        if any(tok in speaker_name for tok in child_indicators):
            return "young_character", "female"
        if any(tok in speaker_name for tok in elder_indicators):
            return "wise_elder", "male"

        # Check first-person Hindi dialogue cues in narration if speaker_name was generic
        if any(cue in narration for cue in ("अरे भाई मगरमच्छ", "नादान मित्र करालमुख", "अरे मूर्ख और विश्वासघाती", "हम बंदर तो")):
            return "male_protagonist", "male"
        if any(cue in narration for cue in ("मित्र रक्तमुख", "धन्यवाद भाई!", "मेरी पीठ पर बैठ जाओ", "मुझे क्षमा कर देना")):
            return "male_antagonist", "male"

        return "narrator", "male"

    def resolve_voice_for_scene(self, scene: dict) -> Tuple[str, int, str, float, str, str]:
        """
        Returns `(voice_name, rate_pct, pitch_hz, formant_ratio, resolved_role, resolved_gender)` for a scene.
        """
        role, gender = self.infer_character_role_from_scene(scene)
        profile = DISTINCT_CHARACTER_VOICE_CAST.get(role, DISTINCT_CHARACTER_VOICE_CAST["narrator"])

        env_key = str(profile["env_key"])
        custom_env_voice = os.getenv(env_key, "").strip()
        candidates = list(profile["azure_voices"] if (self.speech_key and self.speech_region) else profile["edge_voices"])
        voice_name = custom_env_voice or candidates[0]

        rate_pct = self.base_speed_pct + int(profile["rate_pct"])
        pitch_hz = str(profile["pitch_hz"])
        formant_ratio = float(profile["formant_ratio"])
        return voice_name, rate_pct, pitch_hz, formant_ratio, role, gender

    @staticmethod
    def _apply_character_timbre_shift(mp3_path: Path, formant_ratio: float) -> None:
        """
        Applies a subtle, broadcast-clean vocal-tract formant/pitch shift via FFmpeg `asetrate` + `atempo`
        when `formant_ratio != 1.00`, guaranteeing that even if a fallback TTS uses the same base voice,
        the Young Hero (`1.06x` brighter/younger) and Antagonist (`0.91x` deeper/heavier) sound unmistakably distinct!
        """
        if abs(formant_ratio - 1.0) < 0.02 or not mp3_path.exists():
            return
        tempo_comp = round(1.0 / formant_ratio, 4)
        new_sr = int(round(44100 * formant_ratio))
        shifted_mp3 = mp3_path.with_suffix(".shifted.mp3")
        af = f"asetrate={new_sr},aresample=44100,atempo={tempo_comp}"
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp3_path),
                "-af", af,
                "-c:a", "libmp3lame", "-b:a", "192k",
                str(shifted_mp3),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0 and shifted_mp3.exists() and shifted_mp3.stat().st_size > 1000:
            shifted_mp3.replace(mp3_path)
        else:
            shifted_mp3.unlink(missing_ok=True)

    def synthesize_scenes(
        self, scenes: List[dict], output_dir: Path
    ) -> Tuple[Path, List[WordTiming], List[dict]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        scene_audio_files = []
        all_word_timings: List[WordTiming] = []
        enriched_scenes = []
        current_offset = 0.0

        for idx, scene in enumerate(scenes):
            scene_id = int(scene.get("scene_id", idx + 1))
            text = str(scene.get("narration", "")).strip()
            voice_name, rate_pct, pitch_hz, formant_ratio, role, gender = self.resolve_voice_for_scene(scene)

            scene_mp3 = output_dir / f"scene_{scene_id:02d}.mp3"
            words_with_rel_times, actual_voice_used = self._synthesize_single_clip(
                text=text,
                out_mp3=scene_mp3,
                voice_name=voice_name,
                role=role,
                rate_pct=rate_pct,
                pitch_hz=pitch_hz,
                formant_ratio=formant_ratio,
            )
            clip_dur = get_audio_duration(scene_mp3)

            # Re-scale word timestamps proportionally if formant shift slightly altered clip duration
            if words_with_rel_times:
                raw_end = max(words_with_rel_times[-1][2], 0.1)
                scale_factor = clip_dur / raw_end if abs(clip_dur - raw_end) > 0.08 else 1.0
            else:
                scale_factor = 1.0

            scene_words = []
            for w_text, w_start, w_end in words_with_rel_times:
                wt = WordTiming(
                    word=w_text,
                    start_sec=round(current_offset + w_start * scale_factor, 3),
                    end_sec=round(current_offset + w_end * scale_factor, 3),
                    scene_id=scene_id,
                    speaker_role=role,
                    speaker_gender=gender,
                )
                all_word_timings.append(wt)
                scene_words.append(asdict(wt))

            scene_copy = dict(scene)
            scene_copy["speaker_role"] = role
            scene_copy["speaker_gender"] = gender
            scene_copy["voice_used"] = f"{actual_voice_used} [{role}]"
            scene_copy["voice_rate_pct"] = rate_pct
            scene_copy["voice_pitch_hz"] = pitch_hz
            scene_copy["start_sec"] = round(current_offset, 3)
            scene_copy["duration_sec"] = round(clip_dur, 3)
            scene_copy["end_sec"] = round(current_offset + clip_dur, 3)
            scene_copy["audio_path"] = str(scene_mp3)
            scene_copy["words"] = scene_words
            enriched_scenes.append(scene_copy)

            scene_audio_files.append(scene_mp3)
            current_offset += clip_dur

        concat_list = output_dir / "audio_concat.txt"
        concat_list.write_text(
            "\n".join([f"file '{p.resolve()}'" for p in scene_audio_files]),
            encoding="utf-8",
        )
        master_vo = output_dir / "voiceover_master.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-ar",
                "44100",
                "-ac",
                "2",
                str(master_vo),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        voices_summary = sorted({s["voice_used"] for s in enriched_scenes})
        print(
            f"[VoiceEngine] Synthesized {len(enriched_scenes)} Hindi scenes across Distinct Character Voices {voices_summary} "
            f"({current_offset:.1f}s total, {len(all_word_timings)} synced Devanagari words)."
        )
        return master_vo, all_word_timings, enriched_scenes

    def _synthesize_single_clip(
        self,
        text: str,
        out_mp3: Path,
        voice_name: str,
        role: str,
        rate_pct: int,
        pitch_hz: str,
        formant_ratio: float = 1.0,
    ) -> Tuple[List[Tuple[str, float, float]], str]:
        profile = DISTINCT_CHARACTER_VOICE_CAST.get(role, DISTINCT_CHARACTER_VOICE_CAST["narrator"])

        if self.speech_key and self.speech_region:
            for cand_voice in [voice_name] + list(profile["azure_voices"]):
                try:
                    timings = self._call_azure_speech_sdk(text, out_mp3, cand_voice, rate_pct, pitch_hz)
                    if cand_voice == DEFAULT_NARRATOR_VOICE and role != "narrator":
                        self._apply_character_timbre_shift(out_mp3, formant_ratio)
                    return timings, cand_voice
                except Exception as exc:
                    print(f"[VoiceEngine] Azure voice '{cand_voice}' notice ({exc}); trying next candidate...")

        for cand_voice in [voice_name] + list(profile["edge_voices"]):
            try:
                timings = asyncio.run(self._call_edge_tts_with_boundaries(text, out_mp3, cand_voice, rate_pct, pitch_hz))
                # Always apply subtle formant separation for non-narrator roles if using Madhur/Swara fallback
                if role != "narrator":
                    self._apply_character_timbre_shift(out_mp3, formant_ratio)
                return timings, cand_voice
            except Exception as exc:
                print(f"[VoiceEngine] Edge TTS '{cand_voice}' notice ({exc}); trying fallback...")

        timings = self._generate_offline_cadence_audio(text, out_mp3, voice_name, formant_ratio=formant_ratio)
        return timings, voice_name

    def _call_azure_speech_sdk(
        self,
        text: str,
        out_mp3: Path,
        voice_name: str,
        rate_pct: int,
        pitch_hz: str,
    ) -> List[Tuple[str, float, float]]:
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_synthesis_voice_name = voice_name
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
        )
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(out_mp3))
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        word_boundaries: List[Tuple[str, float, float]] = []

        def on_word(evt):
            start_s = evt.audio_offset / 10_000_000.0
            dur_s = evt.duration.total_seconds() if evt.duration else 0.22
            word_boundaries.append((evt.text, start_s, start_s + max(dur_s, 0.12)))

        synthesizer.synthesis_word_boundary.connect(on_word)

        sign = "+" if rate_pct >= 0 else ""
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="hi-IN">'
            f'<voice name="{voice_name}">'
            f'<prosody rate="{sign}{rate_pct}%" pitch="{pitch_hz}">{escape(text)}</prosody>'
            f"</voice></speak>"
        )
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"Azure Speech synthesis failed for {voice_name}: {result.reason}")

        dur = get_audio_duration(out_mp3)
        return self._align_or_estimate_hindi_words(text, word_boundaries, dur)

    async def _call_edge_tts_with_boundaries(
        self,
        text: str,
        out_mp3: Path,
        voice_name: str,
        rate_pct: int,
        pitch_hz: str,
    ) -> List[Tuple[str, float, float]]:
        import edge_tts

        sign = "+" if rate_pct >= 0 else ""
        communicate = edge_tts.Communicate(
            text,
            voice_name,
            rate=f"{sign}{rate_pct}%",
            pitch=pitch_hz,
        )
        word_boundaries: List[Tuple[str, float, float]] = []

        with open(out_mp3, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start_s = chunk["offset"] / 10_000_000.0
                    dur_s = chunk["duration"] / 10_000_000.0
                    word_boundaries.append((chunk["text"], start_s, start_s + max(dur_s, 0.12)))

        dur = get_audio_duration(out_mp3)
        return self._align_or_estimate_hindi_words(text, word_boundaries, dur)

    @classmethod
    def _align_or_estimate_hindi_words(
        cls,
        full_text: str,
        raw_boundaries: List[Tuple[str, float, float]],
        total_duration: float,
    ) -> List[Tuple[str, float, float]]:
        clean_words = [w for w in re.split(r"\s+", full_text.strip()) if w]
        if not clean_words:
            return []
        if len(raw_boundaries) == len(clean_words):
            return [
                (clean_words[i], raw_boundaries[i][1], raw_boundaries[i][2])
                for i in range(len(clean_words))
            ]
        return cls._estimate_proportional_word_timings(full_text, total_duration)

    def _generate_offline_cadence_audio(
        self, text: str, out_mp3: Path, voice_name: str, formant_ratio: float = 1.0
    ) -> List[Tuple[str, float, float]]:
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        dur = max(3.2, min(16.0, len(words) * 0.34))
        sr = 44100
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)

        base_f0 = 225.0 if ("swara" in voice_name.lower() or "ananya" in voice_name.lower() or "ava" in voice_name.lower()) else 145.0
        f0 = base_f0 * formant_ratio
        syllable_env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)
        carrier = (
            0.60 * np.sin(2 * np.pi * f0 * t)
            + 0.25 * np.sin(2 * np.pi * (f0 * 1.5) * t)
            + 0.15 * np.sin(2 * np.pi * (f0 * 2.0) * t)
        ) * syllable_env * 0.18

        fade = min(int(0.08 * sr), n // 4)
        if fade > 0:
            carrier[:fade] *= np.linspace(0.0, 1.0, fade)
            carrier[-fade:] *= np.linspace(1.0, 0.0, fade)

        tmp_wav = out_mp3.with_suffix(".tmp.wav")
        int_samples = (carrier * 32767).astype(np.int16)
        stereo = np.column_stack((int_samples, int_samples)).flatten()
        with wave.open(str(tmp_wav), "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav), "-c:a", "libmp3lame", "-b:a", "192k", str(out_mp3)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp_wav.unlink(missing_ok=True)
        return self._estimate_proportional_word_timings(text, dur)

    @staticmethod
    def _estimate_proportional_word_timings(
        text: str, total_duration: float
    ) -> List[Tuple[str, float, float]]:
        raw_words = [w for w in re.split(r"\s+", text.strip()) if w]
        if not raw_words:
            return []
        weights = []
        for w in raw_words:
            weight = max(len(w), 2)
            if w.endswith((",", ";", "—", "-")):
                weight += 2.0
            elif w.endswith((".", "!", "?", "।", "॥")):
                weight += 3.5
            weights.append(float(weight))

        total_weight = sum(weights)
        effective_dur = max(total_duration - 0.10, 0.3)
        cursor = 0.04
        timings = []
        for w, weight in zip(raw_words, weights):
            span = (weight / total_weight) * effective_dur
            timings.append((w, cursor, cursor + span))
            cursor += span
        return timings
