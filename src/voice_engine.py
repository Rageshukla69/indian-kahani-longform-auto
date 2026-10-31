"""
Young Male & Young Female Multi-Character Hindi Neural Voiceover Engine (`src/voice_engine.py` — v2.0 Youthful Edition):
- Tuned specifically for **Young, Energetic Indian Storytelling** (like top Hindi YouTube creators & Pocket FM):
  1. **Young Male Narrator (`narrator`)**: `hi-IN-MadhurNeural` (or `hi-IN-ArjunNeural` on Azure) with youthful `+6%` speech rate and `+4Hz` bright pitch so the voice never sounds slow or elderly!
  2. **Young Female Character (`female_character` or `speaker_gender == "female"`)**: `hi-IN-SwaraNeural` (or `hi-IN-KavyaNeural` / `hi-IN-AnanyaNeural` on Azure) with lively `+6%` speech rate and `+5Hz` bright youthful female pitch!
  3. **Young Clever Male Protagonist (`male_protagonist`)**: `hi-IN-MadhurNeural` (`+9%` speech rate, `+8Hz` energetic young hero pitch)
  4. **Dramatic Male Antagonist (`male_antagonist`)**: `hi-IN-MadhurNeural` (`+3%` speech rate, `-2Hz` crisp dramatic tone — clear and youthful, never sluggish)
  5. **Young Child / Small Bird (`young_character`)**: `hi-IN-SwaraNeural` (`+10%` speech rate, `+10Hz` cheerful voice)
- Preserves intact Devanagari words and matras (`ा`, `ि`, `ी`, `ु`, `ू`, `े`, `ै`, `ो`, `ौ`, `ं`, `ँ`, `्`, `।`).
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

DEFAULT_YOUNG_HINDI_MALE_VOICE = "hi-IN-MadhurNeural"
DEFAULT_YOUNG_HINDI_FEMALE_VOICE = "hi-IN-SwaraNeural"

# Character Role -> (env_override_key, default_voice, youthful_rate_pct, youthful_pitch_hz)
CHARACTER_VOICE_PROFILES: Dict[str, Tuple[str, str, int, str]] = {
    "narrator": ("HINDI_NARRATOR_VOICE", DEFAULT_YOUNG_HINDI_MALE_VOICE, 6, "+4Hz"),
    "female_character": ("HINDI_FEMALE_VOICE", DEFAULT_YOUNG_HINDI_FEMALE_VOICE, 6, "+5Hz"),
    "male_protagonist": ("HINDI_MALE_HERO_VOICE", DEFAULT_YOUNG_HINDI_MALE_VOICE, 9, "+8Hz"),
    "male_antagonist": ("HINDI_MALE_DEEP_VOICE", DEFAULT_YOUNG_HINDI_MALE_VOICE, 3, "-2Hz"),
    "wise_elder": ("HINDI_ELDER_VOICE", DEFAULT_YOUNG_HINDI_MALE_VOICE, 4, "+2Hz"),
    "young_character": ("HINDI_CHILD_VOICE", DEFAULT_YOUNG_HINDI_FEMALE_VOICE, 10, "+10Hz"),
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
        self.male_voice = os.getenv("AZURE_TTS_VOICE", DEFAULT_YOUNG_HINDI_MALE_VOICE).strip() or DEFAULT_YOUNG_HINDI_MALE_VOICE
        if not self.male_voice.lower().startswith("hi-in"):
            self.male_voice = DEFAULT_YOUNG_HINDI_MALE_VOICE
        self.female_voice = os.getenv("AZURE_TTS_FEMALE_VOICE", DEFAULT_YOUNG_HINDI_FEMALE_VOICE).strip() or DEFAULT_YOUNG_HINDI_FEMALE_VOICE

        raw_speed = os.getenv("VOICE_SPEED_PERCENT", "0").strip().replace("%", "").replace("+", "")
        try:
            self.base_speed_pct = int(float(raw_speed))
        except ValueError:
            self.base_speed_pct = 0

    def resolve_voice_for_scene(self, scene: dict) -> Tuple[str, int, str]:
        """
        Determines the exact `(voice_name, rate_pct, pitch_hz)` for a scene using young male and young female prosody.
        """
        role = str(scene.get("speaker_role") or "narrator").strip().lower()
        gender = str(scene.get("speaker_gender") or "").strip().lower()
        speaker_name = str(scene.get("speaker_name") or "").strip().lower()

        female_indicators = (
            "female", "wife", "queen", "princess", "mother", "पत्नी", "रानी",
            "राजकुमारी", "माँ", "बंदरिया", "ब्राह्मणी", "महिला", "देवी", "चुहिया", "लोमड़ी", "भाभी"
        )
        if gender == "female" or role == "female_character" or any(tok in speaker_name for tok in female_indicators):
            voice = os.getenv("HINDI_FEMALE_VOICE", self.female_voice).strip() or DEFAULT_YOUNG_HINDI_FEMALE_VOICE
            return voice, self.base_speed_pct + 6, "+5Hz"

        env_key, default_v, rate_delta, pitch_hz = CHARACTER_VOICE_PROFILES.get(
            role, CHARACTER_VOICE_PROFILES["narrator"]
        )
        fallback_v = self.female_voice if default_v == DEFAULT_YOUNG_HINDI_FEMALE_VOICE else self.male_voice
        voice = os.getenv(env_key, fallback_v).strip() or fallback_v
        return voice, self.base_speed_pct + rate_delta, pitch_hz

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
            voice_name, rate_pct, pitch_hz = self.resolve_voice_for_scene(scene)
            role = str(scene.get("speaker_role") or "narrator")
            gender = "female" if voice_name == self.female_voice or "swara" in voice_name.lower() or "kavya" in voice_name.lower() else "male"

            scene_mp3 = output_dir / f"scene_{scene_id:02d}.mp3"
            words_with_rel_times = self._synthesize_single_clip(
                text=text,
                out_mp3=scene_mp3,
                voice_name=voice_name,
                rate_pct=rate_pct,
                pitch_hz=pitch_hz,
            )
            clip_dur = get_audio_duration(scene_mp3)

            scene_words = []
            for w_text, w_start, w_end in words_with_rel_times:
                wt = WordTiming(
                    word=w_text,
                    start_sec=round(current_offset + w_start, 3),
                    end_sec=round(current_offset + w_end, 3),
                    scene_id=scene_id,
                    speaker_role=role,
                    speaker_gender=gender,
                )
                all_word_timings.append(wt)
                scene_words.append(asdict(wt))

            scene_copy = dict(scene)
            scene_copy["voice_used"] = voice_name
            scene_copy["voice_rate_pct"] = rate_pct
            scene_copy["voice_pitch_hz"] = pitch_hz
            scene_copy["speaker_gender"] = gender
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
        voices_summary = sorted({f"{s['voice_used']} ({s['voice_pitch_hz']})" for s in enriched_scenes})
        print(
            f"[VoiceEngine] Synthesized {len(enriched_scenes)} Hindi scenes with Young Voices {voices_summary} "
            f"({current_offset:.1f}s total, {len(all_word_timings)} synced Devanagari words)."
        )
        return master_vo, all_word_timings, enriched_scenes

    def _synthesize_single_clip(
        self,
        text: str,
        out_mp3: Path,
        voice_name: str,
        rate_pct: int,
        pitch_hz: str,
    ) -> List[Tuple[str, float, float]]:
        if self.speech_key and self.speech_region:
            try:
                return self._call_azure_speech_sdk(text, out_mp3, voice_name, rate_pct, pitch_hz)
            except Exception as exc:
                print(f"[VoiceEngine] Azure Speech SDK notice ({exc}); falling back to Edge Neural Hindi TTS...")

        try:
            return asyncio.run(self._call_edge_tts_with_boundaries(text, out_mp3, voice_name, rate_pct, pitch_hz))
        except Exception as exc:
            print(f"[VoiceEngine] Edge TTS network notice ({exc}); generating offline Hindi cadence audio for test/dry-run...")
            return self._generate_offline_cadence_audio(text, out_mp3, voice_name)

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
            raise RuntimeError(f"Azure Speech synthesis failed: {result.reason}")

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
        self, text: str, out_mp3: Path, voice_name: str
    ) -> List[Tuple[str, float, float]]:
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        dur = max(3.2, min(16.0, len(words) * 0.34))
        sr = 44100
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)

        f0 = 225.0 if ("swara" in voice_name.lower() or "female" in voice_name.lower()) else 150.0
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
