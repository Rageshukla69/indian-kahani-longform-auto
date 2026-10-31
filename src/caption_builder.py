r"""
Wide 16:9 Devanagari Hindi Storytelling Subtitle Builder (`src/caption_builder.py` — v2.0 Wide Cinema Edition):
- Designed specifically for **Full-Screen 16:9 Widescreen (`1920x1080`) Storytelling**!
- Fixes cramped narrow captions:
  - Uses wide horizontal 16:9 lines (`max_words: 12`, `max_chars_per_clause: 78`, `max_single_line_chars: 64`) spanning smoothly across the widescreen lower third (`MarginL: 110, MarginR: 110, MarginV: 38`).
  - Uses crisp 48pt **Noto Sans Devanagari** with thick dark text outline (`Outline: 4.2, Shadow: 2.2`, `BorderStyle: 1`) so the video needs ZERO ugly black box covering the bottom 20% of the screen!
  - Uses pure Devanagari symbols (`✦`, `•`) instead of color emojis (`🎙️`, `🔔`) that render as broken `?` or `[]` tofu boxes in Linux FFmpeg `libass`.
  - Highlights the active spoken word in **Warm Storytelling Gold (`#FFD700`)** (or **Soft Coral-Rose (`#FF99B6`)** when a female character speaks).
"""
import re
from pathlib import Path
from typing import List
from voice_engine import WordTiming

PUNCTUATION_ONLY_RE = re.compile(r"^[,.;:!?।॥—–\-\"'\)\]]+$")
LEADING_PUNCT_RE = re.compile(r"^([,.;:!?।॥]+)(.+)$")

HINDI_EMPHASIS_WORDS = {
    "पंचतंत्र", "कहानी", "मित्रता", "बुद्धि", "चतुर", "धैर्य", "विश्वासघात", "सीख",
    "शिक्षा", "संकट", "राजा", "रानी", "राजकुमारी", "जंगल", "नदी", "अमृत", "सत्य", "धर्म", "न्याय",
}


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centis = int(round((secs - int(secs)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centis:02d}"


def _clean_hindi_word(word: str) -> str:
    """
    Preserves all Devanagari characters, matras, chandrabindu, anusvara, visarga, halant, and purna viram (`।`),
    stripping only ASS control braces/backslashes and stripping unsupported color emojis that cause `?` boxes.
    """
    cleaned = word.replace("{", "").replace("}", "").replace("\\", "").strip()
    # Remove supplementary-plane color emojis (U+1F000+) that render as tofu boxes in NotoSansDevanagari
    cleaned = re.sub(r"[\U00010000-\U0010ffff\uFE00-\uFE0F]", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?।॥])", r"\1", cleaned)
    return cleaned.strip()


def _get_storytelling_highlight_tag(word_str: str, speaker_gender: str = "male") -> str:
    bare = re.sub(r"[,.;:!?।॥\"'\(\)\[\]]", "", word_str).strip()
    if bare in HINDI_EMPHASIS_WORDS:
        return r"{\c&H00BFFF&\fscx104\fscy104}"
    if speaker_gender == "female":
        return r"{\c&HB699FF&\fscx103\fscy103}"
    return r"{\c&H00D7FF&\fscx103\fscy103}"


def _merge_punctuation_tokens(word_timings: List[WordTiming]) -> List[WordTiming]:
    merged: List[WordTiming] = []
    for wt in word_timings:
        raw = _clean_hindi_word(wt.word)
        if not raw:
            continue

        if PUNCTUATION_ONLY_RE.match(raw) and merged and merged[-1].scene_id == wt.scene_id:
            prev = merged[-1]
            merged[-1] = WordTiming(
                word=f"{prev.word}{raw}",
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, wt.end_sec),
                scene_id=prev.scene_id,
                speaker_role=prev.speaker_role,
                speaker_gender=prev.speaker_gender,
            )
            continue

        m = LEADING_PUNCT_RE.match(raw)
        if m and merged and merged[-1].scene_id == wt.scene_id:
            punct, rest = m.group(1), m.group(2).strip()
            prev = merged[-1]
            merged[-1] = WordTiming(
                word=f"{prev.word}{punct}",
                start_sec=prev.start_sec,
                end_sec=prev.end_sec,
                scene_id=prev.scene_id,
                speaker_role=prev.speaker_role,
                speaker_gender=prev.speaker_gender,
            )
            if rest:
                merged.append(
                    WordTiming(
                        word=rest,
                        start_sec=wt.start_sec,
                        end_sec=wt.end_sec,
                        scene_id=wt.scene_id,
                        speaker_role=wt.speaker_role,
                        speaker_gender=wt.speaker_gender,
                    )
                )
            continue

        merged.append(wt)
    return merged


def _group_into_storytelling_clauses(
    word_timings: List[WordTiming],
    max_words: int = 11,
    max_chars_per_clause: int = 72,
) -> List[List[WordTiming]]:
    """
    Groups Hindi words into wide, comfortable 16:9 storytelling sentences/clauses (up to 11 words or 72 chars)
    so captions stretch naturally across the widescreen bottom bar instead of bunching in a narrow vertical column!
    """
    chunks: List[List[WordTiming]] = []
    i = 0
    n = len(word_timings)

    while i < n:
        scene_id = word_timings[i].scene_id
        chunk: List[WordTiming] = []
        char_len = 0

        while i < n and word_timings[i].scene_id == scene_id and len(chunk) < max_words:
            candidate = _clean_hindi_word(word_timings[i].word)
            cand_len = len(candidate)
            projected = char_len + (1 if chunk else 0) + cand_len

            if chunk and projected > max_chars_per_clause:
                break

            chunk.append(word_timings[i])
            char_len = projected
            i += 1

            if candidate.endswith(("।", "॥", ".", "!", "?", ";", ":")) or (len(chunk) >= 7 and candidate.endswith(",")):
                break

        if chunk:
            chunks.append(chunk)

    return chunks


def _format_storytelling_clause(
    chunk: List[WordTiming],
    active_idx: int,
    max_single_line_chars: int = 56,
) -> str:
    """
    Formats the clause as a wide 1-line widescreen subtitle (up to 56 chars) or a balanced wide 2-line subtitle.
    """
    plain_words = [_clean_hindi_word(wt.word) for wt in chunk]
    speaker_gender = chunk[active_idx].speaker_gender if active_idx < len(chunk) else "male"

    styled_words = []
    for j, w_str in enumerate(plain_words):
        if j == active_idx:
            hl_tag = _get_storytelling_highlight_tag(w_str, speaker_gender=speaker_gender)
            styled_words.append(f"{hl_tag}{w_str}{{\\c&HFFFFFF&\\fscx100\\fscy100}}")
        else:
            styled_words.append(w_str)

    total_chars = sum(len(w) for w in plain_words) + max(0, len(plain_words) - 1)
    prefix = r"{\fad(30,0)}" if active_idx == 0 else ""

    if len(plain_words) <= 9 or total_chars <= max_single_line_chars:
        return prefix + " ".join(styled_words)

    # Split into two wide balanced widescreen lines
    mid = len(plain_words) // 2
    line1 = " ".join(styled_words[:mid])
    line2 = " ".join(styled_words[mid:])
    return f"{prefix}{line1}\\N{line2}"


def generate_synced_ass_captions(
    word_timings: List[WordTiming],
    scenes: List[dict],
    output_ass: Path,
    font_name: str = "Noto Sans Devanagari",
) -> Path:
    """
    Generates a 1920x1080 widescreen `.ass` subtitle file with wide, clean Devanagari storytelling captions
    at the bottom of the 16:9 frame (`MarginV: 38`, `MarginL: 100`, `MarginR: 100`), `Spacing: 0`
    (mandatory for unbroken Devanagari Shirorekha), and zero emoji tofu characters.
    """
    output_ass.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
Title: Indian Hindi Kahani 16:9 Wide Cinema Subtitles (Devanagari Shaped)
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: StoryCaption,{font_name},46,&H00FFFFFF,&H0000D7FF,&H00080808,&HA0000000,-1,0,0,0,100,100,0.0,0,1,4.2,2.2,2,100,100,38,1
Style: SpeakerTag,{font_name},28,&H0000D7FF,&H00FFFFFF,&H00080808,&HB40A0E17,-1,0,0,0,100,100,0.0,0,1,3.2,1.5,2,100,100,118,1
Style: ChapterBadge,{font_name},28,&H0000D7FF,&H00FFFFFF,&H00101010,&HC80C1018,-1,0,0,0,100,100,0.0,0,3,10,0,7,48,48,36,1
Style: SubscribeCTA,{font_name},30,&H00FFFFFF,&H0000D7FF,&H001414CC,&HE61010B0,-1,0,0,0,100,100,0.0,0,3,12,0,8,70,70,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: List[str] = []

    clean_timings = _merge_punctuation_tokens(word_timings)
    chunks = _group_into_storytelling_clauses(clean_timings)

    for chunk in chunks:
        for active_idx, active_wt in enumerate(chunk):
            ev_start = active_wt.start_sec
            if active_idx + 1 < len(chunk):
                ev_end = chunk[active_idx + 1].start_sec
            else:
                ev_end = active_wt.end_sec

            if ev_end <= ev_start:
                ev_end = ev_start + 0.12

            line_text = _format_storytelling_clause(chunk, active_idx)
            events.append(
                f"Dialogue: 2,{_format_ass_time(ev_start)},{_format_ass_time(ev_end)},StoryCaption,,100,100,38,,{line_text}"
            )

    last_badge = ""
    total_end_sec = 0.0
    for scene in scenes:
        s_start = float(scene.get("start_sec", 0.0))
        s_end_val = float(scene.get("end_sec", 0.0))
        if s_end_val > total_end_sec:
            total_end_sec = s_end_val

        # Show clean Hindi speaker label (without English parentheses or unsupported emoji glyphs) when a character speaks
        raw_speaker = _clean_hindi_word(str(scene.get("speaker_name") or ""))
        clean_speaker = re.sub(r"\s*\([^)]*\)", "", raw_speaker).strip()
        speaker_role = str(scene.get("speaker_role") or "narrator").lower()
        if clean_speaker and speaker_role != "narrator" and s_end_val > s_start:
            events.append(
                f"Dialogue: 1,{_format_ass_time(s_start)},{_format_ass_time(s_end_val)},SpeakerTag,,100,100,118,,{{\\fad(80,80)}}✦ {clean_speaker} ✦"
            )

        badge = _clean_hindi_word(str(scene.get("on_screen_callout") or ""))
        if not badge or badge == last_badge:
            continue
        last_badge = badge
        b_start = s_start + 0.05
        b_end = min(s_end_val, b_start + 3.2)
        if b_end > b_start:
            events.append(
                f"Dialogue: 1,{_format_ass_time(b_start)},{_format_ass_time(b_end)},ChapterBadge,,48,48,36,,{{\\fad(120,160)}}  ✦ {badge}  "
            )

    if total_end_sec > 8.0:
        cta_start = max(0.0, total_end_sec - 4.0)
        events.append(
            f"Dialogue: 3,{_format_ass_time(cta_start)},{_format_ass_time(total_end_sec)},SubscribeCTA,,70,70,40,,"
            r"{\fad(150,150)}  ✦ ऐसी और कहानियों के लिए चैनल सब्सक्राइब करें ✦  "
        )

    output_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[CaptionBuilder] Generated {len(events)} Wide 16:9 Devanagari subtitle events -> {output_ass}")
    return output_ass
