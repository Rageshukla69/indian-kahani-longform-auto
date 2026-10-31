r"""
Devanagari Hindi Storytelling Subtitle Builder (`src/caption_builder.py`):
- Designed specifically for **16:9 Long-Form Indian YouTube Storytelling (`1920x1080`)**!
- Uses **Noto Sans Devanagari** (`Fontsize: 44`, `Spacing: 0` so Devanagari Shirorekha/headline bar and matras connect seamlessly without broken glyphs!).
- Positioned cleanly in the **bottom storytelling subtitle bar (`Alignment: 2`, `MarginV: 58`)** with a subtle translucent dark cinema pill/outline so viewers can enjoy the full 16:9 visuals while reading every Hindi word clearly.
- Highlights the currently spoken Hindi word in **Warm Storytelling Gold (`#FFD700` -> ASS `&H00D7FF&`)** (or **Soft Rose Gold (`#FFB6C1`)** when a female character speaks) while keeping surrounding words in crisp cream-white (`#FFFFFF`), matching the classic Indian YouTuber storytelling style (Pocket FM / Kahaniyan / Moral Story channels).
- Displays a small, unobtrusive top-left Story Number & Scene Chapter badge (`कहानी #1 • पंचतंत्र`) at scene transitions.
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
    stripping only ASS control braces/backslashes. Never calls `.upper()` on Devanagari text.
    """
    cleaned = word.replace("{", "").replace("}", "").replace("\\", "").strip()
    cleaned = re.sub(r"\s+([,.;:!?।॥])", r"\1", cleaned)
    return cleaned


def _get_storytelling_highlight_tag(word_str: str, speaker_gender: str = "male") -> str:
    """
    Returns a gentle, legible storytelling word highlight tag without distorting Devanagari shirorekha:
    - Important moral/story words glow in Warm Saffron Gold (`&H00A5FF&` / `#FFA500`)
    - Female character active words glow in Soft Lotus Gold (`&H80E6FF&`)
    - Standard active words glow in Classic Storytelling Yellow-Gold (`&H00D7FF&` / `#FFD700`)
    """
    bare = re.sub(r"[,.;:!?।॥\"'\(\)\[\]]", "", word_str).strip()
    if bare in HINDI_EMPHASIS_WORDS:
        return r"{\c&H00BFFF&\fscx104\fscy104}"
    if speaker_gender == "female":
        return r"{\c&H66E0FF&\fscx103\fscy103}"
    return r"{\c&H00D7FF&\fscx103\fscy103}"


def _merge_punctuation_tokens(word_timings: List[WordTiming]) -> List[WordTiming]:
    merged: List[WordTiming] = []
    for wt in word_timings:
        raw = wt.word.strip()
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
    max_words: int = 8,
    max_chars_per_clause: int = 48,
) -> List[List[WordTiming]]:
    """
    Groups Hindi words into natural, small 1-to-2 line storytelling clauses (6–8 words max)
    breaking cleanly on Hindi Purna Viram (`।`), question/exclamation marks, or commas.
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

            if candidate.endswith(("।", "॥", ".", "!", "?", ";", ":")) or (len(chunk) >= 5 and candidate.endswith(",")):
                break

        if chunk:
            chunks.append(chunk)

    return chunks


def _format_storytelling_clause(
    chunk: List[WordTiming],
    active_idx: int,
    max_single_line_chars: int = 34,
) -> str:
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
    prefix = r"{\fad(35,0)}" if active_idx == 0 else ""

    if len(plain_words) <= 5 or total_chars <= max_single_line_chars:
        return prefix + " ".join(styled_words)

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
    Generates a 1920x1080 widescreen `.ass` subtitle file with small, clean Devanagari storytelling captions
    at the bottom of the frame (`MarginV: 58`), `Spacing: 0` (mandatory for unbroken Devanagari Shirorekha),
    and speaker/chapter callout badges.
    """
    output_ass.parent.mkdir(parents=True, exist_ok=True)

    # Note: Spacing MUST be 0 for Devanagari so libass/HarfBuzz shapes conjuncts & matras properly!
    header = f"""[Script Info]
Title: Indian Hindi Kahani 16:9 Storytelling Subtitles (Devanagari Shaped)
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: StoryCaption,{font_name},42,&H00FFFFFF,&H0000D7FF,&H00121212,&HB408080C,-1,0,0,0,100,100,0.0,0,1,3.5,2.0,2,120,120,56,1
Style: SpeakerTag,{font_name},26,&H0000D7FF,&H00FFFFFF,&H00121212,&HC80E121A,-1,0,0,0,100,100,0.0,0,3,8,0,2,120,120,132,1
Style: ChapterBadge,{font_name},28,&H0000D7FF,&H00FFFFFF,&H00101010,&HD20C1018,-1,0,0,0,100,100,0.0,0,3,10,0,7,55,55,42,1
Style: SubscribeCTA,{font_name},30,&H00FFFFFF,&H0000D7FF,&H001414CC,&HE61010B0,-1,0,0,0,100,100,0.0,0,3,12,0,8,70,70,45,1

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
                f"Dialogue: 2,{_format_ass_time(ev_start)},{_format_ass_time(ev_end)},StoryCaption,,120,120,56,,{line_text}"
            )

    last_badge = ""
    total_end_sec = 0.0
    for scene in scenes:
        s_start = float(scene.get("start_sec", 0.0))
        s_end_val = float(scene.get("end_sec", 0.0))
        if s_end_val > total_end_sec:
            total_end_sec = s_end_val

        # Show subtle speaker name pill when a character other than the narrator speaks
        speaker_name = _clean_hindi_word(str(scene.get("speaker_name") or ""))
        speaker_role = str(scene.get("speaker_role") or "narrator").lower()
        if speaker_name and speaker_role != "narrator" and s_end_val > s_start:
            events.append(
                f"Dialogue: 1,{_format_ass_time(s_start)},{_format_ass_time(s_end_val)},SpeakerTag,,120,120,132,,{{\\fad(80,80)}}🎙️ {speaker_name}"
            )

        badge = _clean_hindi_word(str(scene.get("on_screen_callout") or ""))
        if not badge or badge == last_badge:
            continue
        last_badge = badge
        b_start = s_start + 0.05
        b_end = min(s_end_val, b_start + 3.2)
        if b_end > b_start:
            events.append(
                f"Dialogue: 1,{_format_ass_time(b_start)},{_format_ass_time(b_end)},ChapterBadge,,55,55,42,,{{\\fad(120,160)}}  {badge}  "
            )

    if total_end_sec > 8.0:
        cta_start = max(0.0, total_end_sec - 3.5)
        events.append(
            f"Dialogue: 3,{_format_ass_time(cta_start)},{_format_ass_time(total_end_sec)},SubscribeCTA,,70,70,45,,"
            r"{\fad(150,150)}  ऐसी और कहानियों के लिए चैनल सब्सक्राइब करें 🔔  "
        )

    output_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    print(f"[CaptionBuilder] Generated {len(events)} 16:9 Devanagari storytelling subtitle events -> {output_ass}")
    return output_ass
