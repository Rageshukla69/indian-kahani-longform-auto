"""
Main End-to-End Orchestrator for the Indian Hindi Long-Form Storytelling YouTube Channel (`src/main.py`):
- Target #1: **Panchatantra ki 101 Kahaniyan** (`कहानी #1` to `कहानी #101`), followed automatically by multi-resource
  Indian stories (**Hitopadesha, Jataka Tales, Akbar-Birbal, Tenali Rama, Vikram-Betaal, Kathasaritsagara**).
- Guarantees **Zero Topic Repetition** and strict sequential **Story Numbering**.
- Generates **Multi-Character Hindi Voiceover** (automatically switching between Narrator, Male Hero, Deep Male Antagonist,
  and Female Character `hi-IN-SwaraNeural` voices).
- Generates **16:9 Widescreen (`1920x1080`) Scene-Synced AI Illustrations** via `saurav-z/free-image-generation-api`.
- Renders **Small Devanagari Hindi Subtitles** in classic Indian YouTuber storytelling format (`Noto Sans Devanagari`, `Spacing: 0`).
- Publishes with **YouTube AI Altered/Synthetic Content disclosure & `#AIGenerated` tags**.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs): return False

from sfx_generator import generate_all_sfx
from topic_manager import select_next_topic_with_number, mark_topic_completed
from foundry_llm import FoundryLLMClient
from voice_engine import VoiceEngine
from media_fetcher import MediaFetcher
from caption_builder import generate_synced_ass_captions
from video_composer import VideoComposer
from youtube_uploader import upload_video_to_youtube, build_youtube_upload_payload
from quality_control import validate_rendered_video

HISTORY_FILE = Path("data/upload_history.json")


def _should_skip_backup_cron_slot(min_hours_between_uploads: float = 8.0) -> bool:
    event_name = os.getenv("GITHUB_EVENT_NAME", "").strip()
    if event_name != "schedule":
        return False

    if not HISTORY_FILE.exists():
        return False

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        uploads = data.get("uploads", [])
        if not uploads:
            return False
        last_ts_raw = uploads[-1].get("timestamp_utc", "")
        if not last_ts_raw:
            return False
        last_dt = datetime.fromisoformat(last_ts_raw)
        elapsed_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
        if elapsed_hours < min_hours_between_uploads:
            print(
                f"[Pipeline] Backup cron slot guard: Last Kahani was uploaded {elapsed_hours:.2f}h ago "
                f"(< {min_hours_between_uploads}h threshold). Skipping duplicate backup slot cleanly."
            )
            return True
    except Exception as exc:
        print(f"[Pipeline] Timestamp check notice: {exc}")
    return False


def _write_github_step_summary(
    story_number: int,
    collection_source: str,
    topic: str,
    title: str,
    scenes: list,
    fetcher: MediaFetcher,
    video_id: str,
):
    lines = [
        f"## 🪔 Indian Hindi Kahani Run Summary — `{title}`",
        f"- **Story Number**: `कहानी #{story_number}`",
        f"- **Collection / Resource**: `{collection_source}`",
        f"- **Story Topic**: {topic}",
        f"- **Aspect Ratio**: `16:9 Widescreen (1920x1080)`",
        f"- **YouTube Video ID**: `{video_id}` (AI Tag & Synthetic Disclosure: `Enabled`)",
        "",
        "### 📊 Visual & Multi-Character Voice Breakdown",
        "| Metric | Value | Cost Impact |",
        "| :--- | :--- | :--- |",
        f"| **🎨 Primary `gpt-image-1-mini` Widescreen Shots** | **`{fetcher.gpt_images_used}` shots (`1536x1024` -> `1920x1080`)** | **Default Primary (`{fetcher.image_model}`)** |",
        f"| **⚡ Fallback Cloudflare Worker Shots** | **`{fetcher.free_worker_images_used}` shots** | **Fallback** |",
        f"| **🖼️ Total AI / Storybook Scene Plates** | **`{len(scenes)}` scenes (16:9)** | **₹0 / $0.00** |",
        f"| **🎙️ Hindi Voices Used** | **`{', '.join(sorted({s.get('voice_used', '') for s in scenes}))}`** | **Multi-Character** |",
        "",
        "### 🎞️ Scene-by-Scene Audio, Voice & Visual Audit",
        "| Scene | Speaker & Voice | Media Type | Hindi Narration (Devanagari) |",
        "| :---: | :--- | :--- | :--- |",
    ]
    for idx, scene in enumerate(scenes):
        audit_kind = fetcher.scene_audit_log[idx]["kind"] if idx < len(fetcher.scene_audit_log) else "16:9 SCENE"
        sp_label = f"{scene.get('speaker_name', 'कथावाचक')} (`{scene.get('voice_used', '')}`)"
        lines.append(
            f"| **{scene.get('scene_id', idx+1):02d}** | {sp_label} | `{audit_kind}` | {scene.get('narration', '')} |"
        )
    summary_md = "\n".join(lines) + "\n"
    print("\n" + "=" * 84)
    print(summary_md)
    print("=" * 84 + "\n")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "").strip()
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as sf:
                sf.write(summary_md)
        except Exception as exc:
            print(f"[Pipeline] GITHUB_STEP_SUMMARY write notice: {exc}")


def run_pipeline():
    load_dotenv()
    if _should_skip_backup_cron_slot():
        return

    temp_dir = Path("temp")
    output_dir = Path("output")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (Path("assets/sfx/kahani_bgm.wav")).exists() or not (Path("assets/sfx/hook_impact.wav")).exists():
        generate_all_sfx()

    llm = FoundryLLMClient()
    topic, story_number, collection_source = select_next_topic_with_number(llm_client=llm)

    # Step 1: Generate Multi-Character Hindi Storyboard (Devanagari script + 16:9 scene prompts)
    storyboard = llm.generate_storyboard(topic, story_number=story_number, collection_source=collection_source)
    scenes = storyboard.get("scenes", [])
    if not scenes:
        raise RuntimeError("Storyboard contained no scenes.")

    # Step 2: Synthesize Multi-Character Hindi Voiceover (switches male/female/narrator voices per character)
    voice_engine = VoiceEngine()
    master_vo, word_timings, enriched_scenes = voice_engine.synthesize_scenes(
        scenes, temp_dir / "audio"
    )

    # Step 3: Generate 16:9 Scene-Synced AI Illustrations via `saurav-z/free-image-generation-api`
    fetcher = MediaFetcher(
        topic=topic,
        wikipedia_articles=storyboard.get("wikipedia_articles", ["Panchatantra"]),
        llm_client=llm,
        character_bible=storyboard.get("character_bible") or {},
    )
    for scene in enriched_scenes:
        asset_path, asset_type = fetcher.fetch_scene_asset(scene, temp_dir / "media")
        scene["asset_path"] = str(asset_path)
        scene["asset_type"] = asset_type
        scene["asset_kind"] = fetcher.scene_audit_log[-1].get("kind", "16:9 ILLUSTRATION") if fetcher.scene_audit_log else "16:9 ILLUSTRATION"

    # Step 4: Build Small Devanagari Storytelling Subtitles (.ass)
    ass_path = output_dir / "subtitles.ass"
    generate_synced_ass_captions(word_timings, enriched_scenes, ass_path)

    # Step 5: Render 1920x1080 (16:9) Long-Form MP4 Video
    composer = VideoComposer()
    final_mp4 = output_dir / "final_story.mp4"
    composer.render_story_video(enriched_scenes, master_vo, ass_path, final_mp4, temp_dir / "segments")

    # Step 6: Technical QC (verifies 1920x1080 16:9 resolution, audio track, and zero black frames)
    qc = validate_rendered_video(final_mp4, enriched_scenes, output_dir / "quality_report.json")
    print(f"[QualityControl] PASS: {qc}")

    yt_payload_preview = build_youtube_upload_payload(
        title=storyboard.get("title", topic),
        description=storyboard.get("description", topic),
        tags=storyboard.get("tags", []),
        story_number=story_number,
        moral_hindi=storyboard.get("moral_hindi", ""),
        attributions=fetcher.attributions,
        sources=storyboard.get("sources", []),
    )

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "story_number": story_number,
                "collection_source": collection_source,
                "topic": topic,
                "title": storyboard.get("title"),
                "description": storyboard.get("description"),
                "moral_hindi": storyboard.get("moral_hindi"),
                "aspect_ratio": "16:9 (1920x1080)",
                "youtube_ai_disclosure": yt_payload_preview["status"],
                "youtube_tags": yt_payload_preview["snippet"]["tags"],
                "free_worker_images_used": fetcher.free_worker_images_used,
                "ai_images_used": fetcher.ai_images_used,
                "scene_audit_log": fetcher.scene_audit_log,
                "scenes": enriched_scenes,
                "sources": storyboard.get("sources", []),
                "quality_control": qc,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Step 7: Upload to YouTube with AI Content Tag & Synthetic Media Disclosure
    video_id = upload_video_to_youtube(
        video_path=final_mp4,
        title=storyboard.get("title", topic),
        description=storyboard.get("description", topic),
        tags=storyboard.get("tags", []),
        attributions=fetcher.attributions,
        sources=storyboard.get("sources", []),
        story_number=story_number,
        moral_hindi=storyboard.get("moral_hindi", ""),
    )

    # Step 8: Write Run Summary to GitHub Actions
    _write_github_step_summary(
        story_number=story_number,
        collection_source=collection_source,
        topic=topic,
        title=storyboard.get("title", topic),
        scenes=enriched_scenes,
        fetcher=fetcher,
        video_id=video_id,
    )

    # Step 9: Mark Story Completed so it NEVER repeats
    if video_id != "dry_run_id":
        mark_topic_completed(topic, video_id=video_id, title=storyboard.get("title", topic), story_number=story_number)
    print(f"[Pipeline] Completed Hindi Kahani #{story_number}: '{topic}' (video_id={video_id})")


if __name__ == "__main__":
    run_pipeline()
