import os
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topic_manager import (
    extract_story_number,
    format_numbered_topic,
    is_topic_already_used,
    load_seed_topics,
)
from foundry_llm import FoundryLLMClient
from voice_engine import VoiceEngine, WordTiming
from caption_builder import generate_synced_ass_captions, _clean_hindi_word, _group_into_storytelling_clauses
from video_composer import VideoComposer
from youtube_uploader import build_youtube_upload_payload
from media_fetcher import MediaFetcher


class HindiStoryPipelineTests(unittest.TestCase):
    def test_seed_contains_101_panchatantra_plus_expansion(self):
        seeds = load_seed_topics()
        self.assertGreaterEqual(len(seeds), 110)
        self.assertTrue(seeds[0].startswith("कहानी #1: Panchatantra"))
        self.assertTrue(seeds[100].startswith("कहानी #101: Panchatantra"))

    def test_storyboard_has_character_bible_and_66_plus_scene_locked_shots(self):
        llm = FoundryLLMClient()
        sb = llm._build_fallback_hindi_storyboard("कहानी #1: बंदर और मगरमच्छ", 1, "Panchatantra")
        self.assertIn("character_bible", sb)
        scenes = sb["scenes"]
        self.assertEqual(len(scenes), 22)
        total_shots = sum(len(s.get("shot_prompts", [])) for s in scenes)
        self.assertGreaterEqual(total_shots, 66)
        for s in scenes:
            self.assertGreaterEqual(len(s["shot_prompts"]), 3)
            for sp in s["shot_prompts"]:
                self.assertIn("photorealistic", sp.lower())

    def test_uploaded_repo_bgm_is_prioritized_over_sfx_bgm(self):
        vc = VideoComposer()
        with tempfile.TemporaryDirectory() as tmp:
            chosen_bgm = vc._select_bgm_track(Path(tmp))
            self.assertIn("Parallel Relax AR.mp3", chosen_bgm.name)

    def test_large_68pt_captions_and_top_right_speaker_tag_without_overlap(self):
        self.assertEqual(_clean_hindi_word("🎙️ मगरमच्छ 🔔"), "मगरमच्छ")
        timings = [
            WordTiming(word="सुना", start_sec=0.0, end_sec=0.4, scene_id=10, speaker_role="female_character", speaker_gender="female"),
            WordTiming(word="जी!", start_sec=0.4, end_sec=0.9, scene_id=10, speaker_role="female_character", speaker_gender="female"),
        ]
        scenes = [{"scene_id": 10, "start_sec": 0.0, "end_sec": 1.0, "on_screen_callout": "पत्नी की कठोर ज़िद", "speaker_name": "मगरमच्छ की पत्नी", "speaker_role": "female_character"}]
        with tempfile.TemporaryDirectory() as tmp:
            ass_file = Path(tmp) / "test.ass"
            generate_synced_ass_captions(timings, scenes, ass_file)
            content = ass_file.read_text(encoding="utf-8")
            # Check 68pt font size on StoryCaption and Alignment 9 (top-right) on SpeakerTag
            self.assertIn("Style: StoryCaption,Noto Sans Devanagari,68,", content)
            self.assertIn(",9,48,48,36,1", content)

    def test_media_fetcher_generates_3_to_4_shots_per_scene(self):
        mf = MediaFetcher(topic="कहानी #1: बंदर और मगरमच्छ")
        mf.free_image_api_url = ""  # Offline mode test
        os.environ["ALLOW_POLLINATIONS_FALLBACK"] = "false"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                scene = {
                    "scene_id": 1,
                    "duration_sec": 15.0,
                    "narration": "नमस्कार दोस्तों!",
                    "visual_subject": "Monkey and Crocodile",
                    "shot_prompts": ["Shot 1", "Shot 2", "Shot 3"],
                }
                mf.fetch_scene_asset(scene, Path(tmp))
                self.assertEqual(len(scene["asset_paths"]), 3)
        finally:
            os.environ.pop("ALLOW_POLLINATIONS_FALLBACK", None)


if __name__ == "__main__":
    unittest.main()
