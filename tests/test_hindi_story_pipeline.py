import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topic_manager import (
    extract_story_number,
    format_numbered_topic,
    is_topic_already_used,
    load_seed_topics,
)
from voice_engine import VoiceEngine, WordTiming
from caption_builder import generate_synced_ass_captions, _clean_hindi_word
from youtube_uploader import build_youtube_upload_payload
from media_fetcher import MediaFetcher


class HindiStoryPipelineTests(unittest.TestCase):
    def test_seed_contains_101_panchatantra_plus_expansion(self):
        seeds = load_seed_topics()
        self.assertGreaterEqual(len(seeds), 110)
        self.assertTrue(seeds[0].startswith("कहानी #1: Panchatantra"))
        self.assertTrue(seeds[100].startswith("कहानी #101: Panchatantra"))
        self.assertIn("Hitopadesha", seeds[101])

    def test_anti_repetition_blocks_duplicates_and_reworded_topics(self):
        used = ["कहानी #1: Panchatantra — बंदर और मगरमच्छ (The Monkey and the Crocodile - Raktamukha & Karalamukha)"]
        self.assertTrue(is_topic_already_used("कहानी #99: बंदर और मगरमच्छ (The Monkey and the Crocodile)", used))
        self.assertFalse(is_topic_already_used("कहानी #2: बातूनी कछुआ और दो हंस (The Talkative Tortoise)", used))

    def test_story_numbering_format(self):
        self.assertEqual(extract_story_number("कहानी #42: नाई और भिक्षुओं की कहानी"), 42)
        self.assertEqual(
            format_numbered_topic("कहानी #5: ब्राह्मण और नेवला", 5),
            "कहानी #5: ब्राह्मण और नेवला",
        )

    def test_multi_character_voice_switching_male_and_female(self):
        ve = VoiceEngine()
        narrator_v, _, _ = ve.resolve_voice_for_scene({"speaker_role": "narrator", "speaker_gender": "male"})
        female_v, _, _ = ve.resolve_voice_for_scene({"speaker_role": "female_character", "speaker_gender": "female"})
        wife_v, _, _ = ve.resolve_voice_for_scene({"speaker_name": "मगरमच्छ की पत्नी", "speaker_role": "narrator"})
        hero_v, hero_rate, hero_pitch = ve.resolve_voice_for_scene({"speaker_role": "male_protagonist", "speaker_gender": "male"})
        villain_v, villain_rate, villain_pitch = ve.resolve_voice_for_scene({"speaker_role": "male_antagonist", "speaker_gender": "male"})

        self.assertIn("MadhurNeural", narrator_v)
        self.assertIn("SwaraNeural", female_v)
        self.assertIn("SwaraNeural", wife_v)
        self.assertNotEqual(hero_pitch, villain_pitch)
        self.assertNotEqual(hero_rate, villain_rate)

    def test_devanagari_caption_builder_preserves_matras_and_zero_spacing(self):
        word = "मगरमच्छ"
        self.assertEqual(_clean_hindi_word(word), "मगरमच्छ")
        timings = [
            WordTiming(word="नमस्कार", start_sec=0.0, end_sec=0.4, scene_id=1, speaker_role="narrator", speaker_gender="male"),
            WordTiming(word="दोस्तों!", start_sec=0.4, end_sec=0.9, scene_id=1, speaker_role="narrator", speaker_gender="male"),
        ]
        scenes = [{"scene_id": 1, "start_sec": 0.0, "end_sec": 1.0, "on_screen_callout": "कहानी #1 • पंचतंत्र", "speaker_name": "कथावाचक", "speaker_role": "narrator"}]
        with tempfile.TemporaryDirectory() as tmp:
            ass_file = Path(tmp) / "test.ass"
            generate_synced_ass_captions(timings, scenes, ass_file)
            content = ass_file.read_text(encoding="utf-8")
            self.assertIn("PlayResX: 1920", content)
            self.assertIn("PlayResY: 1080", content)
            self.assertIn("Noto Sans Devanagari", content)
            self.assertIn("नमस्कार", content)

    def test_youtube_ai_disclosure_and_tags_present(self):
        payload = build_youtube_upload_payload(
            title="कहानी #1: बंदर और मगरमच्छ",
            description="पंचतंत्र की कहानी",
            tags=["panchatantra"],
            story_number=1,
            moral_hindi="बुद्धि ही बल है।",
        )
        self.assertTrue(payload["status"]["containsSyntheticMedia"])
        self.assertTrue(payload["status"]["selfDeclaredAsModifiedContent"])
        self.assertIn("AIGenerated", payload["snippet"]["tags"])
        self.assertIn("AIStory", payload["snippet"]["tags"])
        self.assertEqual(payload["snippet"]["defaultLanguage"], "hi")
        self.assertIn("#AIGenerated", payload["snippet"]["description"])

    def test_storybook_16x9_fallback_plate_renders_1920x1080(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plate.jpg"
            MediaFetcher._create_indian_storybook_illustration(
                out,
                scene={"on_screen_callout": "कहानी #1 • पंचतंत्र", "visual_subject": "Monkey and Crocodile"},
                scene_id=1,
            )
            self.assertTrue(out.exists())
            from PIL import Image
            with Image.open(out) as im:
                self.assertEqual(im.size, (1920, 1080))


if __name__ == "__main__":
    unittest.main()
