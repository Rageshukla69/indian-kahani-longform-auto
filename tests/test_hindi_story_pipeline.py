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

    def test_longform_storyboard_has_22_scenes_and_photorealistic_dual_prompts(self):
        llm = FoundryLLMClient()
        sb = llm._build_fallback_hindi_storyboard("कहानी #1: बंदर और मगरमच्छ", 1, "Panchatantra")
        scenes = sb["scenes"]
        self.assertGreaterEqual(len(scenes), 22)
        for s in scenes:
            self.assertIn("photorealistic", s["image_prompt"].lower())
            self.assertIn("photorealistic", s["image_prompt_b"].lower())
            self.assertNotIn("cartoon", s["image_prompt"].lower().split("not cartoon")[0])

    def test_square_flux_image_converts_to_full_bleed_1920x1080_without_pillarboxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            square_img = Image.new("RGB", (1024, 1024), color=(60, 120, 90))
            out_path = Path(tmp) / "full_16x9.jpg"
            MediaFetcher._save_as_full_bleed_16x9(square_img, out_path, vertical_anchor=0.38)
            with Image.open(out_path) as result_im:
                self.assertEqual(result_im.size, (1920, 1080))

    def test_young_male_and_female_voice_switching(self):
        ve = VoiceEngine()
        narrator_v, nar_rate, nar_pitch = ve.resolve_voice_for_scene({"speaker_role": "narrator", "speaker_gender": "male"})
        female_v, fem_rate, fem_pitch = ve.resolve_voice_for_scene({"speaker_role": "female_character", "speaker_gender": "female"})
        hero_v, hero_rate, hero_pitch = ve.resolve_voice_for_scene({"speaker_role": "male_protagonist", "speaker_gender": "male"})

        self.assertIn("MadhurNeural", narrator_v)
        self.assertGreaterEqual(nar_rate, 5)
        self.assertIn("+", nar_pitch)
        self.assertIn("SwaraNeural", female_v)
        self.assertGreaterEqual(fem_rate, 5)
        self.assertGreater(hero_rate, nar_rate)

    def test_wide_16x9_devanagari_captions_and_zero_emoji_tofu(self):
        self.assertEqual(_clean_hindi_word("🎙️ मगरमच्छ 🔔"), "मगरमच्छ")
        words = [
            WordTiming(word=w, start_sec=i * 0.4, end_sec=(i + 1) * 0.4, scene_id=1)
            for i, w in enumerate(["नर्मदा", "नदी", "के", "किनारे", "जामुन", "के", "विशाल", "पेड़", "पर", "रक्तमुख", "बंदर", "रहता", "था।"])
        ]
        clauses = _group_into_storytelling_clauses(words)
        # Should group up to 11 words into the first wide 16:9 clause rather than breaking after 4 words
        self.assertGreaterEqual(len(clauses[0]), 9)

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


if __name__ == "__main__":
    unittest.main()
