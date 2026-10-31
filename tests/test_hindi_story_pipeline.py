import os
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topic_manager import load_seed_topics
from foundry_llm import FoundryLLMClient
from voice_engine import VoiceEngine, WordTiming
from caption_builder import generate_synced_ass_captions, _clean_hindi_word
from video_composer import VideoComposer
from media_fetcher import MediaFetcher


class HindiStoryPipelineTests(unittest.TestCase):
    def test_gpt_image_1_mini_is_default_primary_model(self):
        old_val = os.environ.pop("FOUNDRY_IMAGE_MODEL", None)
        try:
            mf = MediaFetcher(topic="कहानी #1: बंदर और मगरमच्छ")
            self.assertEqual(mf.image_model, "gpt-image-1-mini")
        finally:
            if old_val is not None:
                os.environ["FOUNDRY_IMAGE_MODEL"] = old_val

    def test_gpt_image_prompt_embeds_spoken_narration_and_locked_character_bible(self):
        bible = {
            "Raktamukha": "single adult golden-brown Rhesus macaque monkey with amber eyes",
            "Karalamukha": "14-foot dark olive-green Indian mugger crocodile",
        }
        mf = MediaFetcher(topic="कहानी #1: बंदर और मगरमच्छ", character_bible=bible)
        scene = {
            "scene_id": 15,
            "speaker_name": "कथावाचक",
            "narration": "निश्छल बंदर मगरमच्छ की पीठ पर बैठ गया और वे गहरे पानी में पहुँच गए।",
            "visual_subject": "Monkey riding on crocodile back in deep river water",
        }
        prompt = mf._build_gpt_image_scene_prompt(
            "Medium tracking shot of Raktamukha riding on Karalamukha's back in the middle of the wide Narmada river",
            scene,
            shot_idx=2,
            total_shots=3,
        )
        self.assertIn("single adult golden-brown Rhesus macaque monkey", prompt)
        self.assertIn("14-foot dark olive-green Indian mugger crocodile", prompt)
        self.assertIn("निश्छल बंदर मगरमच्छ की पीठ पर बैठ गया", prompt)
        self.assertIn("16:9", prompt)

    def test_distinct_character_voices_and_dialogue_auto_inference(self):
        ve = VoiceEngine()
        nar_v, _, _, nar_f, nar_role, _ = ve.resolve_voice_for_scene(
            {"speaker_role": "narrator", "speaker_name": "कथावाचक", "narration": "प्राचीन काल में नर्मदा नदी के तट पर एक पेड़ था।"}
        )
        hero_v, _, _, hero_f, hero_role, _ = ve.resolve_voice_for_scene(
            {"speaker_role": "male_protagonist", "speaker_name": "बंदर रक्तमुख", "narration": "अरे भाई मगरमच्छ! लो ये मीठे जामुन खाओ।"}
        )
        villain_v, _, _, vil_f, vil_role, _ = ve.resolve_voice_for_scene(
            {"speaker_role": "male_antagonist", "speaker_name": "मगरमच्छ करालमुख", "narration": "धन्यवाद भाई! ये फल अमृत जैसे हैं।"}
        )
        female_v, _, _, fem_f, fem_role, fem_gender = ve.resolve_voice_for_scene(
            {"speaker_role": "narrator", "speaker_name": "मगरमच्छ की पत्नी", "narration": "सुना जी! मुझे उस बंदर का कलेजा खाना है!"}
        )

        # Verify automatic inference caught the wife's dialogue even when speaker_role was "narrator"
        self.assertEqual(fem_role, "female_character")
        self.assertEqual(fem_gender, "female")
        self.assertIn("SwaraNeural", female_v)

        # Verify Narrator, Hero, Villain, and Female Character all have distinct voice IDs and/or distinct formant ratios
        self.assertEqual(len({(nar_v, nar_f), (hero_v, hero_f), (villain_v, vil_f), (female_v, fem_f)}), 4)

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
            self.assertIn("Style: StoryCaption,Noto Sans Devanagari,68,", content)
            self.assertIn(",9,48,48,36,1", content)


if __name__ == "__main__":
    unittest.main()
