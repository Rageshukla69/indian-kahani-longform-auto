"""
Hindi Long-Form Multi-Character Storyboard Architect (`src/foundry_llm.py`):
- Generates engaging, family-friendly (kids & adults) long-form Hindi stories in pure, natural Devanagari Hindi (`देवनागरी हिन्दी`).
- Supports Microsoft Foundry (`FOUNDRY_TEXT_ENDPOINT`/`FOUNDRY_TEXT_API_KEY`), OpenAI-compatible endpoints, or built-in rich Hindi multi-character story synthesis when running offline/dry-run without keys.
- Assigns every scene or dialogue turn an explicit character role & gender (`narrator`, `male_1`, `male_2`, `female_1`, `female_2`, `wise_elder`, `child`) so `VoiceEngine` automatically switches Hindi voices when a male, female, or narrator character speaks!
- Generates detailed English 16:9 visual prompts (`image_prompt`) for every scene so the Free AI Image Generation API (`saurav-z/free-image-generation-api`) renders visuals matching the exact spoken audio scene.
"""
import json
import os
import re
from typing import Any, Dict, List
import requests
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap
    def stop_after_attempt(*args, **kwargs): return None
    def wait_exponential(*args, **kwargs): return None

SYSTEM_PROMPT = """You are the master Hindi Storyteller (कथावाचक) and Visual Director for India's #1 Long-Form Hindi Kahani YouTube Channel (inspired by Panchatantra, Hitopadesha, Jataka Tales, Akbar-Birbal, Tenali Rama, and Vikram-Betaal).
Write an engaging, heartwarming, family-friendly (suitable for both kids and adults) long-form Hindi story script in 12 to 18 rich scenes.

STORYTELLING & LANGUAGE RULES:
- Write all spoken `narration` in natural, clear, conversational Hindi using proper **Devanagari script (देवनागरी लिपि)** — warm, expressive, and easy for both children and adults to enjoy.
- Number the story in the title (e.g. "कहानी #1: बंदर और मगरमच्छ | Panchatantra Moral Story in Hindi").
- Use multi-character dialogue! Split scenes or dialogue beats so that when the Narrator (कथावाचक) speaks vs. when a Male character (राजा, बंदर, मगरमच्छ, ब्राह्मण, शेर) or a Female character (रानी, राजकुमारी, बंदरिया, मगरमच्छ की पत्नी, ब्राह्मणी, चिड़िया) speaks, the `speaker_role` and `speaker_gender` reflect that character accurately.
- Allowed `speaker_role` values:
  - `"narrator"` (कथावाचक — warm master storyteller)
  - `"male_protagonist"` (मुख्य पुरुष पात्र — energetic/clever male voice)
  - `"male_antagonist"` (खूँखार/भारी पुरुष पात्र — deep dramatic male voice)
  - `"female_character"` (महिला पात्र — expressive female Hindi voice)
  - `"wise_elder"` (बुद्धिमान गुरु/राजा/साधु — calm, resonant voice)
  - `"young_character"` (बाल पात्र / नन्हा जीव — cheerful bright voice)
- Allowed `speaker_gender` values: `"male"`, `"female"`. Whenever ANY female character (wife, queen, mother, goddess, female bird/animal) speaks, you MUST set `"speaker_gender": "female"` and `"speaker_role": "female_character"`.
- End the final scene with the clear Moral of the Story (`सीख / नैतिक शिक्षा`) in Hindi.

16:9 VISUAL & AI IMAGE PROMPT RULES:
- Every scene MUST have a detailed English `image_prompt` describing the exact scene action, characters, expressions, warm Indian storybook illustration style, lush jungle/village/palace setting, 16:9 widescreen cinematic framing, and `"no text, no words, no watermarks"`.
- `on_screen_callout` must be a short 2-4 word Hindi Devanagari chapter/scene badge (e.g. `"कहानी #1 • पंचतंत्र"`, `"चतुर बंदर की चाल"`, `"सच्ची मित्रता की सीख"`).

Return ONLY valid JSON with this exact structure:
{
  "story_number": 1,
  "collection_source": "Panchatantra",
  "title": "कहानी #1: बंदर और मगरमच्छ | Panchatantra Ki Kahani | Hindi Kahaniyan",
  "description": "पंचतंत्र की प्रसिद्ध कहानी #1: बंदर और मगरमच्छ की रोचक और शिक्षाप्रद कहानी। बच्चों और बड़ों सभी के लिए नैतिक शिक्षाप्रद हिंदी कहानी।",
  "moral_hindi": "संकट के समय घबराने के बजाय बुद्धि और धैर्य से काम लेना चाहिए।",
  "tags": ["panchatantra", "hindikahani", "moralstories", "indianstories", "kahaniyan", "aigenerated", "hindistories"],
  "sources": [{"name": "Panchatantra by Pandit Vishnu Sharma (Mitralabha / Mitrabheda)", "url": "https://en.wikipedia.org/wiki/Panchatantra"}],
  "wikipedia_articles": ["Panchatantra", "Hitopadesha"],
  "scenes": [
    {
      "scene_id": 1,
      "speaker_name": "कथावाचक (Narrator)",
      "speaker_role": "narrator",
      "speaker_gender": "male",
      "narration": "नमस्कार दोस्तों! आज की हमारी पहली पंचतंत्र की कहानी है—चतुर बंदर और लालची मगरमच्छ। किसी नदी के किनारे एक विशाल जामुन का पेड़ था, जिस पर रक्तमुख नाम का एक बड़ा ही नेक और मिलनसार बंदर रहता था।",
      "visual_subject": "Lush Indian river bank with a giant Jamun tree and a friendly monkey eating sweet purple berries",
      "visual_keywords": ["monkey", "jamun tree", "indian river", "panchatantra"],
      "image_prompt": "Warm cinematic 3D Pixar and Amar Chitra Katha inspired Indian storybook illustration, 16:9 widescreen composition, a cheerful friendly monkey sitting on a lush green Jamun berry tree beside a sparkling Indian river at golden sunrise, vibrant colors, no text, no watermarks",
      "wikimedia_query": "Panchatantra monkey crocodile illustration",
      "wikipedia_article": "Panchatantra",
      "prefer_media": "ai_image",
      "effect": "smooth_push_in",
      "on_screen_callout": "कहानी #1 • पंचतंत्र"
    }
  ]
}
"""


class FoundryLLMClient:
    def __init__(self):
        self.endpoint = (
            os.getenv("FOUNDRY_TEXT_ENDPOINT")
            or os.getenv("FOUNDRY_ENDPOINT")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        ).strip().rstrip("/")
        self.api_key = (
            os.getenv("FOUNDRY_TEXT_API_KEY")
            or os.getenv("FOUNDRY_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        self.model = (
            os.getenv("FOUNDRY_TEXT_MODEL")
            or os.getenv("FOUNDRY_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-6-luna"
        ).strip()

    def _candidate_urls(self) -> List[str]:
        base = self.endpoint
        if not base:
            return []
        if "/chat/completions" in base:
            root = base.split("/models/")[0].split("/openai/")[0].split("/chat/completions")[0].rstrip("/")
            candidates = [base]
        else:
            root = base.split("/models")[0].split("/openai")[0].rstrip("/")
            candidates = []

        candidates.extend([
            f"{root}/openai/deployments/{self.model}/chat/completions?api-version=2024-12-01-preview",
            f"{root}/openai/deployments/{self.model}/chat/completions?api-version=2024-10-21",
            f"{root}/models/chat/completions?api-version=2024-05-01-preview",
            f"{root}/openai/v1/chat/completions",
            f"{root}/v1/chat/completions",
            f"{root}/chat/completions",
        ])
        return list(dict.fromkeys(candidates))

    def _call_chat_completions(self, messages: List[Dict[str, str]]) -> str:
        if not self.endpoint or not self.api_key:
            raise RuntimeError("Missing FOUNDRY_TEXT_ENDPOINT / FOUNDRY_TEXT_API_KEY.")

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }
        payload_variants = [
            {"model": self.model, "messages": messages, "temperature": 0.7},
            {"messages": messages},
        ]

        last_err_detail = ""
        for url in self._candidate_urls():
            for payload in payload_variants:
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=95)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                return content
                    else:
                        last_err_detail = f"[{resp.status_code} on {url.split('//')[-1][:45]}]: {resp.text[:300]}"
                except Exception as exc:
                    last_err_detail = str(exc)

        raise RuntimeError(f"All LLM endpoint variants failed. Last response: {last_err_detail}")

    def generate_storyboard(self, topic: str, story_number: int = 1, collection_source: str = "Panchatantra") -> Dict[str, Any]:
        if self.endpoint and self.api_key:
            try:
                return self._generate_storyboard_via_api(topic, story_number, collection_source)
            except Exception as exc:
                print(f"[FoundryLLM] API notice ({exc}); using built-in Hindi Multi-Character Story Engine fallback.")
        else:
            print("[FoundryLLM] No LLM endpoint configured; using built-in Hindi Multi-Character Story Engine.")
        return self._build_fallback_hindi_storyboard(topic, story_number, collection_source)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=6))
    def _generate_storyboard_via_api(self, topic: str, story_number: int, collection_source: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Create a complete long-form 16:9 Hindi story script for Story #{story_number}: {topic!r} "
                f"(Source Collection: {collection_source}).\n"
                "Requirements:\n"
                "1. Use 12 to 16 rich storytelling scenes in pure Devanagari Hindi (`narration`).\n"
                "2. Include distinct character dialogue beats with accurate `speaker_role` (`narrator`, `male_protagonist`, `male_antagonist`, `female_character`, `wise_elder`) and `speaker_gender` (`male` or `female`). Include at least one female character voice beat if the story has a wife/queen/mother/bird/female character.\n"
                "3. Make every scene's `image_prompt` a detailed 16:9 Indian storybook illustration description matching the exact audio narration of that scene.\n"
                "4. Number the title with `कहानी #" + str(story_number) + "` and include AI content tags. Return valid JSON only."
            )},
        ]
        content = self._call_chat_completions(messages)
        match = re.search(r"\{[\s\S]*\}", content)
        storyboard = json.loads(match.group(0) if match else content)
        return self._normalize_storyboard(storyboard, topic, story_number, collection_source)

    @staticmethod
    def _normalize_storyboard(storyboard: Dict[str, Any], topic: str, story_number: int, collection_source: str) -> Dict[str, Any]:
        storyboard["topic"] = topic
        storyboard["story_number"] = story_number
        storyboard["collection_source"] = storyboard.get("collection_source") or collection_source

        raw_title = str(storyboard.get("title") or topic).strip()
        if f"#{story_number}" not in raw_title:
            raw_title = f"कहानी #{story_number}: {raw_title}"
        storyboard["title"] = raw_title[:98]

        tags = list(storyboard.get("tags") or [])
        for req_tag in ("panchatantra", "hindikahani", "moralstories", "indianstories", "ai", "aigenerated", "aistory"):
            if req_tag not in [t.lower() for t in tags]:
                tags.append(req_tag)
        storyboard["tags"] = tags

        scenes = storyboard.get("scenes") or []
        if not isinstance(scenes, list) or not scenes:
            raise RuntimeError("Storyboard contained no scenes.")

        effects = ["smooth_push_in", "cinema_pan_right", "smooth_pull_out", "cinema_pan_left"]
        for idx, scene in enumerate(scenes, 1):
            scene["scene_id"] = int(scene.get("scene_id") or idx)
            scene["narration"] = str(scene.get("narration") or "").strip()
            role = str(scene.get("speaker_role") or "narrator").strip().lower()
            gender = str(scene.get("speaker_gender") or ("female" if "female" in role else "male")).strip().lower()
            if gender not in ("male", "female"):
                gender = "female" if "female" in role else "male"
            scene["speaker_role"] = role
            scene["speaker_gender"] = gender
            scene["speaker_name"] = str(scene.get("speaker_name") or ("महिला पात्र" if gender == "female" else "कथावाचक")).strip()
            scene["visual_subject"] = str(scene.get("visual_subject") or topic).strip()
            scene["image_prompt"] = str(
                scene.get("image_prompt")
                or f"Warm Indian storybook illustration, 16:9 widescreen, {scene['visual_subject']}, lush traditional colors, no text, no watermark"
            ).strip()
            scene["prefer_media"] = str(scene.get("prefer_media") or "ai_image").strip()
            scene["effect"] = str(scene.get("effect") or effects[(idx - 1) % len(effects)]).strip()
            scene["on_screen_callout"] = str(scene.get("on_screen_callout") or f"कहानी #{story_number}").strip()

        print(f"[FoundryLLM] Prepared Hindi Storyboard '{storyboard['title']}' ({len(scenes)} scenes, Story #{story_number}).")
        return storyboard

    def _build_fallback_hindi_storyboard(self, topic: str, story_number: int, collection_source: str) -> Dict[str, Any]:
        """
        Rich 12-scene multi-character Hindi storyboard generator (with Narrator, Male Protagonist, Male Antagonist,
        and Female Character voices in pure Devanagari) so dry-runs and offline tests produce a genuine Hindi story!
        """
        # Clean title extraction
        clean_name = re.sub(r"^(?:कहानी|story)\s*#\s*\d+\s*[:\-—]\s*", "", topic, flags=re.IGNORECASE).strip()
        hindi_title_part = clean_name.split("(")[0].replace("Panchatantra —", "").replace("Hitopadesha —", "").strip()
        if not hindi_title_part:
            hindi_title_part = "बंदर और मगरमच्छ की पंचतंत्र कथा"

        if "बंदर और मगरमच्छ" in topic or "Monkey and the Crocodile" in topic or story_number == 1:
            scenes = [
                {
                    "scene_id": 1,
                    "speaker_name": "कथावाचक (Narrator)",
                    "speaker_role": "narrator",
                    "speaker_gender": "male",
                    "narration": f"नमस्कार दोस्तों! हमारी पंचतंत्र श्रृंखला की कहानी संख्या {story_number} में आपका स्वागत है। आज की कहानी है—चतुर बंदर रक्तमुख और मगरमच्छ करालमुख की।",
                    "visual_subject": "Ancient Indian riverbank at sunrise with a giant Jamun tree and a wise monkey",
                    "image_prompt": "Warm cinematic 3D Indian storybook illustration, 16:9 widescreen, a lush green Jamun tree on a peaceful Indian river bank at golden sunrise, a cheerful monkey sitting on a branch holding sweet purple jamun fruits, vibrant lighting, no text, no watermark",
                    "prefer_media": "ai_image",
                    "effect": "smooth_push_in",
                    "on_screen_callout": f"कहानी #{story_number} • पंचतंत्र",
                },
                {
                    "scene_id": 2,
                    "speaker_name": "कथावाचक (Narrator)",
                    "speaker_role": "narrator",
                    "speaker_gender": "male",
                    "narration": "नर्मदा नदी के किनारे जामुन के एक विशाल पेड़ पर रक्तमुख नाम का बंदर रहता था। उस पेड़ के जामुन शहद जैसे मीठे थे। एक दिन करालमुख नाम का एक थका हुआ मगरमच्छ नदी से बाहर आया।",
                    "visual_subject": "A large green crocodile resting on the sandy riverbank below the Jamun tree",
                    "image_prompt": "Warm Indian storybook illustration, 16:9 widescreen, a large crocodile resting near the roots of a giant Jamun tree on a river bank while a friendly monkey looks down kindly offering purple berries, rich detail, no text",
                    "prefer_media": "ai_image",
                    "effect": "cinema_pan_right",
                    "on_screen_callout": "नदी किनारे मित्रता",
                },
                {
                    "scene_id": 3,
                    "speaker_name": "बंदर रक्तमुख (Monkey)",
                    "speaker_role": "male_protagonist",
                    "speaker_gender": "male",
                    "narration": "अरे भाई मगरमच्छ! तुम हमारे अतिथि हो। लो, ये मीठे और रसीले जामुन खाओ, तुम्हारी सारी थकान दूर हो जाएगी!",
                    "visual_subject": "Close up of the cheerful monkey tossing sweet purple jamun berries to the smiling crocodile",
                    "image_prompt": "Vibrant Indian animated storybook art, 16:9 widescreen, close-up of a joyful monkey on a branch tossing glowing purple berries down to a grateful crocodile in the crystal river water, sunlit forest, no text",
                    "prefer_media": "ai_image",
                    "effect": "smooth_pull_out",
                    "on_screen_callout": "अतिथि का सत्कार",
                },
                {
                    "scene_id": 4,
                    "speaker_name": "मगरमच्छ करालमुख (Crocodile)",
                    "speaker_role": "male_antagonist",
                    "speaker_gender": "male",
                    "narration": "वाह मित्र रक्तमुख! इतने अमृत जैसे मीठे फल मैंने जीवन में कभी नहीं खाए। मैं थोड़े जामुन अपनी पत्नी के लिए भी ले जाता हूँ।",
                    "visual_subject": "Crocodile happily swimming home across the river carrying sweet jamun fruits on a leaf",
                    "image_prompt": "Cinematic Indian folklore illustration, 16:9 widescreen, a crocodile swimming across a golden sunlit river carrying purple jamun fruits on a lotus leaf, lush Indian jungle background, no text",
                    "prefer_media": "ai_image",
                    "effect": "cinema_pan_left",
                    "on_screen_callout": "मीठे जामुन का उपहार",
                },
                {
                    "scene_id": 5,
                    "speaker_name": "मगरमच्छ की पत्नी (Crocodile's Wife)",
                    "speaker_role": "female_character",
                    "speaker_gender": "female",
                    "narration": "सुना जी! जो बंदर रोज़ इतने मीठे और रसीले जामुन खाता है, उसका कलेजा कितना मीठा होगा! मुझे तो बस उसी बंदर का कलेजा खाना है, वरना मैं अपने प्राण त्याग दूँगी!",
                    "visual_subject": "Greedy female crocodile wearing subtle traditional ornaments demanding the monkey's heart from her worried husband",
                    "image_prompt": "Expressive Indian storybook illustration, 16:9 widescreen, a cunning female crocodile on a rocky river grotto speaking dramatically to a worried male crocodile, dramatic warm lighting, no text, no watermark",
                    "prefer_media": "ai_image",
                    "effect": "smooth_push_in",
                    "on_screen_callout": "लालची पत्नी की ज़िद",
                },
                {
                    "scene_id": 6,
                    "speaker_name": "मगरमच्छ करालमुख (Crocodile)",
                    "speaker_role": "male_antagonist",
                    "speaker_gender": "male",
                    "narration": "मित्र रक्तमुख! तुम्हारी भाभी ने आज तुम्हें अपने घर भोजन पर बुलाया है। मेरी पीठ पर बैठ जाओ, मैं तुम्हें सुरक्षित नदी पार ले चलूँगा।",
                    "visual_subject": "Monkey sitting trustingly on the crocodile's back in the middle of the deep river",
                    "image_prompt": "Cinematic 16:9 Indian storybook art, a trusting monkey riding on the back of a large crocodile into the deep blue river waters, mist and dramatic clouds gathering, no text",
                    "prefer_media": "ai_image",
                    "effect": "cinema_pan_right",
                    "on_screen_callout": "कपटपूर्ण निमंत्रण",
                },
                {
                    "scene_id": 7,
                    "speaker_name": "कथावाचक (Narrator)",
                    "speaker_role": "narrator",
                    "speaker_gender": "male",
                    "narration": "जब वे गहरे पानी के बीच पहुँचे, तो मगरमच्छ ने दुखी मन से सच बता दिया कि उसकी पत्नी बंदर का कलेजा खाना चाहती है। यह सुनकर बंदर के होश उड़ गए, पर उसने तुरंत धैर्य से काम लिया।",
                    "visual_subject": "Monkey looking shocked on the crocodile's back in deep water, then having a clever idea",
                    "image_prompt": "Dramatic 16:9 Indian animated illustration, a clever monkey on a crocodile's back in the middle of a deep river with a bright spark of wisdom in his eyes, cinematic lighting, no text",
                    "prefer_media": "ai_image",
                    "effect": "smooth_pull_out",
                    "on_screen_callout": "संकट में धैर्य",
                },
                {
                    "scene_id": 8,
                    "speaker_name": "बंदर रक्तमुख (Monkey)",
                    "speaker_role": "male_protagonist",
                    "speaker_gender": "male",
                    "narration": "अरे मित्र! तुमने यह बात पहले क्यों नहीं बताई? हम बंदर तो अपना कलेजा जामुन के पेड़ की खोखल में ही संभाल कर रखते हैं! चलो वापस पेड़ पर चलें, मैं अभी अपना कलेजा ले आता हूँ।",
                    "visual_subject": "Monkey pointing back toward the giant Jamun tree on the riverbank while speaking calmly to the foolish crocodile",
                    "image_prompt": "Warm 16:9 Indian storybook illustration, a clever monkey on a crocodile's back pointing back toward the lush Jamun tree on the shore with a convincing smile, crocodile turning around in the water, no text",
                    "prefer_media": "ai_image",
                    "effect": "cinema_pan_left",
                    "on_screen_callout": "चतुर बंदर की युक्ति",
                },
                {
                    "scene_id": 9,
                    "speaker_name": "कथावाचक (Narrator)",
                    "speaker_role": "narrator",
                    "speaker_gender": "male",
                    "narration": "मूर्ख मगरमच्छ बंदर की बातों में आ गया और तैरकर वापस किनारे की ओर चल पड़ा। जैसे ही वे किनारे पहुँचे, बंदर ने एक लंबी छलांग लगाई और सीधा पेड़ की सबसे ऊँची डाल पर जा बैठा!",
                    "visual_subject": "Monkey leaping high onto the safe top branch of the Jamun tree while the crocodile waits below",
                    "image_prompt": "Dynamic 16:9 Indian storybook art, a joyful monkey leaping gracefully from a crocodile's back onto the high branches of a giant Jamun tree, water splashes sparkling in sunlight, no text",
                    "prefer_media": "ai_image",
                    "effect": "smooth_push_in",
                    "on_screen_callout": "सुरक्षित छलांग",
                },
                {
                    "scene_id": 10,
                    "speaker_name": "बंदर रक्तमुख (Monkey)",
                    "speaker_role": "male_protagonist",
                    "speaker_gender": "male",
                    "narration": "अरे मूर्ख करालमुख! क्या किसी का कलेजा भी शरीर से अलग पेड़ पर रखा जाता है? तुमने मित्रता के नाम पर विश्वासघात किया, इसलिए आज से हमारी दोस्ती समाप्त!",
                    "visual_subject": "Monkey speaking boldly from the high tree branch as the ashamed crocodile looks up from the river",
                    "image_prompt": "Expressive 16:9 Indian storybook illustration, a wise monkey sitting high on a Jamun tree branch looking down at an embarrassed crocodile at the water's edge, golden sunset, no text",
                    "prefer_media": "ai_image",
                    "effect": "cinema_pan_right",
                    "on_screen_callout": "विश्वासघात का उत्तर",
                },
                {
                    "scene_id": 11,
                    "speaker_name": "कथावाचक (Narrator)",
                    "speaker_role": "wise_elder",
                    "speaker_gender": "male",
                    "narration": "दोस्तों, पंचतंत्र की इस अमर कहानी से हमें यह सीख मिलती है कि बड़े से बड़े संकट में भी घबराना नहीं चाहिए। बुद्धि और सूझबूझ से हर मुसीबत को हराया जा सकता है।",
                    "visual_subject": "Pandit Vishnu Sharma teaching young princes under a banyan tree with a glowing golden scroll of wisdom",
                    "image_prompt": "Serene 16:9 Indian heritage illustration, ancient sage Pandit Vishnu Sharma sitting under a grand banyan tree narrating wisdom to young princes in warm golden lantern light, peaceful Indian ashram, no text",
                    "prefer_media": "ai_image",
                    "effect": "smooth_pull_out",
                    "on_screen_callout": "कहानी की नैतिक शिक्षा",
                },
            ]
        else:
            scenes = [
                {
                    "scene_id": 1,
                    "speaker_name": "कथावाचक (Narrator)",
                    "speaker_role": "narrator",
                    "speaker_gender": "male",
                    "narration": f"नमस्कार दोस्तों! हमारी भारतीय कथा श्रृंखला की कहानी संख्या {story_number} में आपका स्वागत है। आज हम सुनेंगे—{hindi_title_part} की एक बेहद रोचक और शिक्षाप्रद कहानी।",
                    "visual_subject": f"Ancient Indian village and lush forest setting for {clean_name}",
                    "image_prompt": f"Warm cinematic 3D Indian storybook illustration, 16:9 widescreen composition, ancient Indian kingdom and lush forest at sunrise depicting {clean_name}, vibrant heritage colors, no text, no watermark",
                    "prefer_media": "ai_image",
                    "effect": "smooth_push_in",
                    "on_screen_callout": f"कहानी #{story_number} • {collection_source.split()[0]}",
                },
                {
                    "scene_id": 2,
                    "speaker_name": "कथावाचक (Narrator)",
                    "speaker_role": "narrator",
                    "speaker_gender": "male",
                    "narration": f"बहुत समय पहले चंपकवन के सुंदर राज्य में यह अद्भुत घटना घटी। वहाँ के सभी प्राणी और नागरिक मिलजुल कर रहते थे, परंतु एक दिन उनके सामने एक बड़ी परीक्षा आ खड़ी हुई।",
                    "visual_subject": f"Vibrant scene from {clean_name} in an ancient Indian forest clearing",
                    "image_prompt": f"Cinematic 16:9 Indian folklore art, animals and villagers gathered in a sunlit ancient Indian forest clearing for {clean_name}, rich storybook detail, no text",
                    "prefer_media": "ai_image",
                    "effect": "cinema_pan_right",
                    "on_screen_callout": "चंपकवन की कथा",
                },
                {
                    "scene_id": 3,
                    "speaker_name": "बुद्धिमान पात्र (Protagonist)",
                    "speaker_role": "male_protagonist",
                    "speaker_gender": "male",
                    "narration": "भाइयों! हमें किसी भी लालच या डर के आगे अपना धैर्य नहीं खोना चाहिए। यदि हम मिलकर विचार करें, तो हर समस्या का समाधान निकल सकता है।",
                    "visual_subject": f"Clever protagonist speaking with courage in {clean_name}",
                    "image_prompt": f"Warm 16:9 Indian storybook illustration, central protagonist of {clean_name} speaking bravely with expressive eyes in a lush Indian setting, golden hour lighting, no text",
                    "prefer_media": "ai_image",
                    "effect": "smooth_pull_out",
                    "on_screen_callout": "बुद्धि और साहस",
                },
                {
                    "scene_id": 4,
                    "speaker_name": "महिला पात्र (Wise Companion)",
                    "speaker_role": "female_character",
                    "speaker_gender": "female",
                    "narration": "तुम बिल्कुल ठीक कहते हो! बिना सोचे-समझे किया गया कार्य हमेशा दुख देता है, जबकि सच्ची नीति और ईमानदारी हमेशा विजय दिलाती है।",
                    "visual_subject": f"Graceful female character offering wise counsel in {clean_name}",
                    "image_prompt": f"Expressive 16:9 Indian storybook illustration, a wise female character in traditional Indian attire or graceful forest creature sharing counsel in {clean_name}, warm glowing atmosphere, no text",
                    "prefer_media": "ai_image",
                    "effect": "cinema_pan_left",
                    "on_screen_callout": "सच्ची सलाह",
                },
                {
                    "scene_id": 5,
                    "speaker_name": "घमंडी पात्र (Antagonist)",
                    "speaker_role": "male_antagonist",
                    "speaker_gender": "male",
                    "narration": "मुझसे अधिक शक्तिशाली यहाँ कोई नहीं है! मैं अपनी ताकत से जो चाहूँ वह हासिल कर सकता हूँ, मुझे किसी की सलाह की आवश्यकता नहीं!",
                    "visual_subject": f"Arrogant antagonist boasting proudly in {clean_name}",
                    "image_prompt": f"Dramatic 16:9 Indian storybook illustration, proud antagonist of {clean_name} standing boastfully under dramatic shadows, rich colors, no text, no watermark",
                    "prefer_media": "ai_image",
                    "effect": "smooth_push_in",
                    "on_screen_callout": "अहंकार की परीक्षा",
                },
                {
                    "scene_id": 6,
                    "speaker_name": "कथावाचक (Narrator)",
                    "speaker_role": "narrator",
                    "speaker_gender": "male",
                    "narration": "परंतु जल्द ही समय ने करवट ली। घमंड और लालच के कारण वह मुसीबत में फँस गया, जबकि समझदार मित्रों ने अपनी सूझबूझ से न केवल अपनी रक्षा की, बल्कि सबको सही मार्ग भी दिखाया।",
                    "visual_subject": f"Climax of {clean_name} where wisdom triumphs over arrogance",
                    "image_prompt": f"Dynamic 16:9 Indian folklore illustration showing wisdom triumphing over pride in {clean_name}, warm cinematic lighting, no text",
                    "prefer_media": "ai_image",
                    "effect": "cinema_pan_right",
                    "on_screen_callout": "सूझबूझ की जीत",
                },
                {
                    "scene_id": 7,
                    "speaker_name": "आचार्य विष्णु शर्मा (Moral)",
                    "speaker_role": "wise_elder",
                    "speaker_gender": "male",
                    "narration": f"दोस्तों, कहानी संख्या {story_number} से हमें यह अनमोल सीख मिलती है कि बुद्धि, एकता और धैर्य के सामने बड़ी से बड़ी कठिनाई भी हार मान लेती है।",
                    "visual_subject": "Ancient Indian guru under a banyan tree imparting the moral lesson to children and adults",
                    "image_prompt": "Peaceful 16:9 Indian storybook illustration, a wise Indian storyteller under a glowing banyan tree at dusk with lanterns, children and villagers listening happily, no text",
                    "prefer_media": "ai_image",
                    "effect": "smooth_pull_out",
                    "on_screen_callout": "कहानी की सीख",
                },
            ]

        raw = {
            "story_number": story_number,
            "collection_source": collection_source,
            "title": f"कहानी #{story_number}: {hindi_title_part} | {collection_source.split()[0]} Moral Story in Hindi",
            "description": (
                f"कहानी #{story_number}: {clean_name}\n"
                f"बच्चों और बड़ों सभी के लिए पंचतंत्र और भारतीय लोककथाओं की शिक्षाप्रद हिंदी कहानी।\n"
                "इस वीडियो में दृश्य और आवाज़ कृत्रिम बुद्धिमत्ता (AI) की सहायता से तैयार किए गए हैं।"
            ),
            "moral_hindi": "संकट के समय घबराने के बजाय बुद्धि, धैर्य और सच्ची मित्रता से काम लेना चाहिए।",
            "tags": ["panchatantra", "hindikahani", "moralstories", "indianstories", "kahaniyan", "ai", "aigenerated", "hindistories"],
            "sources": [
                {"name": "Panchatantra by Pandit Vishnu Sharma", "url": "https://en.wikipedia.org/wiki/Panchatantra"},
                {"name": "Hitopadesha Sanskrit Fables", "url": "https://en.wikipedia.org/wiki/Hitopadesha"},
            ],
            "wikipedia_articles": ["Panchatantra", "Hitopadesha"],
            "scenes": scenes,
        }
        return self._normalize_storyboard(raw, topic, story_number, collection_source)

    def generate_fresh_topic(self, recent_topics: List[str]) -> str:
        if not self.endpoint or not self.api_key:
            return f"Hitopadesha — बुद्धिमान हंस और चतुर कछुए की नई शिक्षाप्रद कहानी #{len(recent_topics) + 1}"
        messages = [
            {
                "role": "system",
                "content": (
                    "Return ONLY a single line with one classic Indian moral story topic from Panchatantra, Hitopadesha, "
                    "Jataka Tales, Akbar-Birbal, Tenali Rama, or Vikram-Betaal in format: "
                    "`<Source> — <Hindi Title in Devanagari> (<English Title>)`."
                ),
            },
            {
                "role": "user",
                "content": f"Suggest 1 classic Indian story not in this list: {', '.join(recent_topics[-40:])}",
            },
        ]
        content = self._call_chat_completions(messages)
        return content.strip().strip('"')

    def generate_fresh_topics_batch(self, recent_topics: List[str], count: int = 15, start_number: int = 102) -> List[str]:
        if not self.endpoint or not self.api_key:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert curator of Indian folklore (Panchatantra, Hitopadesha, Jataka Tales, Akbar-Birbal, "
                    "Tenali Rama, Vikram-Betaal, Kathasaritsagara). Return ONLY a JSON array of strings in format: "
                    "`[\"कहानी #102: Hitopadesha — <Hindi Title> (<English Title>)\", ...]`."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate {count} brand-new classic Indian story topics starting at कहानी #{start_number} "
                    f"that do NOT overlap with any of these: {json.dumps(recent_topics[-60:], ensure_ascii=False)}. "
                    "Return ONLY a valid JSON array of strings."
                ),
            },
        ]
        content = self._call_chat_completions(messages)
        match = re.search(r"\[[\s\S]*\]", content)
        arr = json.loads(match.group(0) if match else content)
        return [str(x).strip() for x in arr if str(x).strip()]
