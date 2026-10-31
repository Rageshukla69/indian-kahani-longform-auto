"""
Hindi Long-Form Multi-Character Storyboard Architect (`src/foundry_llm.py` — v2.0 Photorealistic Long-Form Edition):
- Generates deep, engaging 6 to 10 minute long-form Hindi stories (24 to 30 rich scenes, 750 to 1,100 spoken Hindi words)
  in pure, natural Devanagari Hindi (`देवनागरी हिन्दी`).
- Assigns every scene TWO distinct **Photorealistic 8K Live-Action Cinema (`image_prompt` + `image_prompt_b`)** descriptions
  so the video cuts to a brand-new photorealistic 16:9 image every 6–9 seconds (producing **30 to 50 free AI images per story**!).
- Strictly bans cartoon/comic/2D illustration styles — every visual prompt is engineered for **Ultra-Realistic 8K Live-Action
  National Geographic / Epic Indian Period Cinema Photography**.
- Assigns young male & young female character roles (`narrator`, `male_protagonist`, `male_antagonist`, `female_character`, `young_character`).
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

PHOTOREALISTIC_STYLE_SUFFIX = (
    ", ultra-realistic 8k live-action cinema photography, shot on ARRI Alexa 65 with 35mm anamorphic lens, "
    "photorealistic natural skin/fur/water/foliage textures, dramatic golden-hour volumetric lighting, "
    "wide 16:9 horizontal composition, National Geographic and epic Indian historical film realism, "
    "NOT cartoon, NOT 2D illustration, NOT comic book, no text, no watermarks"
)

SYSTEM_PROMPT = """You are the master Hindi Storyteller (कथावाचक) and Cinematographer for India's #1 Long-Form Hindi Kahani YouTube Channel (covering Panchatantra ki 101 Kahaniyan, Hitopadesha, Jataka Tales, Akbar-Birbal, Tenali Rama, and Vikram-Betaal).
Write a rich, immersive, 6-to-9 minute long-form Hindi story script in **22 to 28 scenes** (750 to 1,050 spoken Hindi words total, ~32-42 Hindi words per scene).

STORYTELLING & VOICE RULES:
- Write all spoken `narration` in natural, expressive Devanagari Hindi (`देवनागरी लिपि`) suitable for both kids and adults.
- Build suspense, emotional depth, vivid sensory details, and lively character dialogues across 22 to 28 scenes so the video is a true long-form story (6–9 minutes).
- Assign each scene an accurate character speaker so our Young Hindi Voice Engine switches voices dynamically:
  - `"narrator"`: Young energetic male storyteller (`hi-IN-MadhurNeural` / `hi-IN-ArjunNeural`)
  - `"male_protagonist"`: Young clever male hero voice
  - `"female_character"`: Young expressive female voice (`hi-IN-SwaraNeural` / `hi-IN-KavyaNeural` — MUST be used whenever ANY female character, wife, queen, princess, mother, goddess, or female animal/bird speaks!)
  - `"male_antagonist"`: Dramatic male character voice
  - `"young_character"`: Bright youthful character voice
- Set `"speaker_gender": "female"` whenever a female character speaks, and `"male"` otherwise.

PHOTOREALISTIC 16:9 MULTI-IMAGE VISUAL RULES (CRITICAL):
- NEVER use words like "cartoon", "Pixar", "3D animation", "storybook illustration", "comic", or "Amar Chitra Katha".
- Every scene MUST provide TWO distinct photorealistic live-action camera prompts (`image_prompt` for the Wide Establishing Shot, and `image_prompt_b` for the Mid-Scene Action/Close-Up Shot) so the video changes visuals every 6–8 seconds (generating 35 to 50 free AI images per story!).
- Describe real-world physical subjects (e.g., "Photorealistic live-action wildlife photograph of a real Rhesus macaque monkey sitting on a mossy Jamun tree branch overhanging the Narmada river at sunrise, 8k ARRI Alexa cinema lighting").

Return ONLY valid JSON with this structure:
{
  "story_number": 1,
  "collection_source": "Panchatantra",
  "title": "कहानी #1: बंदर और मगरमच्छ | Panchatantra Moral Story in Hindi",
  "description": "पंचतंत्र की प्रसिद्ध कहानी #1...",
  "moral_hindi": "संकट के समय घबराने के बजाय बुद्धि और धैर्य से काम लेना चाहिए।",
  "tags": ["panchatantra", "hindikahani", "moralstories", "indianstories", "kahaniyan", "ai", "aigenerated"],
  "sources": [{"name": "Panchatantra by Pandit Vishnu Sharma", "url": "https://en.wikipedia.org/wiki/Panchatantra"}],
  "wikipedia_articles": ["Panchatantra", "Hitopadesha"],
  "scenes": [
    {
      "scene_id": 1,
      "speaker_name": "कथावाचक",
      "speaker_role": "narrator",
      "speaker_gender": "male",
      "narration": "नमस्कार दोस्तों! हमारी पंचतंत्र की 101 कहानियों की इस विशेष श्रृंखला की पहली कहानी में आपका हार्दिक स्वागत है...",
      "visual_subject": "Photorealistic aerial establishing shot of the Narmada river winding through a misty ancient Indian forest at sunrise",
      "image_prompt": "Photorealistic 8k live-action aerial cinema shot of the sacred Narmada river winding through a lush misty tropical Indian jungle at golden sunrise, volumetric sunbeams piercing ancient banyan and jamun trees, shot on ARRI Alexa 65, 16:9 widescreen, no text",
      "image_prompt_b": "Photorealistic 8k National Geographic close-up wildlife photograph of a real intelligent Rhesus monkey sitting on a fruit-laden purple Jamun tree branch above sparkling river water, golden morning light, shallow depth of field, 16:9 widescreen, no text",
      "prefer_media": "ai_image",
      "effect": "smooth_push_in",
      "on_screen_callout": "कहानी #1 • पंचतंत्र"
    }
  ]
}
"""


def _ensure_photorealistic_prompt(raw_prompt: str, fallback_subject: str) -> str:
    p = (raw_prompt or fallback_subject or "Ancient Indian forest river landscape at sunrise").strip()
    # Strip any cartoon/illustration keywords if an LLM ever includes them
    p = re.sub(
        r"\b(cartoon|pixar|3d\s*animation|animated|storybook\s*illustration|illustration|amar\s*chitra\s*katha|comic|2d|drawing|sketch)\b",
        "photorealistic live-action cinema",
        p,
        flags=re.IGNORECASE,
    )
    if "photorealistic" not in p.lower() and "live-action" not in p.lower():
        p = f"Photorealistic 8k live-action cinema photograph, {p}"
    if "not cartoon" not in p.lower():
        p = f"{p}{PHOTOREALISTIC_STYLE_SUFFIX}"
    return p


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
            {"model": self.model, "messages": messages, "temperature": 0.72},
            {"messages": messages},
        ]

        last_err_detail = ""
        for url in self._candidate_urls():
            for payload in payload_variants:
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=120)
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
                print(f"[FoundryLLM] API notice ({exc}); using built-in Long-Form Photorealistic Hindi Story Engine.")
        else:
            print("[FoundryLLM] No LLM endpoint configured; using built-in 24-Scene Long-Form Photorealistic Hindi Story Engine.")
        return self._build_fallback_hindi_storyboard(topic, story_number, collection_source)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=6))
    def _generate_storyboard_via_api(self, topic: str, story_number: int, collection_source: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Create a deep, immersive 6-to-9 minute long-form 16:9 Hindi story script for Story #{story_number}: {topic!r} "
                f"(Source Collection: {collection_source}).\n"
                "Requirements:\n"
                "1. Use **22 to 28 scenes** (at least 750-1,000 spoken Hindi words total, ~32-42 words per scene) in pure Devanagari Hindi (`narration`).\n"
                "2. Include rich character dialogues with `speaker_role` (`narrator`, `male_protagonist`, `male_antagonist`, `female_character`, `young_character`) and `speaker_gender` (`male` or `female`).\n"
                "3. Give EVERY scene TWO distinct **photorealistic 8k live-action cinema** image prompts (`image_prompt` and `image_prompt_b`) — NEVER cartoon or 2D illustration.\n"
                "4. Number the title with `कहानी #" + str(story_number) + "`. Return valid JSON only."
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

            base_p1 = str(scene.get("image_prompt") or scene["visual_subject"]).strip()
            base_p2 = str(
                scene.get("image_prompt_b")
                or f"Close-up cinematic angle of {scene['visual_subject']}, expressive detail, dramatic natural lighting"
            ).strip()
            scene["image_prompt"] = _ensure_photorealistic_prompt(base_p1, scene["visual_subject"])
            scene["image_prompt_b"] = _ensure_photorealistic_prompt(base_p2, scene["visual_subject"])
            scene["prefer_media"] = "ai_image"
            scene["effect"] = str(scene.get("effect") or effects[(idx - 1) % len(effects)]).strip()
            scene["on_screen_callout"] = str(scene.get("on_screen_callout") or f"कहानी #{story_number}").strip()

        print(
            f"[FoundryLLM] Prepared Photorealistic Long-Form Hindi Storyboard '{storyboard['title']}' "
            f"({len(scenes)} scenes, {len(scenes) * 2} planned 16:9 AI visual shots, Story #{story_number})."
        )
        return storyboard

    def _build_fallback_hindi_storyboard(self, topic: str, story_number: int, collection_source: str) -> Dict[str, Any]:
        """
        22-Scene Long-Form Photorealistic Hindi Storyboard (producing 44 photorealistic 16:9 visual shots
        and ~6+ minutes of multi-character young Hindi narration & dialogue).
        """
        clean_name = re.sub(r"^(?:कहानी|story)\s*#\s*\d+\s*[:\-—]\s*", "", topic, flags=re.IGNORECASE).strip()
        hindi_title_part = clean_name.split("(")[0].replace("Panchatantra —", "").replace("Hitopadesha —", "").strip()
        if not hindi_title_part:
            hindi_title_part = "बंदर और मगरमच्छ की पंचतंत्र कथा"

        scenes = [
            {
                "scene_id": 1,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": f"नमस्कार दोस्तों! हमारी पंचतंत्र की एक सौ एक कहानियों की इस विशेष श्रृंखला की कहानी संख्या {story_number} में आपका स्वागत है। आज हम सुनने जा रहे हैं चतुर बंदर रक्तमुख और मगरमच्छ करालमुख की वह अमर कहानी, जो हमें संकट में धैर्य और बुद्धि का सही मार्ग दिखाती है।",
                "visual_subject": "Aerial sunrise view over the sacred Narmada river winding through an ancient Indian jungle",
                "image_prompt": "Photorealistic 8k live-action aerial cinema photograph of the wide Narmada river winding through a lush misty Indian forest at golden sunrise, sunbeams piercing ancient trees, shot on ARRI Alexa 65",
                "image_prompt_b": "Photorealistic 8k live-action close-up of dewdrops on deep purple Jamun berries hanging from a lush green branch over a sparkling river at sunrise, cinematic depth of field",
                "on_screen_callout": f"कहानी #{story_number} • पंचतंत्र",
            },
            {
                "scene_id": 2,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "प्राचीन काल में नर्मदा नदी के पावन तट पर एक बहुत ही विशाल और घना जामुन का वृक्ष खड़ा था। उस पेड़ की डालियाँ नदी के ठंडे जल को छूती थीं, और उस पर साल भर बड़े-बड़े, काले और शहद जैसे रसीले जामुन लगते थे।",
                "visual_subject": "Giant ancient Jamun tree on the bank of the Narmada river heavy with ripe dark purple berries",
                "image_prompt": "Photorealistic 8k live-action nature photography of a massive ancient Jamun tree on a mossy Indian riverbank, branches heavy with ripe dark purple jamun fruits hanging over clear turquoise water",
                "image_prompt_b": "Photorealistic 8k low-angle cinema shot looking up through the sunlit canopy of an ancient riverine forest in central India, golden rays reflecting off gentle river ripples",
                "on_screen_callout": "नर्मदा तट का विशाल वृक्ष",
            },
            {
                "scene_id": 3,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "उसी विशाल पेड़ पर रक्तमुख नाम का एक बड़ा ही फुर्तीला, नेक और बुद्धिमान बंदर रहता था। वह दिन भर पेड़ की ऊँची टहनियों पर उछल-कूद करता और मीठे-मीठे जामुन खाकर आनंद से अपना जीवन बिताता था।",
                "visual_subject": "Real intelligent Rhesus macaque monkey sitting on a sunlit Jamun branch eating ripe purple berries",
                "image_prompt": "Photorealistic 8k National Geographic wildlife photograph of a real expressive Rhesus macaque monkey sitting on a thick Jamun tree branch by an Indian river, holding ripe purple berries in warm morning light",
                "image_prompt_b": "Photorealistic 8k extreme close-up wildlife portrait of a clever young macaque monkey with bright intelligent amber eyes looking out over a misty river valley at sunrise",
                "on_screen_callout": "बुद्धिमान बंदर रक्तमुख",
            },
            {
                "scene_id": 4,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "एक दोपहर जब सूरज की धूप तेज़ थी, नर्मदा की गहरी लहरों को चीरता हुआ करालमुख नाम का एक विशाल मगरमच्छ नदी के किनारे आया। वह सुबह से भोजन की तलाश में भटक रहा था और भूख तथा थकान से बिल्कुल निढाल हो चुका था।",
                "visual_subject": "Massive real Indian mugger crocodile emerging from the river water onto the warm sandy bank",
                "image_prompt": "Photorealistic 8k National Geographic wildlife photograph of a large real Indian mugger crocodile emerging from shimmering river water onto a sunlit sandy riverbank below a giant tree, wet scales glistening",
                "image_prompt_b": "Photorealistic 8k eye-level cinema shot of a tired massive crocodile resting on a shaded river rock under overhanging tree roots, water droplets on textured scales",
                "on_screen_callout": "थका हुआ मगरमच्छ करालमुख",
            },
            {
                "scene_id": 5,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "अरे भाई मगरमच्छ! तुम तो बहुत थके हुए और भूखे लग रहे हो। इस पेड़ की ठंडी छाँव में तुम मेरे अतिथि हो। रुको, मैं अभी तुम्हारे लिए इस पेड़ के सबसे मीठे और ताज़े जामुन तोड़कर नीचे गिराता हूँ!",
                "visual_subject": "Monkey looking down kindly from a low branch toward the crocodile on the riverbank",
                "image_prompt": "Photorealistic 8k wildlife cinema shot of a friendly macaque monkey leaning down from a low mossy branch over the riverbank, looking warmly at a crocodile resting on the sand below",
                "image_prompt_b": "Photorealistic 8k high-speed wildlife shot of ripe dark purple jamun berries falling through sunlit air into shallow crystal water near a crocodile on a riverbank",
                "on_screen_callout": "अतिथि धर्म का पालन",
            },
            {
                "scene_id": 6,
                "speaker_name": "मगरमच्छ करालमुख",
                "speaker_role": "male_antagonist",
                "speaker_gender": "male",
                "narration": "धन्यवाद भाई! आज से पहले किसी ने मेरे साथ इतना मधुर व्यवहार नहीं किया। सचमुच, ये जामुन तो अमृत के समान मीठे हैं! इन्हें खाकर मेरी सारी भूख और थकान पल भर में दूर हो गई।",
                "visual_subject": "Crocodile eating ripe purple jamun fruits on the sandy riverbank as the monkey watches happily",
                "image_prompt": "Photorealistic 8k wildlife photograph of a large Indian crocodile on a golden sunlit riverbank surrounded by fallen ripe purple jamun berries, peaceful tropical river background",
                "image_prompt_b": "Photorealistic 8k cinema shot of a macaque monkey perched on a low root by the water's edge near a calm crocodile, warm golden afternoon light reflecting on the river",
                "on_screen_callout": "गहरी मित्रता की शुरुआत",
            },
            {
                "scene_id": 7,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "उस दिन के बाद से करालमुख रोज़ नदी के किनारे आने लगा। रक्तमुख उसे ढेर सारे मीठे जामुन खिलाता और दोनों घंटों बैठकर देश-दुनिया और जंगल की रोचक बातें करते। देखते ही देखते उन दोनों के बीच बहुत गहरी मित्रता हो गई।",
                "visual_subject": "Monkey and crocodile spending peaceful afternoons together by the sunlit Narmada riverbank",
                "image_prompt": "Photorealistic 8k wide cinematic nature shot of a real macaque monkey sitting on a riverside boulder beside a large crocodile basking at the water's edge, golden sunset over an Indian river",
                "image_prompt_b": "Photorealistic 8k close-up of clusters of glistening dark purple jamun fruits gathered on a broad green lotus leaf by the riverbank",
                "on_screen_callout": "सच्चे मित्रों का साथ",
            },
            {
                "scene_id": 8,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "मित्र करालमुख! आज मैंने पेड़ की सबसे ऊँची डाल से ये विशेष रसीले जामुन चुने हैं। इन्हें तुम अपनी पत्नी के लिए घर ले जाओ। जब तुम्हारी पत्नी यह मीठा उपहार चखेगी, तो उसे भी बहुत प्रसन्नता होगी!",
                "visual_subject": "Monkey handing a lotus leaf filled with ripe purple berries to the crocodile at the river shore",
                "image_prompt": "Photorealistic 8k wildlife photograph of a macaque monkey on a low overhanging branch placing ripe purple jamun berries onto a large green leaf near a crocodile in the river",
                "image_prompt_b": "Photorealistic 8k cinema shot of a crocodile swimming across a tranquil golden river at dusk, ripples glowing in amber sunset light",
                "on_screen_callout": "भाभी के लिए मीठा उपहार",
            },
            {
                "scene_id": 9,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "करालमुख बड़े उत्साह के साथ वे मीठे जामुन लेकर नदी के दूसरे छोर पर बने अपने घर पहुँचा और अपनी पत्नी को भेंट किए। परंतु जैसे ही उसकी पत्नी ने उन रसीले फलों का स्वाद चखा, उसके मन में एक भयानक लालच जाग उठा।",
                "visual_subject": "Two crocodiles on a rocky river island at twilight surrounded by misty water",
                "image_prompt": "Photorealistic 8k wildlife cinema shot of two large crocodiles resting together on a secluded rocky island in the middle of a misty Indian river at twilight",
                "image_prompt_b": "Photorealistic 8k close-up of a female crocodile's intense yellow reptilian eye reflecting the amber glow of sunset over dark river water",
                "on_screen_callout": "मन में जागा लालच",
            },
            {
                "scene_id": 10,
                "speaker_name": "मगरमच्छ की पत्नी",
                "speaker_role": "female_character",
                "speaker_gender": "female",
                "narration": "सुना जी! ज़रा सोचिए, जो बंदर प्रतिदिन ऐसे अमृत जैसे मीठे जामुन खाता है, उसका अपना कलेजा कितना मीठा और स्वादिष्ट होगा! यदि आप मुझसे सच्चा प्रेम करते हैं, तो कल ही उस बंदर को यहाँ लाइए, मुझे उसका कलेजा खाना है!",
                "visual_subject": "Female crocodile aggressively confronting her husband on the dark rocky riverbank",
                "image_prompt": "Photorealistic 8k dramatic wildlife photograph of a female crocodile raising her head on a dark mossy river rock, confronting another crocodile beside swirling twilight water",
                "image_prompt_b": "Photorealistic 8k moody cinematic shot of dark storm clouds gathering over a wide Indian river at dusk, dramatic rim lighting on the water",
                "on_screen_callout": "पत्नी की कठोर ज़िद",
            },
            {
                "scene_id": 11,
                "speaker_name": "मगरमच्छ करालमुख",
                "speaker_role": "male_antagonist",
                "speaker_gender": "male",
                "narration": "अरे प्रिये, तुम यह कैसी अनर्थकारी बात कर रही हो? रक्तमुख मेरा धर्म-भाई और परम मित्र है! वह रोज़ मुझे निस्वार्थ भाव से मीठे फल खिलाता है। मैं अपने ही मित्र के साथ ऐसा घोर विश्वासघात कैसे कर सकता हूँ?",
                "visual_subject": "Distressed male crocodile looking away toward the river under moonlight",
                "image_prompt": "Photorealistic 8k cinematic wildlife shot of a solitary crocodile half-submerged in moonlit river water looking back toward the distant forest shore, moody blue night lighting",
                "image_prompt_b": "Photorealistic 8k close-up of gentle river waves lapping against dark wet stones under a silver full moon",
                "on_screen_callout": "धर्मसंकट और मित्रता",
            },
            {
                "scene_id": 12,
                "speaker_name": "मगरमच्छ की पत्नी",
                "speaker_role": "female_character",
                "speaker_gender": "female",
                "narration": "मैं कुछ नहीं जानती! मुझे गंभीर रोग हो गया है और वैद्य ने कहा है कि केवल बंदर के कलेजे से ही मेरे प्राण बच सकते हैं। यदि कल सूर्यास्त से पहले तुम उस बंदर को लेकर नहीं आए, तो मैं अन्न-जल त्याग कर अपने प्राण दे दूँगी!",
                "visual_subject": "Female crocodile lying stubbornly on the rock slabs refusing to move as morning mist rises",
                "image_prompt": "Photorealistic 8k wildlife photograph of a crocodile lying motionless on a misty rocky river ledge at dawn, dramatic shafts of morning light piercing the fog",
                "image_prompt_b": "Photorealistic 8k cinema shot of a conflicted crocodile slowly sliding into the deep misty river current at dawn",
                "on_screen_callout": "त्रिया-हठ के आगे विवश",
            },
            {
                "scene_id": 13,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "पत्नी के इस हठ के आगे करालमुख अंततः विवश हो गया। भारी मन और कपटपूर्ण योजना के साथ वह अगले दिन जामुन के पेड़ के पास पहुँचा। उसने अपने चेहरे पर झूठी मुस्कान सजाई और बंदर को पुकारा।",
                "visual_subject": "Crocodile arriving at the Jamun tree riverbank in the morning light looking up at the monkey",
                "image_prompt": "Photorealistic 8k wildlife photograph of a large crocodile floating near the roots of a giant Jamun tree in morning light, looking up toward a macaque monkey on the branch",
                "image_prompt_b": "Photorealistic 8k close-up of a curious macaque monkey peering down from a leafy green branch toward the shimmering water below",
                "on_screen_callout": "छलपूर्ण निमंत्रण",
            },
            {
                "scene_id": 14,
                "speaker_name": "मगरमच्छ करालमुख",
                "speaker_role": "male_antagonist",
                "speaker_gender": "male",
                "narration": "मित्र रक्तमुख! तुम्हारे भेजे हुए मीठे जामुन खाकर तुम्हारी भाभी अत्यंत प्रसन्न हुईं। उन्होंने आज तुम्हें हमारे घर विशेष दावत पर आमंत्रित किया है। आओ, मेरी पीठ पर बैठ जाओ, मैं तुम्हें सुरक्षित नदी की सैर कराते हुए घर ले चलूँगा।",
                "visual_subject": "Crocodile positioning its broad back near the shallow riverbank for the monkey",
                "image_prompt": "Photorealistic 8k wildlife cinema shot of a large crocodile resting in shallow clear water right beside a mossy tree root while a macaque monkey prepares to step down",
                "image_prompt_b": "Photorealistic 8k live-action shot of a macaque monkey sitting perched on the rough textured back of a swimming crocodile in calm river water",
                "on_screen_callout": "पीठ पर बैठकर यात्रा",
            },
            {
                "scene_id": 15,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "निश्छल और भोला बंदर अपने मित्र की बातों पर पूरा भरोसा करके खुशी-खुशी उसकी पीठ पर बैठ गया। मगरमच्छ तेजी से तैरता हुआ किनारे से दूर, नर्मदा नदी की सबसे गहरी और तेज़ धार के बीचों-बीच पहुँच गया।",
                "visual_subject": "Crocodile swimming into the deep turbulent center of the wide river with the monkey on its back",
                "image_prompt": "Photorealistic 8k wide cinematic shot of a crocodile swimming through deep dark blue river currents far from shore with a small macaque monkey sitting on its back, dramatic sky",
                "image_prompt_b": "Photorealistic 8k close-up of a macaque monkey looking around nervously at the deep swirling river water surrounding him on all sides",
                "on_screen_callout": "गहरे पानी में संकट",
            },
            {
                "scene_id": 16,
                "speaker_name": "मगरमच्छ करालमुख",
                "speaker_role": "male_antagonist",
                "speaker_gender": "male",
                "narration": "मित्र रक्तमुख, मुझे क्षमा कर देना! अब यहाँ चारों ओर गहरा जल है और तुम कहीं भाग नहीं सकते। सच यह है कि मेरी पत्नी ने कोई दावत नहीं रखी है, बल्कि वह तुम्हारा मीठा कलेजा खाना चाहती है!",
                "visual_subject": "Dramatic close-up of the crocodile speaking in the deep dark river water as waves splash",
                "image_prompt": "Photorealistic 8k dramatic wildlife close-up of a large crocodile in deep dark churning river water, water splashing around its jaws under tense overcast light",
                "image_prompt_b": "Photorealistic 8k close-up portrait of a macaque monkey with wide alert eyes in the middle of a river, sudden realization and sharp intelligence on its face",
                "on_screen_callout": "विश्वासघात का खुलासा",
            },
            {
                "scene_id": 17,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "यह भयानक सच सुनकर एक पल के लिए तो रक्तमुख के पैरों तले ज़मीन खिसक गई। चारों तरफ गहरा पानी था और तैरना उसे आता नहीं था। किंतु उसने तुरंत अपने मन को शांत किया और संकट से निकलने की एक चतुर युक्ति सोची।",
                "visual_subject": "Calm intelligent monkey composing himself on the crocodile's back in the middle of the river",
                "image_prompt": "Photorealistic 8k cinema shot of a poised macaque monkey sitting calmly on a crocodile's back in deep river water, golden rays breaking through clouds onto the monkey's face",
                "image_prompt_b": "Photorealistic 8k wide shot looking across the sparkling river toward the distant giant Jamun tree standing tall on the green shore",
                "on_screen_callout": "संकट में अद्भुत धैर्य",
            },
            {
                "scene_id": 18,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "अरे नादान मित्र करालमुख! इतनी सी बात थी तो तुमने मुझे पेड़ के पास ही क्यों नहीं बता दिया? भाभी की खुशी के लिए तो मैं खुशी-खुशी अपना कलेजा दे देता! परंतु हम बंदर तो उछल-कूद करते समय अपना कलेजा जामुन के पेड़ की खोखल में ही रख छोड़ते हैं!",
                "visual_subject": "Monkey gesturing calmly back toward the Jamun tree on the shore while riding the crocodile",
                "image_prompt": "Photorealistic 8k wildlife cinema shot of an expressive macaque monkey on a crocodile's back looking back toward the lush green riverbank, warm natural sunlight",
                "image_prompt_b": "Photorealistic 8k close-up of a massive hollow trunk inside an ancient Jamun tree on the riverbank, dappled forest sunlight",
                "on_screen_callout": "कलेजा पेड़ पर होने की युक्ति",
            },
            {
                "scene_id": 19,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "इस समय तो मेरा शरीर बिल्कुल खाली है। यदि तुम मुझे बिना कलेजे के ले जाओगे, तो भाभी तुम पर बहुत क्रोधित होंगी। इसलिए जल्दी से वापस किनारे चलो, ताकि मैं पेड़ से अपना मीठा कलेजा उतार लाऊँ!",
                "visual_subject": "Crocodile turning around in the river current and swimming rapidly back toward the Jamun tree",
                "image_prompt": "Photorealistic 8k dynamic wildlife photograph of a large crocodile turning sharply in the river current, creating white water wakes as it swims back toward the green shore",
                "image_prompt_b": "Photorealistic 8k POV shot approaching the lush sandy bank of the Narmada river where the giant Jamun tree roots touch the water",
                "on_screen_callout": "मूर्ख मगरमच्छ लौटा किनारे",
            },
            {
                "scene_id": 20,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "मूर्ख मगरमच्छ बंदर की इस चतुर बात पर तुरंत विश्वास कर बैठा और तेज़ी से तैरता हुआ वापस नदी के किनारे ले आया। जैसे ही मगरमच्छ किनारे के पास पहुँचा, रक्तमुख ने बिजली की फुर्ती से एक लंबी छलांग लगाई और सीधा पेड़ की सबसे ऊँची डाल पर जा पहुँचा!",
                "visual_subject": "Monkey leaping gracefully through the air from the crocodile's back onto the Jamun tree trunk",
                "image_prompt": "Photorealistic 8k high-speed National Geographic action photograph of a real macaque monkey leaping mid-air from the water's edge onto the trunk of a giant Jamun tree, water droplets frozen in sunlight",
                "image_prompt_b": "Photorealistic 8k low-angle wildlife shot of a macaque monkey sitting safely high up in the green canopy of a Jamun tree looking down",
                "on_screen_callout": "बिजली सी तेज़ छलांग",
            },
            {
                "scene_id": 21,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "अरे मूर्ख और विश्वासघाती करालमुख! भला दुनिया में किसी प्राणी के दो कलेजे होते हैं क्या, या कोई अपना कलेजा शरीर से निकाल कर पेड़ पर रखता है? जा, अब लौट जा! जो मित्र के विश्वास की रक्षा नहीं कर सकता, उससे मित्रता रखना महामूर्खता है!",
                "visual_subject": "Monkey looking down boldly from the high tree branch while the ashamed crocodile waits in the shallow water below",
                "image_prompt": "Photorealistic 8k cinema shot from behind a macaque monkey high on a tree branch looking down at a solitary crocodile in the river below during golden sunset",
                "image_prompt_b": "Photorealistic 8k wildlife photograph of a crocodile slowly sinking back into the dark river water alone as evening mist settles",
                "on_screen_callout": "विश्वासघाती को करारा उत्तर",
            },
            {
                "scene_id": 22,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": f"दोस्तों, पंचतंत्र की इस पहली अमर कहानी से हमें जीवन की दो सबसे बड़ी सीख मिलती हैं—पहली, कठिन से कठिन संकट आने पर भी कभी घबराना नहीं चाहिए, क्योंकि शांत दिमाग और सूझबूझ से हर मुसीबत को हराया जा सकता है। और दूसरी, कपटी मित्रों से सदैव सावधान रहना चाहिए। अगली कहानी में फिर मिलेंगे!",
                "visual_subject": "Serene sunset over the Narmada river and ancient Indian forest with golden light on the water",
                "image_prompt": "Photorealistic 8k epic cinema landscape of a tranquil Indian river valley at sunset, golden orange sky reflecting on calm water beside ancient banyan and jamun trees, ARRI Alexa 65",
                "image_prompt_b": "Photorealistic 8k atmospheric twilight shot of glowing brass oil diyas floating gently on the edge of a peaceful Indian river at dusk, warm bokeh",
                "on_screen_callout": "कहानी की अनमोल सीख",
            },
        ]

        # If the topic is another story (#2+), customize the narrative and visuals to that topic while keeping 22 scenes
        if story_number > 1 and "बंदर और मगरमच्छ" not in topic and "Monkey and the Crocodile" not in topic:
            for s in scenes:
                s["narration"] = s["narration"].replace("बंदर रक्तमुख और मगरमच्छ करालमुख", hindi_title_part)
                s["visual_subject"] = f"Photorealistic live-action Indian cinema scene for {clean_name} (Beat {s['scene_id']})"
                s["image_prompt"] = _ensure_photorealistic_prompt(
                    f"Photorealistic 8k live-action cinema wide shot depicting {clean_name} in ancient India, scene {s['scene_id']}, natural dramatic lighting",
                    clean_name,
                )
                s["image_prompt_b"] = _ensure_photorealistic_prompt(
                    f"Photorealistic 8k live-action cinema close-up shot depicting {clean_name} in ancient India, scene {s['scene_id']}, expressive detail",
                    clean_name,
                )

        raw = {
            "story_number": story_number,
            "collection_source": collection_source,
            "title": f"कहानी #{story_number}: {hindi_title_part} | {collection_source.split()[0]} Moral Story in Hindi",
            "description": (
                f"कहानी #{story_number}: {clean_name}\n"
                f"बच्चों और बड़ों सभी के लिए पंचतंत्र और भारतीय लोककथाओं की शिक्षाप्रद हिंदी कहानी।\n"
                "इस वीडियो में दृश्य और आवाज़ कृत्रिम बुद्धिमत्ता (AI) की सहायता से तैयार किए गए हैं।"
            ),
            "moral_hindi": "संकट के समय घबराने के बजाय बुद्धि, धैर्य और सूझबूझ से काम लेना चाहिए।",
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
