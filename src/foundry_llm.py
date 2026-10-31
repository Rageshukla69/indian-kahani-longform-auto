"""
Hindi Long-Form Multi-Character Storyboard Architect (`src/foundry_llm.py` — v3.0 Character-Consistent & 66-84 Shots Edition):
- Solves Character Inconsistency:
  Defines a strict **`character_bible`** for every story (locking exact species, age, fur/skin color, facial features,
  and environment details) and automatically injects the relevant character visual anchors into EVERY generated image prompt!
- Solves Visual-Audio Relevance & Image Count:
  Every scene (22 to 28 scenes per story) now contains **3 Beat-Synced Visual Prompts (`shot_prompts`: Beat A Opening,
  Beat B Mid-Action, Beat C Reaction/Climax)** directly tied to each sentence spoken in that scene — generating
  **66 to 84 free AI images per 6-minute video** (a brand-new scene-matched image every ~4.5 to 5.5 seconds!).
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
    "consistent character appearance, photorealistic natural textures, dramatic golden-hour volumetric lighting, "
    "wide 16:9 horizontal composition, National Geographic and epic Indian historical film realism, "
    "NOT cartoon, NOT 2D illustration, NOT comic book, no text, no watermarks"
)

DEFAULT_MONKEY_CROCODILE_BIBLE = {
    "Raktamukha (Monkey)": (
        "adult slender golden-brown Indian Rhesus macaque monkey with warm amber eyes, reddish-tan face, "
        "and expressive intelligent posture (SINGLE adult monkey, no baby monkey)"
    ),
    "Karalamukha (Crocodile)": (
        "massive 14-foot dark olive-green Indian mugger crocodile with heavy armored dorsal ridges, "
        "broad snout, and deep yellow reptilian eyes"
    ),
    "Crocodile Wife": (
        "large sleeker dark-bronze female Indian mugger crocodile with sharp golden eyes resting on wet mossy river rocks"
    ),
    "Setting": (
        "ancient Narmada riverbank in central India with a giant gnarled Jamun tree heavy with clusters of dark-purple berries overhanging turquoise water"
    ),
}

SYSTEM_PROMPT = """You are the master Hindi Storyteller (कथावाचक) and Cinematographer for India's #1 Long-Form Hindi Kahani YouTube Channel.
Write a rich, immersive, 6-to-9 minute long-form Hindi story script in **22 to 28 scenes** (750 to 1,050 spoken Hindi words total).

CHARACTER CONSISTENCY & VISUAL RELEVANCE RULES (CRITICAL):
1. First, define a `character_bible` object mapping each main character and the primary setting to a fixed, ultra-specific physical description (exact species/age/clothing/fur color/facial traits).
2. In EVERY scene, provide `shot_prompts`: a list of **3 distinct, consecutive English visual prompts** (`[Shot 1, Shot 2, Shot 3]`) that correspond to the 1st third, 2nd third, and final third of that scene's spoken Hindi `narration`.
3. Each prompt in `shot_prompts` MUST:
   - Explicitly repeat the exact physical description from `character_bible` for whichever character(s) appear in that shot (so characters look 100% consistent across all 66–84 images!).
   - Depict the EXACT physical action, emotion, and interaction being spoken in the Hindi narration at that moment (e.g., if the voice says the monkey is sitting on the crocodile's back in deep water, both the monkey AND the crocodile in the water must be in the prompt — never a random empty landscape or unrelated baby monkey!).
   - Be photorealistic 8K live-action cinema photography (NEVER cartoon or 2D illustration).

VOICE & DIALOGUE RULES:
- Allowed `speaker_role`: `"narrator"`, `"male_protagonist"`, `"male_antagonist"`, `"female_character"`, `"young_character"`.
- Set `"speaker_gender": "female"` whenever ANY female character speaks, and `"male"` otherwise.

Return ONLY valid JSON with this structure:
{
  "story_number": 1,
  "collection_source": "Panchatantra",
  "title": "कहानी #1: बंदर और मगरमच्छ | Panchatantra Moral Story in Hindi",
  "description": "पंचतंत्र की प्रसिद्ध कहानी #1...",
  "moral_hindi": "संकट के समय घबराने के बजाय बुद्धि और धैर्य से काम लेना चाहिए।",
  "character_bible": {
    "Protagonist": "Exact physical traits...",
    "Antagonist": "Exact physical traits...",
    "Setting": "Exact setting traits..."
  },
  "tags": ["panchatantra", "hindikahani", "moralstories", "indianstories", "kahaniyan", "ai", "aigenerated"],
  "sources": [{"name": "Panchatantra by Pandit Vishnu Sharma", "url": "https://en.wikipedia.org/wiki/Panchatantra"}],
  "wikipedia_articles": ["Panchatantra", "Hitopadesha"],
  "scenes": [
    {
      "scene_id": 1,
      "speaker_name": "कथावाचक",
      "speaker_role": "narrator",
      "speaker_gender": "male",
      "narration": "Three sentences of natural Devanagari Hindi narration...",
      "visual_subject": "Concrete summary of the scene action",
      "shot_prompts": [
        "Shot 1 (first 5s): Photorealistic 8k live-action shot of [exact character bible traits] doing [action in sentence 1]...",
        "Shot 2 (middle 5s): Photorealistic 8k live-action shot of [exact character bible traits] doing [action in sentence 2]...",
        "Shot 3 (final 5s): Photorealistic 8k live-action close-up of [exact character bible traits] doing [action in sentence 3]..."
      ],
      "prefer_media": "ai_image",
      "effect": "smooth_push_in",
      "on_screen_callout": "कहानी #1 • पंचतंत्र"
    }
  ]
}
"""


def _ensure_photorealistic_prompt(raw_prompt: str, fallback_subject: str, character_bible_summary: str = "") -> str:
    p = (raw_prompt or fallback_subject or "Ancient Indian forest river landscape at sunrise").strip()
    p = re.sub(
        r"\b(cartoon|pixar|3d\s*animation|animated|storybook\s*illustration|illustration|amar\s*chitra\s*katha|comic|2d|drawing|sketch)\b",
        "photorealistic live-action cinema",
        p,
        flags=re.IGNORECASE,
    )
    if character_bible_summary and character_bible_summary[:30].lower() not in p.lower():
        p = f"{p}. Visual consistency anchor: {character_bible_summary}"
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
            {"model": self.model, "messages": messages, "temperature": 0.7},
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
                print(f"[FoundryLLM] API notice ({exc}); using built-in 66-Shot Character-Consistent Hindi Story Engine.")
        else:
            print("[FoundryLLM] No LLM endpoint configured; using built-in 22-Scene (66-Shot) Character-Consistent Hindi Story Engine.")
        return self._build_fallback_hindi_storyboard(topic, story_number, collection_source)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=2, max=6))
    def _generate_storyboard_via_api(self, topic: str, story_number: int, collection_source: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Create a deep, immersive 6-to-9 minute long-form 16:9 Hindi story script for Story #{story_number}: {topic!r} "
                f"(Source Collection: {collection_source}).\n"
                "Requirements:\n"
                "1. Define a strict `character_bible` locking each character's exact visual appearance.\n"
                "2. Use **22 to 28 scenes** (750-1,050 spoken Hindi words total) in pure Devanagari Hindi (`narration`).\n"
                "3. Give EVERY scene `shot_prompts` containing **3 distinct photorealistic 8K live-action prompts** matched to the beginning, middle, and end of that scene's spoken narration, repeating the character's exact physical traits from `character_bible` in every prompt so characters stay 100% consistent across all 66+ images.\n"
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

        char_bible = storyboard.get("character_bible") or DEFAULT_MONKEY_CROCODILE_BIBLE
        storyboard["character_bible"] = char_bible
        bible_summary = "; ".join(f"{k}: {v}" for k, v in list(char_bible.items())[:4])

        tags = list(storyboard.get("tags") or [])
        for req_tag in ("panchatantra", "hindikahani", "moralstories", "indianstories", "ai", "aigenerated", "aistory"):
            if req_tag not in [t.lower() for t in tags]:
                tags.append(req_tag)
        storyboard["tags"] = tags

        scenes = storyboard.get("scenes") or []
        if not isinstance(scenes, list) or not scenes:
            raise RuntimeError("Storyboard contained no scenes.")

        effects = ["smooth_push_in", "cinema_pan_right", "smooth_pull_out", "cinema_pan_left"]
        total_shots = 0
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

            raw_shots = scene.get("shot_prompts") or []
            if not isinstance(raw_shots, list) or len(raw_shots) < 3:
                p1 = str(scene.get("image_prompt") or scene["visual_subject"]).strip()
                p2 = str(scene.get("image_prompt_b") or f"Medium action shot of {scene['visual_subject']}, {bible_summary}").strip()
                p3 = str(scene.get("image_prompt_c") or f"Close-up emotional reaction shot of {scene['visual_subject']}, {bible_summary}").strip()
                raw_shots = [p1, p2, p3]

            cleaned_shots = [
                _ensure_photorealistic_prompt(str(sp), scene["visual_subject"], bible_summary)
                for sp in raw_shots[:4]
            ]
            scene["shot_prompts"] = cleaned_shots
            scene["image_prompt"] = cleaned_shots[0]
            scene["image_prompt_b"] = cleaned_shots[1]
            scene["image_prompt_c"] = cleaned_shots[2]
            total_shots += len(cleaned_shots)

            scene["prefer_media"] = "ai_image"
            scene["effect"] = str(scene.get("effect") or effects[(idx - 1) % len(effects)]).strip()
            scene["on_screen_callout"] = str(scene.get("on_screen_callout") or f"कहानी #{story_number}").strip()

        print(
            f"[FoundryLLM] Prepared Character-Consistent Hindi Storyboard '{storyboard['title']}' "
            f"({len(scenes)} scenes, {total_shots} scene-locked 16:9 AI visual shots, Story #{story_number})."
        )
        return storyboard

    def _build_fallback_hindi_storyboard(self, topic: str, story_number: int, collection_source: str) -> Dict[str, Any]:
        """
        22-Scene (66-Shot) Character-Consistent Long-Form Hindi Storyboard where EVERY 5-second visual shot
        explicitly locks the physical traits of Raktamukha (single adult golden-brown Rhesus macaque monkey) and
        Karalamukha (14-foot dark olive-green Indian mugger crocodile) to the exact spoken clause!
        """
        clean_name = re.sub(r"^(?:कहानी|story)\s*#\s*\d+\s*[:\-—]\s*", "", topic, flags=re.IGNORECASE).strip()
        hindi_title_part = clean_name.split("(")[0].replace("Panchatantra —", "").replace("Hitopadesha —", "").strip()
        if not hindi_title_part:
            hindi_title_part = "बंदर और मगरमच्छ की पंचतंत्र कथा"

        M_DESC = "a single adult golden-brown Rhesus macaque monkey (Raktamukha) with warm amber eyes and reddish-tan face (no baby monkey)"
        C_DESC = "a massive 14-foot dark olive-green Indian mugger crocodile (Karalamukha) with rough armored scales and yellow eyes"
        W_DESC = "a large dark-bronze female Indian mugger crocodile (Karalamukha's wife) with sharp golden eyes"
        TREE_DESC = "the giant ancient Jamun berry tree heavy with dark purple fruits on the Narmada riverbank"

        scenes = [
            {
                "scene_id": 1,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": f"नमस्कार दोस्तों! हमारी पंचतंत्र की एक सौ एक कहानियों की इस विशेष श्रृंखला की कहानी संख्या {story_number} में आपका स्वागत है। आज हम सुनने जा रहे हैं चतुर बंदर रक्तमुख और मगरमच्छ करालमुख की वह अमर कहानी, जो हमें संकट में धैर्य और बुद्धि का सही मार्ग दिखाती है।",
                "visual_subject": "Establishing shots of the Narmada river, the Jamun tree, and both main characters Raktamukha the monkey and Karalamukha the crocodile",
                "shot_prompts": [
                    f"Wide aerial sunrise cinema shot of {TREE_DESC} beside the sparkling turquoise Narmada river winding through a misty Indian jungle",
                    f"Medium wildlife cinema shot of {M_DESC} sitting alertly on a thick branch of {TREE_DESC} holding ripe dark-purple jamun berries at sunrise",
                    f"Dramatic river-level wildlife shot of {C_DESC} swimming near the sandy bank of the Narmada river below {TREE_DESC} in golden morning light",
                ],
                "on_screen_callout": f"कहानी #{story_number} • पंचतंत्र",
            },
            {
                "scene_id": 2,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "प्राचीन काल में नर्मदा नदी के पावन तट पर एक बहुत ही विशाल और घना जामुन का वृक्ष खड़ा था। उस पेड़ की डालियाँ नदी के ठंडे जल को छूती थीं, और उस पर साल भर बड़े-बड़े, काले और शहद जैसे रसीले जामुन लगते थे।",
                "visual_subject": "The giant Jamun tree on the Narmada riverbank with low branches touching the water and clusters of ripe purple berries",
                "shot_prompts": [
                    f"Wide establishing cinema shot of {TREE_DESC} standing majestically on the mossy Narmada riverbank in warm morning sunlight",
                    f"Medium cinema shot of low leafy branches of {TREE_DESC} dipping into the clear flowing water of the Narmada river with ripples",
                    f"Macro close-up cinema shot of clusters of glossy dark-purple ripe jamun berries hanging on {TREE_DESC} with morning dewdrops sparkling",
                ],
                "on_screen_callout": "नर्मदा तट का विशाल वृक्ष",
            },
            {
                "scene_id": 3,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "उसी विशाल पेड़ पर रक्तमुख नाम का एक बड़ा ही फुर्तीला, नेक और बुद्धिमान बंदर रहता था। वह दिन भर पेड़ की ऊँची टहनियों पर उछल-कूद करता और मीठे-मीठे जामुन खाकर आनंद से अपना जीवन बिताता था।",
                "visual_subject": "Raktamukha the adult golden-brown macaque monkey living happily on the Jamun tree and eating sweet purple berries",
                "shot_prompts": [
                    f"Full-body wildlife cinema shot of {M_DESC} perched high on a sunlit branch of {TREE_DESC} overlooking the Narmada river",
                    f"Dynamic action wildlife shot of {M_DESC} leaping agilely between two mossy branches of {TREE_DESC} surrounded by purple jamun fruits",
                    f"Close-up wildlife portrait of {M_DESC} happily eating a ripe dark-purple jamun berry on {TREE_DESC} in warm golden light",
                ],
                "on_screen_callout": "बुद्धिमान बंदर रक्तमुख",
            },
            {
                "scene_id": 4,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "एक दोपहर जब सूरज की धूप तेज़ थी, नर्मदा की गहरी लहरों को चीरता हुआ करालमुख नाम का एक विशाल मगरमच्छ नदी के किनारे आया। वह सुबह से भोजन की तलाश में भटक रहा था और भूख तथा थकान से बिल्कुल निढाल हो चुका था।",
                "visual_subject": "Karalamukha the massive olive-green crocodile swimming out of the river and resting exhausted on the sandy bank under the Jamun tree",
                "shot_prompts": [
                    f"Wide river-level cinema shot of {C_DESC} cutting through the blue waves of the Narmada river toward the shaded shore under bright midday sun",
                    f"Medium wildlife shot of {C_DESC} crawling slowly onto the warm sandy riverbank right beneath the roots of {TREE_DESC}",
                    f"Close-up cinema shot of {C_DESC} resting exhausted on the shaded riverbank sand with wet glistening scales while {M_DESC} looks down from the branch above",
                ],
                "on_screen_callout": "थका हुआ मगरमच्छ करालमुख",
            },
            {
                "scene_id": 5,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "अरे भाई मगरमच्छ! तुम तो बहुत थके हुए और भूखे लग रहे हो। इस पेड़ की ठंडी छाँव में तुम मेरे अतिथि हो। रुको, मैं अभी तुम्हारे लिए इस पेड़ के सबसे मीठे और ताज़े जामुन तोड़कर नीचे गिराता हूँ!",
                "visual_subject": "Raktamukha the monkey speaking kindly from the branch and plucking ripe purple jamun fruits to toss down to Karalamukha the crocodile",
                "shot_prompts": [
                    f"Low-angle wildlife shot looking up at {M_DESC} leaning down kindly from a low branch of {TREE_DESC} calling out to {C_DESC} on the bank below",
                    f"Medium wildlife shot of {M_DESC} plucking ripe dark-purple jamun fruits from {TREE_DESC} with both hands",
                    f"Two-shot cinema frame showing {M_DESC} on the overhanging branch tossing purple jamun berries down to {C_DESC} waiting on the sandy riverbank below",
                ],
                "on_screen_callout": "अतिथि धर्म का पालन",
            },
            {
                "scene_id": 6,
                "speaker_name": "मगरमच्छ करालमुख",
                "speaker_role": "male_antagonist",
                "speaker_gender": "male",
                "narration": "धन्यवाद भाई! आज से पहले किसी ने मेरे साथ इतना मधुर व्यवहार नहीं किया। सचमुच, ये जामुन तो अमृत के समान मीठे हैं! इन्हें खाकर मेरी सारी भूख और थकान पल भर में दूर हो गई।",
                "visual_subject": "Karalamukha the crocodile eating the sweet purple jamun berries on the sandy bank and looking up gratefully at Raktamukha the monkey",
                "shot_prompts": [
                    f"Close-up wildlife shot of {C_DESC} on the sandy riverbank surrounded by fresh dark-purple jamun berries fallen from {TREE_DESC}",
                    f"Medium wildlife cinema shot of {C_DESC} eating the ripe purple jamun fruits on the riverbank with a calm grateful expression",
                    f"Warm two-shot of {C_DESC} on the sandy shore looking up appreciatively at {M_DESC} sitting on the low root of {TREE_DESC}",
                ],
                "on_screen_callout": "गहरी मित्रता की शुरुआत",
            },
            {
                "scene_id": 7,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "उस दिन के बाद से करालमुख रोज़ नदी के किनारे आने लगा। रक्तमुख उसे ढेर सारे मीठे जामुन खिलाता और दोनों घंटों बैठकर देश-दुनिया और जंगल की रोचक बातें करते। देखते ही देखते उन दोनों के बीच बहुत गहरी मित्रता हो गई।",
                "visual_subject": "Raktamukha the monkey and Karalamukha the crocodile sitting side by side on the riverbank every day sharing jamun fruits as close friends",
                "shot_prompts": [
                    f"Wide golden-hour cinema shot of {M_DESC} sitting on a riverside rock right next to {C_DESC} basking at the water's edge under {TREE_DESC}",
                    f"Medium wildlife shot of {M_DESC} sharing a pile of dark-purple jamun fruits with {C_DESC} on the riverbank sand",
                    f"Warm sunset silhouette shot of {M_DESC} and {C_DESC} together by the calm Narmada river reflecting orange twilight skies",
                ],
                "on_screen_callout": "सच्चे मित्रों का साथ",
            },
            {
                "scene_id": 8,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "मित्र करालमुख! आज मैंने पेड़ की सबसे ऊँची डाल से ये विशेष रसीले जामुन चुने हैं। इन्हें तुम अपनी पत्नी के लिए घर ले जाओ। जब तुम्हारी पत्नी यह मीठा उपहार चखेगी, तो उसे भी बहुत प्रसन्नता होगी!",
                "visual_subject": "Raktamukha the monkey placing a bundle of ripe purple jamun berries on a lotus leaf for Karalamukha to carry across the river to his wife",
                "shot_prompts": [
                    f"Medium wildlife shot of {M_DESC} high in {TREE_DESC} picking the largest glossy purple jamun berries in golden afternoon light",
                    f"Close-up shot of {M_DESC} placing a generous heap of dark-purple jamun berries onto a broad green lotus leaf beside {C_DESC} at the riverbank",
                    f"Wide cinema shot of {C_DESC} swimming across the golden Narmada river carrying theripe purple jamun fruits toward the opposite shore",
                ],
                "on_screen_callout": "भाभी के लिए मीठा उपहार",
            },
            {
                "scene_id": 9,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "करालमुख बड़े उत्साह के साथ वे मीठे जामुन लेकर नदी के दूसरे छोर पर बने अपने घर पहुँचा और अपनी पत्नी को भेंट किए। परंतु जैसे ही उसकी पत्नी ने उन रसीले फलों का स्वाद चखा, उसके मन में एक भयानक लालच जाग उठा।",
                "visual_subject": "Karalamukha arriving at the rocky river cove and giving the purple jamun berries to his wife, whose eyes gleam with greed",
                "shot_prompts": [
                    f"Wide twilight cinema shot of {C_DESC} crawling onto a secluded rocky river island where {W_DESC} is waiting near the water",
                    f"Medium wildlife shot of {C_DESC} presenting the dark-purple jamun berries on a green leaf to {W_DESC} on the mossy rock",
                    f"Extreme close-up cinema shot of {W_DESC} tasting the purple jamun fruit, her sharp golden reptilian eyes narrowing with cunning greed",
                ],
                "on_screen_callout": "मन में जागा लालच",
            },
            {
                "scene_id": 10,
                "speaker_name": "मगरमच्छ की पत्नी",
                "speaker_role": "female_character",
                "speaker_gender": "female",
                "narration": "सुना जी! ज़रा सोचिए, जो बंदर प्रतिदिन ऐसे अमृत जैसे मीठे जामुन खाता है, उसका अपना कलेजा कितना मीठा और स्वादिष्ट होगा! यदि आप मुझसे सच्चा प्रेम करते हैं, तो कल ही उस बंदर को यहाँ लाइए, मुझे उसका कलेजा खाना है!",
                "visual_subject": "The female crocodile aggressively demanding the monkey's heart from Karalamukha on the dark rocky riverbank",
                "shot_prompts": [
                    f"Dramatic two-shot of {W_DESC} raising her head aggressively as she confronts {C_DESC} on the dark wet river rocks at dusk",
                    f"Close-up low-angle shot of {W_DESC} with open jaws demanding a sacrifice beside swirling dark river water",
                    f"Reaction shot of {C_DESC} looking shocked and troubled on the rocky shore as dark storm clouds gather over the river",
                ],
                "on_screen_callout": "पत्नी की कठोर ज़िद",
            },
            {
                "scene_id": 11,
                "speaker_name": "मगरमच्छ करालमुख",
                "speaker_role": "male_antagonist",
                "speaker_gender": "male",
                "narration": "अरे प्रिये, तुम यह कैसी अनर्थकारी बात कर रही हो? रक्तमुख मेरा धर्म-भाई और परम मित्र है! वह रोज़ मुझे निस्वार्थ भाव से मीठे फल खिलाता है। मैं अपने ही मित्र के साथ ऐसा घोर विश्वासघात कैसे कर सकता हूँ?",
                "visual_subject": "Karalamukha pleading with his wife not to force him to betray his loyal friend Raktamukha",
                "shot_prompts": [
                    f"Medium moonlit wildlife shot of {C_DESC} facing {W_DESC} on the rocky bank with a sorrowful reluctant posture",
                    f"Memory flashback shot of {M_DESC} kindly offering purple jamun berries from {TREE_DESC} in warm sunlight",
                    f"Close-up moonlit portrait of {C_DESC} half-submerged in dark river water looking conflicted and sad",
                ],
                "on_screen_callout": "धर्मसंकट और मित्रता",
            },
            {
                "scene_id": 12,
                "speaker_name": "मगरमच्छ की पत्नी",
                "speaker_role": "female_character",
                "speaker_gender": "female",
                "narration": "मैं कुछ नहीं जानती! मुझे गंभीर रोग हो गया है और वैद्य ने कहा है कि केवल बंदर के कलेजे से ही मेरे प्राण बच सकते हैं। यदि कल सूर्यास्त से पहले तुम उस बंदर को लेकर नहीं आए, तो मैं अन्न-जल त्याग कर अपने प्राण दे दूँगी!",
                "visual_subject": "The female crocodile feigning severe illness on the rock slab and threatening to starve herself unless Karalamukha brings the monkey",
                "shot_prompts": [
                    f"Dramatic dawn wildlife shot of {W_DESC} lying motionless on a cold misty rock ledge feigning illness while {C_DESC} watches anxiously",
                    f"Close-up of {W_DESC} turning her head away stubbornly on the wet stone in thick morning fog",
                    f"Wide moody shot of {C_DESC} slowly sliding into the misty river current at dawn to carry out the reluctant plot",
                ],
                "on_screen_callout": "त्रिया-हठ के आगे विवश",
            },
            {
                "scene_id": 13,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "पत्नी के इस हठ के आगे करालमुख अंततः विवश हो गया। भारी मन और कपटपूर्ण योजना के साथ वह अगले दिन जामुन के पेड़ के पास पहुँचा। उसने अपने चेहरे पर झूठी मुस्कान सजाई और बंदर को पुकारा।",
                "visual_subject": "Karalamukha swimming across the misty river to the Jamun tree and calling out to Raktamukha the monkey",
                "shot_prompts": [
                    f"Wide morning cinema shot of {C_DESC} swimming across the misty Narmada river toward {TREE_DESC}",
                    f"Medium wildlife shot of {C_DESC} arriving at the shallow bank below {TREE_DESC} and looking up into the branches",
                    f"Top-down wildlife shot from the branch showing {M_DESC} looking down cheerfully at {C_DESC} waiting in the water below",
                ],
                "on_screen_callout": "छलपूर्ण निमंत्रण",
            },
            {
                "scene_id": 14,
                "speaker_name": "मगरमच्छ करालमुख",
                "speaker_role": "male_antagonist",
                "speaker_gender": "male",
                "narration": "मित्र रक्तमुख! तुम्हारे भेजे हुए मीठे जामुन खाकर तुम्हारी भाभी अत्यंत प्रसन्न हुईं। उन्होंने आज तुम्हें हमारे घर विशेष दावत पर आमंत्रित किया है। आओ, मेरी पीठ पर बैठ जाओ, मैं तुम्हें सुरक्षित नदी की सैर कराते हुए घर ले चलूँगा।",
                "visual_subject": "Karalamukha inviting Raktamukha onto his back at the river shore for a feast across the river",
                "shot_prompts": [
                    f"Close-up wildlife shot of {C_DESC} at the water's edge looking up persuasively toward {M_DESC} on the branch",
                    f"Medium wildlife shot of {M_DESC} climbing down the gnarled roots of {TREE_DESC} toward {C_DESC}'s broad back",
                    f"Clear two-shot of {M_DESC} stepping trustingly onto the rough armored back of {C_DESC} in the shallow river water",
                ],
                "on_screen_callout": "पीठ पर बैठकर यात्रा",
            },
            {
                "scene_id": 15,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "निश्छल और भोला बंदर अपने मित्र की बातों पर पूरा भरोसा करके खुशी-खुशी उसकी पीठ पर बैठ गया। मगरमच्छ तेजी से तैरता हुआ किनारे से दूर, नर्मदा नदी की सबसे गहरी और तेज़ धार के बीचों-बीच पहुँच गया।",
                "visual_subject": "Raktamukha the monkey riding on Karalamukha the crocodile's back into the deep turbulent middle of the Narmada river",
                "shot_prompts": [
                    f"Wide cinematic shot of {M_DESC} sitting on the back of {C_DESC} as the crocodile swims away from the green shore into open river water",
                    f"Medium tracking shot of {M_DESC} riding on {C_DESC}'s back in the middle of the wide turbulent Narmada river far from land",
                    f"Close-up of {M_DESC} perched on {C_DESC}'s wet scaly back looking down nervously at the deep swirling dark-blue river currents",
                ],
                "on_screen_callout": "गहरे पानी में संकट",
            },
            {
                "scene_id": 16,
                "speaker_name": "मगरमच्छ करालमुख",
                "speaker_role": "male_antagonist",
                "speaker_gender": "male",
                "narration": "मित्र रक्तमुख, मुझे क्षमा कर देना! अब यहाँ चारों ओर गहरा जल है और तुम कहीं भाग नहीं सकते। सच यह है कि मेरी पत्नी ने कोई दावत नहीं रखी है, बल्कि वह तुम्हारा मीठा कलेजा खाना चाहती है!",
                "visual_subject": "Karalamukha revealing the betrayal in the middle of the deep river while Raktamukha sits trapped on his back",
                "shot_prompts": [
                    f"Dramatic close-up of {C_DESC}'s head in deep churning river water as waves splash around his jaws under dark clouds",
                    f"Tense two-shot in the middle of the deep river showing {M_DESC} sitting trapped on the back of {C_DESC} surrounded by deep water",
                    f"Extreme close-up of {M_DESC}'s wide amber eyes in shock as he sits on {C_DESC}'s back in the middle of the river",
                ],
                "on_screen_callout": "विश्वासघात का खुलासा",
            },
            {
                "scene_id": 17,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "यह भयानक सच सुनकर एक पल के लिए तो रक्तमुख के पैरों तले ज़मीन खिसक गई। चारों तरफ गहरा पानी था और तैरना उसे आता नहीं था। किंतु उसने तुरंत अपने मन को शांत किया और संकट से निकलने की एक चतुर युक्ति सोची।",
                "visual_subject": "Raktamukha calming his mind while sitting on the crocodile's back in deep water and devising a clever plan",
                "shot_prompts": [
                    f"High-angle cinema shot looking down at {M_DESC} sitting on {C_DESC}'s back isolated in the vast deep river water",
                    f"Close-up portrait of {M_DESC} taking a calm breath, his amber eyes turning sharp and composed as a beam of sunlight hits his face",
                    f"Over-the-shoulder shot from {M_DESC} on {C_DESC}'s back looking across the water toward {TREE_DESC} on the distant shore",
                ],
                "on_screen_callout": "संकट में अद्भुत धैर्य",
            },
            {
                "scene_id": 18,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "अरे नादान मित्र करालमुख! इतनी सी बात थी तो तुमने मुझे पेड़ के पास ही क्यों नहीं बता दिया? भाभी की खुशी के लिए तो मैं खुशी-खुशी अपना कलेजा दे देता! परंतु हम बंदर तो उछल-कूद करते समय अपना कलेजा जामुन के पेड़ की खोखल में ही रख छोड़ते हैं!",
                "visual_subject": "Raktamukha calmly telling Karalamukha that he left his heart inside the hollow of the Jamun tree on the shore",
                "shot_prompts": [
                    f"Medium two-shot of {M_DESC} sitting calmly on {C_DESC}'s back in the river, gesturing toward the shore with a confident expression",
                    f"Close-up of {C_DESC} pausing in the river water and turning his head slightly to listen to {M_DESC} on his back",
                    f"Detailed cinema shot of a deep dark hollow cavity inside the massive trunk of {TREE_DESC} on the riverbank",
                ],
                "on_screen_callout": "कलेजा पेड़ पर होने की युक्ति",
            },
            {
                "scene_id": 19,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "इस समय तो मेरा शरीर बिल्कुल खाली है। यदि तुम मुझे बिना कलेजे के ले जाओगे, तो भाभी तुम पर बहुत क्रोधित होंगी। इसलिए जल्दी से वापस किनारे चलो, ताकि मैं पेड़ से अपना मीठा कलेजा उतार लाऊँ!",
                "visual_subject": "Karalamukha turning around in the river with Raktamukha on his back and swimming fast toward the Jamun tree shore",
                "shot_prompts": [
                    f"Close-up of {M_DESC} on {C_DESC}'s back pointing urgently toward {TREE_DESC} on the riverbank",
                    f"Dynamic wildlife action shot of {C_DESC} making a sharp U-turn in the river current with {M_DESC} riding on his back, white water splashing",
                    f"Tracking shot of {C_DESC} carrying {M_DESC} on his back swimming rapidly toward the roots of {TREE_DESC} at the shore",
                ],
                "on_screen_callout": "मूर्ख मगरमच्छ लौटा किनारे",
            },
            {
                "scene_id": 20,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": "मूर्ख मगरमच्छ बंदर की इस चतुर बात पर तुरंत विश्वास कर बैठा और तेज़ी से तैरता हुआ वापस नदी के किनारे ले आया। जैसे ही मगरमच्छ किनारे के पास पहुँचा, रक्तमुख ने बिजली की फुर्ती से एक लंबी छलांग लगाई और सीधा पेड़ की सबसे ऊँची डाल पर जा पहुँचा!",
                "visual_subject": "Raktamukha leaping off Karalamukha's back at the riverbank and climbing high into the safe branches of the Jamun tree",
                "shot_prompts": [
                    f"Medium shot of {C_DESC} reaching the shallow sandy bank right beneath {TREE_DESC} with {M_DESC} poised on his back",
                    f"High-speed action wildlife photograph of {M_DESC} leaping mid-air off {C_DESC}'s back onto the lower trunk of {TREE_DESC}, water droplets frozen in sunlight",
                    f"Low-angle shot looking up as {M_DESC} climbs swiftly to a high safe branch of {TREE_DESC} while {C_DESC} looks up from the water below",
                ],
                "on_screen_callout": "बिजली सी तेज़ छलांग",
            },
            {
                "scene_id": 21,
                "speaker_name": "बंदर रक्तमुख",
                "speaker_role": "male_protagonist",
                "speaker_gender": "male",
                "narration": "अरे मूर्ख और विश्वासघाती करालमुख! भला दुनिया में किसी प्राणी के दो कलेजे होते हैं क्या, या कोई अपना कलेजा शरीर से निकाल कर पेड़ पर रखता है? जा, अब लौट जा! जो मित्र के विश्वास की रक्षा नहीं कर सकता, उससे मित्रता रखना महामूर्खता है!",
                "visual_subject": "Raktamukha scolding the ashamed crocodile Karalamukha from the high branch of the Jamun tree as the crocodile retreats into the river",
                "shot_prompts": [
                    f"Medium wildlife shot of {M_DESC} sitting safely high on a branch of {TREE_DESC} looking down boldly in golden sunset light",
                    f"High-angle shot looking down at {C_DESC} waiting foolishly in the shallow water below {TREE_DESC}, looking up in shame",
                    f"Wide twilight shot of {C_DESC} slowly sinking back into the dark Narmada river water alone while {M_DESC} watches from high in {TREE_DESC}",
                ],
                "on_screen_callout": "विश्वासघाती को करारा उत्तर",
            },
            {
                "scene_id": 22,
                "speaker_name": "कथावाचक",
                "speaker_role": "narrator",
                "speaker_gender": "male",
                "narration": f"दोस्तों, पंचतंत्र की इस पहली अमर कहानी से हमें जीवन की दो सबसे बड़ी सीख मिलती हैं—पहली, कठिन से कठिन संकट आने पर भी कभी घबराना नहीं चाहिए, क्योंकि शांत दिमाग और सूझबूझ से हर मुसीबत को हराया जा सकता है। और दूसरी, कपटी मित्रों से सदैव सावधान रहना चाहिए। अगली कहानी में फिर मिलेंगे!",
                "visual_subject": "Peaceful sunset over Raktamukha safe in the Jamun tree and the tranquil Narmada river valley",
                "shot_prompts": [
                    f"Warm golden sunset portrait of {M_DESC} sitting peacefully on the top branch of {TREE_DESC} overlooking the Narmada river",
                    f"Wide epic cinema landscape of {TREE_DESC} and the tranquil Narmada river valley glowing under a dramatic orange and purple sunset sky",
                    f"Atmospheric twilight close-up of glowing brass oil diyas floating gently on the calm Narmada river water at dusk",
                ],
                "on_screen_callout": "कहानी की अनमोल सीख",
            },
        ]

        if story_number > 1 and "बंदर और मगरमच्छ" not in topic and "Monkey and the Crocodile" not in topic:
            custom_bible = {
                "Main Characters": f"Consistent photorealistic live-action characters of {clean_name} in ancient India",
                "Setting": f"Authentic ancient Indian forest and village setting of {clean_name}",
            }
            for s in scenes:
                s["narration"] = s["narration"].replace("बंदर रक्तमुख और मगरमच्छ करालमुख", hindi_title_part)
                s["visual_subject"] = f"Photorealistic live-action Indian cinema scene for {clean_name} (Scene {s['scene_id']})"
                s["shot_prompts"] = [
                    f"Wide establishing 8k live-action cinema shot of {clean_name}, scene {s['scene_id']} opening action: {s['narration'][:60]}",
                    f"Medium 8k live-action cinema action shot of {clean_name}, scene {s['scene_id']} middle beat, consistent characters",
                    f"Close-up 8k live-action cinema reaction shot of {clean_name}, scene {s['scene_id']} climax beat, dramatic lighting",
                ]
        else:
            custom_bible = DEFAULT_MONKEY_CROCODILE_BIBLE

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
            "character_bible": custom_bible,
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
