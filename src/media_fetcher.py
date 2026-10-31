"""
GPT-Image-1-Mini Primary Widescreen 16:9 Visual Engine (`src/media_fetcher.py` — v4.0 GPT-Image-1-Mini Default Edition):
1. **Default Primary Engine: `gpt-image-1-mini` (`FOUNDRY_IMAGE_MODEL="gpt-image-1-mini"`)**:
   - Uses your deployed `gpt-image-1-mini` model on Microsoft Foundry / Azure OpenAI (`FOUNDRY_ENDPOINT` / `FOUNDRY_TEXT_ENDPOINT`
     and `FOUNDRY_API_KEY` / `FOUNDRY_TEXT_API_KEY`) as the #1 primary image generator for every scene!
   - Requests native landscape **`size="1536x1024"`** (`3:2` widescreen) with **`quality="high"`** (falling back automatically to `"medium"` or `"auto"` if needed) and converts it cleanly to full-bleed `1920x1080` (`16:9` YouTube HD) with minimal top/bottom trim and zero sidebars!
2. **Maximum Scene Fidelity & Character Consistency (`_build_gpt_image_scene_prompt`)**:
   - Because `gpt-image-1-mini` has a GPT-4o multimodal language encoder, it understands long, structured instructions far better than Flux!
   - Every single shot prompt sent to `gpt-image-1-mini` includes:
     (a) **EXACT SPOKEN SCENE ACTION**: The literal visual action happening in that specific 5-second beat + the Hindi narration context.
     (b) **LOCKED CHARACTER BIBLE**: Exact physical traits of every character in the story so characters stay 100% consistent across all shots.
     (c) **CINEMA 16:9 STYLE**: High-resolution 16:9 YouTube storytelling cinematography, natural lighting, zero text/watermarks.
3. **Fallback Chain**: If `FOUNDRY_ENDPOINT` is not set or quota is reached, falls back to `FREE_IMAGE_API_URL` (`saurav-z/free-image-generation-api`), Direct Cloudflare REST AI, Pollinations, or procedural 16:9 plates.
"""
import base64
import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote, urlparse
import numpy as np
from PIL import Image, ImageEnhance
import requests

try:
    from archive_media import ArchiveMediaFinder, rank_candidates
except ImportError:
    from src.archive_media import ArchiveMediaFinder, rank_candidates

USER_AGENT = (
    "IndianKahaniLongformBot/4.0 "
    "(GPT-Image-1-Mini-Primary; CharacterConsistent16x9Pipeline/4.0)"
)

DEFAULT_CHARACTER_BIBLE_TEXT = (
    "Raktamukha (Monkey): single adult slender golden-brown Indian Rhesus macaque monkey with warm amber eyes, "
    "reddish-tan face, and expressive intelligent posture (strictly ONE adult monkey, no baby monkey). "
    "Karalamukha (Crocodile): massive 14-foot dark olive-green Indian mugger crocodile with heavy armored dorsal ridges, "
    "broad snout, and deep yellow reptilian eyes. "
    "Crocodile's Wife: large dark-bronze female Indian mugger crocodile with sharp golden eyes on wet mossy river rocks. "
    "Setting: ancient Narmada riverbank in central India with a giant gnarled Jamun tree heavy with dark-purple berries overhanging turquoise water."
)


class MediaFetcher:
    def __init__(
        self,
        topic: str,
        era_keywords: Optional[List[str]] = None,
        wikipedia_articles: Optional[List[str]] = None,
        video_pool_queries: Optional[List[str]] = None,
        required_visual_keywords: Optional[List[str]] = None,
        is_space_or_science: bool = False,
        llm_client: Any = None,
        character_bible: Optional[Dict[str, str]] = None,
    ):
        self.topic = topic
        self.era_keywords = era_keywords or []
        self.wikipedia_articles = (wikipedia_articles or ["Panchatantra", "Hitopadesha"])[:5]
        self.video_pool_queries = video_pool_queries or []
        self.llm_client = llm_client
        self.character_bible = character_bible or {}
        if self.character_bible:
            self.character_bible_str = " | ".join(f"{k}: {v}" for k, v in self.character_bible.items())
        else:
            self.character_bible_str = DEFAULT_CHARACTER_BIBLE_TEXT

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.media_cache_dir = Path(os.getenv("MEDIA_CACHE_DIR", ".cache/media"))
        self.media_cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. PRIMARY: Deployed gpt-image-1-mini on Microsoft Foundry / Azure OpenAI
        self.foundry_endpoint = (
            os.getenv("FOUNDRY_IMAGE_ENDPOINT")
            or os.getenv("FOUNDRY_ENDPOINT")
            or os.getenv("FOUNDRY_TEXT_ENDPOINT")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        ).strip().rstrip("/")
        self.foundry_key = (
            os.getenv("FOUNDRY_IMAGE_API_KEY")
            or os.getenv("FOUNDRY_API_KEY")
            or os.getenv("FOUNDRY_TEXT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        raw_img_model = (os.getenv("FOUNDRY_IMAGE_MODEL") or "gpt-image-1-mini").strip()
        # Automatically upgrade legacy MAI-Image default to gpt-image-1-mini unless explicitly overridden
        if not raw_img_model or raw_img_model == "MAI-Image-2.6-Flash":
            self.image_model = "gpt-image-1-mini"
        else:
            self.image_model = raw_img_model
        self.image_quality = (os.getenv("GPT_IMAGE_QUALITY") or "high").strip().lower()

        # 2. SECONDARY FALLBACK: Cloudflare Worker (saurav-z/free-image-generation-api)
        self.free_image_api_url = (
            os.getenv("FREE_IMAGE_API_URL")
            or os.getenv("CLOUDFLARE_WORKER_URL")
            or ""
        ).strip().rstrip("/")
        self.free_image_api_key = (
            os.getenv("FREE_IMAGE_API_KEY")
            or os.getenv("CLOUDFLARE_WORKER_API_KEY")
            or ""
        ).strip()
        self.free_image_model = (
            os.getenv("FREE_IMAGE_MODEL")
            or "@cf/black-forest-labs/flux-1-schnell"
        ).strip()

        self.cf_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.cf_api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

        raw_max_ai = os.getenv("MAX_AI_IMAGES_PER_VIDEO") or os.getenv("MAX_AI_IMAGES_PER_SHORT")
        if raw_max_ai is None or str(raw_max_ai).strip() == "":
            raw_max_ai = "90"
        try:
            self.max_ai_images = max(0, int(str(raw_max_ai).strip()))
        except (TypeError, ValueError):
            self.max_ai_images = 90
        if (self.foundry_endpoint or self.free_image_api_url) and 0 < self.max_ai_images < 66:
            self.max_ai_images = 80

        self.gpt_images_used = 0
        self.free_worker_images_used = 0
        self.ai_images_used = 0
        self.video_clips_used = 0
        self.archival_photos_used = 0
        self.reframed_cuts_used = 0
        self.used_urls: Set[str] = set()
        self.used_hashes: Set[str] = set()
        self.attributions: List[str] = []
        self.scene_audit_log: List[Dict[str, Any]] = []
        self.archive_finder = ArchiveMediaFinder(session=self.session, cache_dir=self.media_cache_dir)

    @staticmethod
    def _is_valid_photographic_image(rgb_img: Image.Image) -> bool:
        small = rgb_img.resize((200, 200), Image.Resampling.BILINEAR)
        gray = np.asarray(small.convert("L"), dtype=np.float32)
        mean_l = float(gray.mean())
        std_l = float(gray.std())
        if mean_l < 8.0 or std_l < 6.0:
            return False
        return True

    @staticmethod
    def _save_as_full_bleed_16x9(img: Image.Image, dest_path: Path, vertical_anchor: float = 0.42) -> Path:
        """
        Masters any generated image (such as `1536x1024` landscape from `gpt-image-1-mini` or `1024x1024`)
        into a crisp edge-to-edge `1920x1080` (`16:9` Full HD) frame with zero sidebars.
        Because `1536x1024` is already 1.50:1 landscape, converting to `1920x1080` (`1.77:1`) preserves
        virtually the entire horizontal width of `gpt-image-1-mini`'s composition!
        """
        target_w, target_h = 1920, 1080
        rgb = img.convert("RGB")
        w, h = rgb.size

        scale = max(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(target_w, int(round(w * scale)))
        new_h = max(target_h, int(round(h * scale)))
        resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

        left = (new_w - target_w) // 2
        top = min(max(0, int((new_h - target_h) * vertical_anchor)), new_h - target_h)
        cropped = resized.crop((left, top, left + target_w, top + target_h))

        cropped = ImageEnhance.Contrast(cropped).enhance(1.04)
        cropped = ImageEnhance.Sharpness(cropped).enhance(1.06)
        cropped.save(dest_path, quality=96)
        return dest_path

    @staticmethod
    def _scene_terms(scene: Dict[str, Any]) -> Set[str]:
        stop = {
            "this", "that", "with", "from", "into", "over", "were", "have", "their", "about",
            "story", "scene", "history", "historical", "photo", "video", "footage", "archive", "image",
        }
        raw = " ".join([
            str(scene.get("visual_subject", "")),
            str(scene.get("video_query", "")),
            str(scene.get("wikimedia_query", "")),
            " ".join(map(str, scene.get("visual_keywords", []))),
        ])
        return {w.lower() for w in re.findall(r"[A-Za-z0-9]{3,}", raw) if w.lower() not in stop}

    @classmethod
    def _relevance_score(cls, candidate: Dict[str, Any], terms: Set[str]) -> int:
        text = (str(candidate.get("title", "")) + " " + str(candidate.get("desc", ""))).lower()
        if not terms or not text:
            return 0
        return sum(3 if term in str(candidate.get("title", "")).lower() else 1 for term in terms if term in text)

    def _build_gpt_image_scene_prompt(
        self,
        raw_shot_prompt: str,
        scene: Dict[str, Any],
        shot_idx: int,
        total_shots: int,
    ) -> str:
        """
        Builds a rich, structured scene-faithful prompt specifically optimized for `gpt-image-1-mini`:
        - Explicitly binds the exact action happening in the current spoken beat + Hindi narration context
        - Injects the story's `character_bible` so every character has identical physical traits in every frame
        - Enforces 16:9 widescreen cinematic framing (`1536x1024` -> `1920x1080`) with all subjects centered in the middle 80% vertical safe zone.
        """
        subject = str(scene.get("visual_subject") or self.topic).strip()
        narration = str(scene.get("narration") or "").strip()
        speaker = str(scene.get("speaker_name") or "कथावाचक").strip()

        clean_shot = re.sub(
            r"\b(cartoon|pixar|3d\s*animation|animated|storybook\s*illustration|illustration|amar\s*chitra\s*katha|comic|2d|drawing|sketch)\b",
            "photorealistic live-action cinema",
            (raw_shot_prompt or subject).strip(),
            flags=re.IGNORECASE,
        )

        prompt = (
            f"MANDATORY SCENE ACTION (Beat {shot_idx} of {total_shots}): {clean_shot}. "
            f"SPOKEN HINDI NARRATION CONTEXT ({speaker}): \"{narration}\". "
            f"LOCKED CHARACTER CONSISTENCY BIBLE (must match exact appearance in every frame): {self.character_bible_str}. "
            "VISUAL STYLE & 16:9 FRAMING: Ultra-high-quality photorealistic 8K live-action cinema still frame, "
            "shot on ARRI Alexa 65 with 35mm anamorphic widescreen lens, natural textures, rich volumetric golden lighting, "
            "wide 16:9 horizontal YouTube composition with all characters and key action framed cleanly inside the central "
            "horizontal safe zone. Strictly photorealistic live-action — NOT cartoon, NOT 2D illustration, no text, no captions, no watermarks."
        )
        return prompt[:3800]

    def _generate_single_16x9_image(self, prompt: str, dest_path: Path, scene: dict, scene_id: int, shot_idx: int = 1) -> Tuple[Path, str]:
        can_use_ai = self.ai_images_used < self.max_ai_images
        anchors = (0.38, 0.44, 0.48, 0.40)
        anchor = anchors[(shot_idx - 1) % len(anchors)]

        # 1. PRIMARY DEFAULT: Deployed `gpt-image-1-mini` on Microsoft Foundry / Azure OpenAI
        if can_use_ai and self.foundry_endpoint and self.foundry_key:
            saved = self._generate_foundry_image(prompt, dest_path, vertical_anchor=anchor)
            if saved:
                self.gpt_images_used += 1
                self.ai_images_used += 1
                return saved, f"GPT-IMAGE-1-MINI 16:9 ({self.image_model})"

        # 2. SECONDARY FALLBACK: saurav-z/free-image-generation-api Cloudflare Worker
        if can_use_ai and self.free_image_api_url:
            saved = self._generate_via_free_cloudflare_worker(prompt, dest_path, vertical_anchor=anchor)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                return saved, "FREE AI 16:9 (Cloudflare Worker)"

        # 3. Direct Cloudflare Workers AI REST API
        if can_use_ai and self.cf_account_id and self.cf_api_token:
            saved = self._generate_via_cloudflare_rest_api(prompt, dest_path, vertical_anchor=anchor)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                return saved, "FREE AI 16:9 (Cloudflare Workers AI)"

        # 4. Free Pollinations 16:9 Image API
        if can_use_ai and os.getenv("ALLOW_POLLINATIONS_FALLBACK", "true").lower() == "true":
            saved = self._generate_via_pollinations_free(prompt, dest_path, seed=1000 + scene_id * 10 + shot_idx, vertical_anchor=anchor)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                return saved, "FREE AI 16:9 (Flux Widescreen)"

        # 5. Offline CI Fallback: Full-bleed 1920x1080 procedural cinema landscape
        self._create_indian_storybook_illustration(dest_path, scene=scene, topic=self.topic, scene_id=scene_id * 3 + shot_idx)
        self.reframed_cuts_used += 1
        return dest_path, "16:9 CINEMA PLATE"

    def fetch_scene_asset(self, scene: dict, assets_dir: Path) -> Tuple[Path, str]:
        """
        Generates **3 to 4 distinct full-bleed `1920x1080` (16:9) `gpt-image-1-mini` images per scene**
        locked to both the exact spoken Hindi narration beat and the story's `character_bible`!
        """
        assets_dir.mkdir(parents=True, exist_ok=True)
        scene_id = int(scene.get("scene_id", 1))
        narration = str(scene.get("narration", "")).strip()
        subject = str(scene.get("visual_subject") or self.topic).strip()
        dur = float(scene.get("duration_sec", 15.0) or 15.0)

        desired_shots = 4 if dur >= 18.5 else (3 if dur >= 9.0 else 2)

        raw_shot_prompts = list(scene.get("shot_prompts") or [])
        if not raw_shot_prompts:
            raw_shot_prompts = [
                str(scene.get("image_prompt") or f"Wide establishing shot of {subject}"),
                str(scene.get("image_prompt_b") or f"Medium action shot of {subject}"),
                str(scene.get("image_prompt_c") or f"Close-up reaction shot of {subject}"),
            ]
        while len(raw_shot_prompts) < desired_shots:
            raw_shot_prompts.append(
                f"Dramatic alternate cinema angle of {subject}, expressive photorealistic detail, shot {len(raw_shot_prompts) + 1}"
            )

        asset_paths: List[str] = []
        kinds: List[str] = []

        for s_idx in range(1, desired_shots + 1):
            if s_idx > 1 and self.ai_images_used >= self.max_ai_images:
                break
            prompt_str = self._build_gpt_image_scene_prompt(
                raw_shot_prompts[s_idx - 1], scene, shot_idx=s_idx, total_shots=desired_shots
            )
            shot_dest = assets_dir / f"scene_{scene_id:02d}_shot{s_idx}.jpg"
            saved_p, kind_str = self._generate_single_16x9_image(
                prompt_str, shot_dest, scene, scene_id=scene_id, shot_idx=s_idx
            )
            asset_paths.append(str(saved_p))
            kinds.append(kind_str)

        scene["asset_paths"] = asset_paths
        self.scene_audit_log.append({
            "scene_id": scene_id,
            "kind": f"{kinds[0]} ({len(asset_paths)} shots)",
            "shots_count": len(asset_paths),
            "source": ", ".join(Path(p).name for p in asset_paths),
            "narration": narration,
            "source_url": f"Azure OpenAI / Foundry ({self.image_model})",
        })
        return Path(asset_paths[0]), "image"

    def _generate_foundry_image(self, prompt: str, dest_path: Path, vertical_anchor: float = 0.42) -> Optional[Path]:
        """
        Calls your deployed `gpt-image-1-mini` (or `gpt-image-1` / `MAI-Image-2.6-Flash`) endpoint on Azure OpenAI / Microsoft Foundry.
        Requests native widescreen `size="1536x1024"` and `quality="high"` (with automatic payload & route fallbacks)
        and masters the resulting image into a crisp edge-to-edge `1920x1080` (`16:9`) frame.
        """
        try:
            parsed = urlparse(self.foundry_endpoint)
            if parsed.scheme and parsed.netloc:
                resource_root = f"{parsed.scheme}://{parsed.netloc}"
            else:
                resource_root = self.foundry_endpoint.split("/api/projects")[0].split("/openai")[0].split("/models")[0].rstrip("/")

            headers = {
                "Content-Type": "application/json",
                "api-key": self.foundry_key,
                "Authorization": f"Bearer {self.foundry_key}",
            }

            # Candidate URLs for Azure OpenAI / Foundry deployments of gpt-image-1-mini
            candidate_urls = []
            if "/images/generations" in self.foundry_endpoint:
                candidate_urls.append(self.foundry_endpoint)
            candidate_urls.extend([
                f"{resource_root}/openai/deployments/{self.image_model}/images/generations?api-version=2025-04-01-preview",
                f"{resource_root}/openai/deployments/{self.image_model}/images/generations?api-version=2024-12-01-preview",
                f"{resource_root}/openai/v1/images/generations",
                f"{resource_root}/v1/images/generations",
                f"{resource_root}/mai/v1/images/generations",
            ])
            candidate_urls = list(dict.fromkeys(candidate_urls))

            # Candidate payloads for gpt-image-1-mini (1536x1024 landscape widescreen)
            payload_variants = [
                {"model": self.image_model, "prompt": prompt, "n": 1, "size": "1536x1024", "quality": self.image_quality},
                {"prompt": prompt, "n": 1, "size": "1536x1024", "quality": self.image_quality},
                {"model": self.image_model, "prompt": prompt, "n": 1, "size": "1536x1024"},
                {"model": self.image_model, "prompt": prompt, "n": 1, "size": "1024x1024"},
                {"model": self.image_model, "prompt": prompt, "width": 1344, "height": 768},
            ]

            for url in candidate_urls:
                for payload in payload_variants:
                    try:
                        resp = self.session.post(url, headers=headers, json=payload, timeout=75)
                        if resp.status_code != 200:
                            continue
                        data = resp.json()
                        items = data.get("data", [])
                        if items and "b64_json" in items[0]:
                            dest_path.write_bytes(base64.b64decode(items[0]["b64_json"]))
                        elif items and "url" in items[0]:
                            img_r = self.session.get(items[0]["url"], timeout=40)
                            img_r.raise_for_status()
                            dest_path.write_bytes(img_r.content)
                        elif "image" in data and isinstance(data["image"], str):
                            dest_path.write_bytes(base64.b64decode(data["image"]))
                        else:
                            continue

                        with Image.open(dest_path) as im:
                            rgb = im.convert("RGB")
                            if not self._is_valid_photographic_image(rgb):
                                dest_path.unlink(missing_ok=True)
                                continue
                            self._save_as_full_bleed_16x9(rgb, dest_path, vertical_anchor=vertical_anchor)
                        return dest_path
                    except Exception:
                        dest_path.unlink(missing_ok=True)
                        continue
            return None
        except Exception as exc:
            print(f"[MediaFetcher] Foundry {self.image_model} notice: {exc}")
            dest_path.unlink(missing_ok=True)
            return None

    def _generate_via_free_cloudflare_worker(self, prompt: str, dest_path: Path, vertical_anchor: float = 0.38) -> Optional[Path]:
        headers = {"Content-Type": "application/json"}
        if self.free_image_api_key:
            headers["Authorization"] = f"Bearer {self.free_image_api_key}"
            headers["x-api-key"] = self.free_image_api_key

        payload = {
            "prompt": prompt[:1900],
            "model": self.free_image_model,
            "width": 1344,
            "height": 768,
            "num_steps": 4,
        }
        for attempt in (1, 2):
            try:
                resp = self.session.post(self.free_image_api_url, headers=headers, json=payload, timeout=65)
                if resp.status_code != 200:
                    if attempt == 1:
                        time.sleep(0.8)
                        continue
                    return None

                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "application/json" in ctype or resp.content[:1] == b"{":
                    data = resp.json()
                    b64 = data.get("image") or (data.get("result") or {}).get("image")
                    if not b64:
                        return None
                    raw_bytes = base64.b64decode(b64)
                else:
                    raw_bytes = resp.content

                dest_path.write_bytes(raw_bytes)
                with Image.open(dest_path) as im:
                    rgb = im.convert("RGB")
                    if not self._is_valid_photographic_image(rgb):
                        dest_path.unlink(missing_ok=True)
                        return None
                    self._save_as_full_bleed_16x9(rgb, dest_path, vertical_anchor=vertical_anchor)
                return dest_path
            except Exception:
                dest_path.unlink(missing_ok=True)
                if attempt == 1:
                    time.sleep(0.8)
        return None

    def _generate_via_cloudflare_rest_api(self, prompt: str, dest_path: Path, vertical_anchor: float = 0.38) -> Optional[Path]:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.cf_account_id}/ai/run/{self.free_image_model}"
        headers = {
            "Authorization": f"Bearer {self.cf_api_token}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": prompt[:1900], "steps": 4}
        try:
            resp = self.session.post(url, headers=headers, json=payload, timeout=65)
            if resp.status_code != 200:
                return None
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in ctype:
                data = resp.json()
                b64 = (data.get("result") or {}).get("image") or data.get("image")
                if not b64:
                    return None
                dest_path.write_bytes(base64.b64decode(b64))
            else:
                dest_path.write_bytes(resp.content)
            with Image.open(dest_path) as im:
                self._save_as_full_bleed_16x9(im.convert("RGB"), dest_path, vertical_anchor=vertical_anchor)
            return dest_path
        except Exception:
            dest_path.unlink(missing_ok=True)
            return None

    def _generate_via_pollinations_free(self, prompt: str, dest_path: Path, seed: int = 42, vertical_anchor: float = 0.38) -> Optional[Path]:
        try:
            encoded = quote(prompt[:600])
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&seed={seed}&nologo=true&model=flux"
            resp = self.session.get(url, timeout=18)
            if resp.status_code == 200 and len(resp.content) > 15_000:
                dest_path.write_bytes(resp.content)
                with Image.open(dest_path) as im:
                    rgb = im.convert("RGB")
                    if self._is_valid_photographic_image(rgb):
                        self._save_as_full_bleed_16x9(rgb, dest_path, vertical_anchor=vertical_anchor)
                        return dest_path
            dest_path.unlink(missing_ok=True)
            return None
        except Exception:
            dest_path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _create_indian_storybook_illustration(
        dest_path: Path,
        scene: Optional[Dict[str, Any]] = None,
        topic: str = "पंचतंत्र की कहानी",
        scene_id: int = 1,
    ):
        w, h = 1920, 1080
        yy, xx = np.mgrid[0:h, 0:w]
        ty = yy.astype(np.float32) / max(1, h - 1)
        tx = xx.astype(np.float32) / max(1, w - 1)

        palettes = [
            ((28, 38, 64), (165, 88, 52), (235, 165, 75)),
            ((14, 42, 50), (32, 92, 76), (175, 145, 68)),
            ((22, 24, 58), (85, 48, 92), (215, 132, 68)),
            ((16, 32, 58), (44, 84, 118), (210, 160, 72)),
        ]
        top_c, mid_c, bot_c = palettes[(scene_id - 1) % len(palettes)]

        r = np.where(ty < 0.55, top_c[0] + (mid_c[0] - top_c[0]) * (ty / 0.55), mid_c[0] + (bot_c[0] - mid_c[0]) * ((ty - 0.55) / 0.45))
        g = np.where(ty < 0.55, top_c[1] + (mid_c[1] - top_c[1]) * (ty / 0.55), mid_c[1] + (bot_c[1] - mid_c[1]) * ((ty - 0.55) / 0.45))
        b = np.where(ty < 0.55, top_c[2] + (mid_c[2] - top_c[2]) * (ty / 0.55), mid_c[2] + (bot_c[2] - mid_c[2]) * ((ty - 0.55) / 0.45))

        sun_x = 0.72 if scene_id % 2 == 1 else 0.28
        sun_y = 0.34
        dist = np.sqrt((tx - sun_x) ** 2 * 1.8 + (ty - sun_y) ** 2)
        glow = np.exp(-dist * 5.5) * 95.0

        hill1 = 0.62 + 0.04 * np.sin(tx * 7.0 + scene_id) + 0.02 * np.cos(tx * 15.0)
        hill2 = 0.72 + 0.03 * np.cos(tx * 9.0 + scene_id * 2)
        in_hill1 = ty > hill1
        in_hill2 = ty > hill2

        r = np.clip(r + glow, 0, 255)
        g = np.clip(g + glow * 0.78, 0, 255)
        b = np.clip(b + glow * 0.42, 0, 255)

        r = np.where(in_hill1, r * 0.48, r)
        g = np.where(in_hill1, g * 0.55, g)
        b = np.where(in_hill1, b * 0.52, b)

        r = np.where(in_hill2, r * 0.32, r)
        g = np.where(in_hill2, g * 0.38, g)
        b = np.where(in_hill2, b * 0.42, b)

        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img.save(dest_path, quality=95)
