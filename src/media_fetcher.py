"""
Scene-Synced Multi-Shot Photorealistic 16:9 Free AI Image Generator (`src/media_fetcher.py` — v2.0 Full-Bleed 16:9 Edition):
1. Generates **2 Distinct Photorealistic 16:9 Images per Scene** (`Shot A: Wide Establishing` + `Shot B: Close-Up/Action`)
   for scenes >= 7.0 seconds — producing **30 to 50 free AI images per story** (since Cloudflare Workers AI gives ~173 free
   FLUX images/day and you upload 2 stories/day)!
2. Strictly enforces **Photorealistic 8K Live-Action Cinema** prompts (stripping any cartoon/2D illustration terms).
3. **Full-Bleed 16:9 Widescreen Conversion (`_save_as_full_bleed_16x9`)**:
   - Because `@cf/black-forest-labs/flux-1-schnell` natively returns `1024x1024` (1:1) images, saving them raw caused pillarboxed
     sidebars! Now `_save_as_full_bleed_16x9` automatically scales and smart-crops every generated AI image into a true edge-to-edge
     `1920x1080` (`16:9`) widescreen cinema frame (`aspect = 1.7778`) with **ZERO side bars**!
"""
import base64
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote, urlparse
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import requests

try:
    from archive_media import ArchiveMediaFinder, rank_candidates
except ImportError:
    from src.archive_media import ArchiveMediaFinder, rank_candidates

USER_AGENT = (
    "IndianKahaniLongformBot/2.0 "
    "(https://github.com/saurav-z/free-image-generation-api; PhotorealisticPipeline/2.0)"
)

PHOTOREALISTIC_ENFORCER = (
    ", ultra-realistic 8k live-action cinema photography, shot on ARRI Alexa 65 with 35mm anamorphic lens, "
    "photorealistic natural textures, dramatic golden-hour volumetric lighting, wide 16:9 horizontal composition, "
    "National Geographic and epic Indian historical film realism, NOT cartoon, NOT 2D illustration, NOT comic, no text, no watermarks"
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
    ):
        self.topic = topic
        self.era_keywords = era_keywords or []
        self.wikipedia_articles = (wikipedia_articles or ["Panchatantra", "Hitopadesha"])[:5]
        self.video_pool_queries = video_pool_queries or []
        self.llm_client = llm_client

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.media_cache_dir = Path(os.getenv("MEDIA_CACHE_DIR", ".cache/media"))
        self.media_cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Primary: saurav-z/free-image-generation-api Cloudflare Worker URL & Key
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

        # 2. Secondary: Direct Cloudflare Account ID & Token
        self.cf_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.cf_api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

        # 3. Optional Foundry endpoint
        self.foundry_endpoint = (
            os.getenv("FOUNDRY_ENDPOINT")
            or os.getenv("FOUNDRY_TEXT_ENDPOINT")
            or ""
        ).strip().rstrip("/")
        self.foundry_key = (
            os.getenv("FOUNDRY_API_KEY")
            or os.getenv("FOUNDRY_TEXT_API_KEY")
            or ""
        ).strip()
        self.image_model = (
            os.getenv("FOUNDRY_IMAGE_MODEL") or "MAI-Image-2.6-Flash"
        ).strip()

        # Default cap is 60 AI images per video (2 stories/day x ~40 images = 80 images/day, well within 173/day free quota!)
        raw_max_ai = os.getenv("MAX_AI_IMAGES_PER_VIDEO") or os.getenv("MAX_AI_IMAGES_PER_SHORT")
        if raw_max_ai is None or str(raw_max_ai).strip() == "":
            raw_max_ai = "60"
        try:
            self.max_ai_images = max(0, int(str(raw_max_ai).strip()))
        except (TypeError, ValueError):
            self.max_ai_images = 60

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
    def _save_as_full_bleed_16x9(img: Image.Image, dest_path: Path, vertical_anchor: float = 0.38) -> Path:
        """
        Converts ANY generated image (including 1024x1024 square outputs from Flux-1-Schnell)
        into a true full-bleed `1920x1080` (16:9) widescreen frame with ZERO side pillarboxes!
        Uses `vertical_anchor` (0.35 for Shot A, 0.48 for Shot B) to preserve subjects cleanly.
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
        cropped.save(dest_path, quality=95)
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

    @staticmethod
    def _clean_photorealistic_prompt(raw_prompt: str, subject: str) -> str:
        p = (raw_prompt or subject).strip()
        p = re.sub(
            r"\b(cartoon|pixar|3d\s*animation|animated|storybook\s*illustration|illustration|amar\s*chitra\s*katha|comic|2d|drawing|sketch)\b",
            "photorealistic live-action cinema",
            p,
            flags=re.IGNORECASE,
        )
        return f"{p}{PHOTOREALISTIC_ENFORCER}"

    def _generate_single_16x9_image(self, prompt: str, dest_path: Path, scene: dict, scene_id: int, shot_idx: int = 1) -> Tuple[Path, str]:
        can_use_ai = self.ai_images_used < self.max_ai_images
        anchor = 0.34 if shot_idx == 1 else 0.46

        # 1. Try saurav-z/free-image-generation-api Cloudflare Worker
        if can_use_ai and self.free_image_api_url:
            saved = self._generate_via_free_cloudflare_worker(prompt, dest_path, vertical_anchor=anchor)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                return saved, "FREE AI 16:9 (Cloudflare Worker)"

        # 2. Try Direct Cloudflare Workers AI REST API
        if can_use_ai and self.cf_account_id and self.cf_api_token:
            saved = self._generate_via_cloudflare_rest_api(prompt, dest_path, vertical_anchor=anchor)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                return saved, "FREE AI 16:9 (Cloudflare Workers AI)"

        # 3. Try Microsoft Foundry Image Endpoint
        if can_use_ai and self.foundry_endpoint and self.foundry_key:
            saved = self._generate_foundry_image(prompt, dest_path, vertical_anchor=anchor)
            if saved:
                self.ai_images_used += 1
                return saved, f"AI 16:9 ({self.image_model})"

        # 4. Try Free Pollinations 16:9 Image API
        if can_use_ai and os.getenv("ALLOW_POLLINATIONS_FALLBACK", "true").lower() == "true":
            saved = self._generate_via_pollinations_free(prompt, dest_path, seed=1000 + scene_id * 10 + shot_idx, vertical_anchor=anchor)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                return saved, "FREE AI 16:9 (Flux Widescreen)"

        # 5. Offline CI Fallback: Full-bleed 1920x1080 procedural cinema landscape
        self._create_indian_storybook_illustration(dest_path, scene=scene, topic=self.topic, scene_id=scene_id + shot_idx)
        self.reframed_cuts_used += 1
        return dest_path, "16:9 CINEMA PLATE"

    def fetch_scene_asset(self, scene: dict, assets_dir: Path) -> Tuple[Path, str]:
        """
        Generates 1 or 2 full-bleed `1920x1080` (16:9) photorealistic images for `scene`:
        - Shot A (`scene_XX_shot1.jpg`): Wide Establishing 16:9 Photorealistic Shot
        - Shot B (`scene_XX_shot2.jpg`): Mid-Scene Action/Close-Up 16:9 Photorealistic Shot (generated whenever `duration_sec >= 6.5` or `image_prompt_b` is present and AI budget remains)
        Stores `scene["asset_paths"] = [str(shot1), str(shot2)]` so `VideoComposer` cuts to a fresh image every ~6 seconds!
        """
        assets_dir.mkdir(parents=True, exist_ok=True)
        scene_id = int(scene.get("scene_id", 1))
        narration = str(scene.get("narration", "")).strip()
        subject = str(scene.get("visual_subject") or self.topic).strip()
        dur = float(scene.get("duration_sec", 9.0) or 9.0)

        p1 = self._clean_photorealistic_prompt(str(scene.get("image_prompt") or subject), subject)
        shot1_dest = assets_dir / f"scene_{scene_id:02d}_shot1.jpg"
        saved1, kind1 = self._generate_single_16x9_image(p1, shot1_dest, scene, scene_id, shot_idx=1)

        asset_paths = [str(saved1)]
        kinds = [kind1]

        # Generate a second distinct photorealistic 16:9 image for this scene if duration >= 6.0s
        raw_p2 = str(scene.get("image_prompt_b") or "").strip()
        if (dur >= 6.0 or raw_p2) and self.ai_images_used < self.max_ai_images:
            p2 = self._clean_photorealistic_prompt(
                raw_p2 or f"Close-up dramatic camera angle of {subject}, expressive photorealistic detail",
                subject,
            )
            shot2_dest = assets_dir / f"scene_{scene_id:02d}_shot2.jpg"
            saved2, kind2 = self._generate_single_16x9_image(p2, shot2_dest, scene, scene_id, shot_idx=2)
            asset_paths.append(str(saved2))
            kinds.append(kind2)

        scene["asset_paths"] = asset_paths
        self.scene_audit_log.append({
            "scene_id": scene_id,
            "kind": f"{kinds[0]} ({len(asset_paths)} shots)",
            "shots_count": len(asset_paths),
            "source": ", ".join(Path(p).name for p in asset_paths),
            "narration": narration,
            "source_url": "https://github.com/saurav-z/free-image-generation-api",
        })
        return saved1, "image"

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
        try:
            resp = self.session.post(self.free_image_api_url, headers=headers, json=payload, timeout=65)
            if resp.status_code != 200:
                print(f"[MediaFetcher] Cloudflare Worker HTTP {resp.status_code}: {resp.text[:200]}")
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
        except Exception as exc:
            print(f"[MediaFetcher] Free Image Worker notice: {exc}")
            dest_path.unlink(missing_ok=True)
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
        except Exception as exc:
            print(f"[MediaFetcher] Cloudflare REST AI notice: {exc}")
            dest_path.unlink(missing_ok=True)
            return None

    def _generate_foundry_image(self, prompt: str, dest_path: Path, vertical_anchor: float = 0.38) -> Optional[Path]:
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
            route_attempts = [
                (
                    f"{resource_root}/mai/v1/images/generations",
                    {"model": self.image_model, "prompt": prompt, "width": 1344, "height": 768},
                ),
                (
                    f"{resource_root}/openai/v1/images/generations",
                    {"model": self.image_model, "prompt": prompt, "n": 1, "size": "1792x1024"},
                ),
            ]
            for url, payload in route_attempts:
                resp = self.session.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", [])
                    if items and "b64_json" in items[0]:
                        dest_path.write_bytes(base64.b64decode(items[0]["b64_json"]))
                    elif items and "url" in items[0]:
                        img_r = self.session.get(items[0]["url"], timeout=35)
                        img_r.raise_for_status()
                        dest_path.write_bytes(img_r.content)
                    else:
                        continue
                    with Image.open(dest_path) as im:
                        self._save_as_full_bleed_16x9(im.convert("RGB"), dest_path, vertical_anchor=vertical_anchor)
                    return dest_path
            return None
        except Exception:
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
        """Renders a full-bleed 1920x1080 (16:9) atmospheric cinema landscape for offline test runs."""
        w, h = 1920, 1080
        scene = scene or {}
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
