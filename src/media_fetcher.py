"""
Scene-Synced Free AI Image Generator & Multi-Archive Visual Resolver (`src/media_fetcher.py`):
1. Primary Engine: **[saurav-z/free-image-generation-api](https://github.com/saurav-z/free-image-generation-api)**
   - Calls your deployed Cloudflare Worker (`FREE_IMAGE_API_URL` + `FREE_IMAGE_API_KEY`) to generate custom **16:9 widescreen (`1344x768`)** AI storybook illustrations for EVERY scene at **$0.00 cost**!
2. Direct Cloudflare Workers AI REST Fallback (`CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN`):
   - Calls `@cf/black-forest-labs/flux-1-schnell` or `@cf/bytedance/stable-diffusion-xl-lightning` directly if you prefer not to deploy a Worker manually.
3. Optional Microsoft Foundry Image Endpoint (`FOUNDRY_ENDPOINT` + `FOUNDRY_API_KEY`) or Free Pollinations FLUX 16:9 Fallback.
4. Multi-Archive Indian Art & Wildlife Fallback (`ArchiveMediaFinder`) + Rich 1920x1080 Indian Storybook Canvas Renderer (with proper Devanagari typography) so a build NEVER fails even in offline CI tests!
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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
import requests

try:
    from archive_media import ArchiveMediaFinder, rank_candidates
except ImportError:
    from src.archive_media import ArchiveMediaFinder, rank_candidates

USER_AGENT = (
    "IndianKahaniLongformBot/2.0 "
    "(https://github.com/saurav-z/free-image-generation-api; StorytellingPipeline/2.0)"
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

        # For Hindi storytelling with free AI image generation, default cap is 20 scenes per video
        raw_max_ai = os.getenv("MAX_AI_IMAGES_PER_VIDEO") or os.getenv("MAX_AI_IMAGES_PER_SHORT")
        if raw_max_ai is None or str(raw_max_ai).strip() == "":
            raw_max_ai = "20"
        try:
            self.max_ai_images = max(0, int(str(raw_max_ai).strip()))
        except (TypeError, ValueError):
            self.max_ai_images = 20

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

    def fetch_scene_asset(self, scene: dict, assets_dir: Path) -> Tuple[Path, str]:
        """
        Scene-Synced 16:9 Visual Pipeline:
        1. Free AI Image Generation API (`saurav-z/free-image-generation-api` Cloudflare Worker or Direct Cloudflare Workers AI)
        2. Optional Foundry AI Image Generation (`FOUNDRY_ENDPOINT`)
        3. Free Pollinations FLUX 16:9 Illustration Generator (when internet is enabled and no keys are set)
        4. Wikimedia Commons / Openverse Indian Art & Wildlife Search
        5. Rich 1920x1080 Indian Storybook Illustrated Scene Plate (with Devanagari scene title & procedural landscape art)
        """
        assets_dir.mkdir(parents=True, exist_ok=True)
        scene_id = int(scene.get("scene_id", 1))
        narration = str(scene.get("narration", "")).strip()
        subject = str(scene.get("visual_subject") or self.topic).strip()
        raw_prompt = str(scene.get("image_prompt") or subject).strip()
        styled_prompt = (
            f"{raw_prompt}, warm cinematic Indian storybook illustration, Amar Chitra Katha and Pixar 3D lighting, "
            "vibrant Indian heritage colors, 16:9 widescreen composition, clean composition, no text, no watermarks"
        )

        can_use_ai = self.ai_images_used < self.max_ai_images
        ai_dest = assets_dir / f"scene_{scene_id:02d}_ai.jpg"

        # 1. Try saurav-z/free-image-generation-api Cloudflare Worker
        if can_use_ai and self.free_image_api_url:
            saved = self._generate_via_free_cloudflare_worker(styled_prompt, ai_dest)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                self.scene_audit_log.append({
                    "scene_id": scene_id,
                    "kind": "FREE AI IMAGE (Cloudflare Worker)",
                    "source": f"saurav-z/free-image-generation-api ({self.free_image_model})",
                    "narration": narration,
                    "source_url": "https://github.com/saurav-z/free-image-generation-api",
                })
                return saved, "image"

        # 2. Try Direct Cloudflare Workers AI REST API (if CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN set)
        if can_use_ai and self.cf_account_id and self.cf_api_token:
            saved = self._generate_via_cloudflare_rest_api(styled_prompt, ai_dest)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                self.scene_audit_log.append({
                    "scene_id": scene_id,
                    "kind": "FREE AI IMAGE (Cloudflare Workers AI)",
                    "source": f"Cloudflare Workers AI ({self.free_image_model})",
                    "narration": narration,
                    "source_url": "https://developers.cloudflare.com/workers-ai/",
                })
                return saved, "image"

        # 3. Try Microsoft Foundry Image Endpoint (if configured)
        if can_use_ai and self.foundry_endpoint and self.foundry_key:
            saved = self._generate_foundry_image(styled_prompt, ai_dest)
            if saved:
                self.ai_images_used += 1
                self.scene_audit_log.append({
                    "scene_id": scene_id,
                    "kind": f"AI IMAGE ({self.image_model})",
                    "source": saved.name,
                    "narration": narration,
                    "source_url": "AI-generated illustration",
                })
                return saved, "image"

        # 4. Try Free Pollinations 16:9 Image API if outbound network is available
        if can_use_ai and os.getenv("ALLOW_POLLINATIONS_FALLBACK", "true").lower() == "true":
            saved = self._generate_via_pollinations_free(styled_prompt, ai_dest, seed=100 + scene_id)
            if saved:
                self.free_worker_images_used += 1
                self.ai_images_used += 1
                self.scene_audit_log.append({
                    "scene_id": scene_id,
                    "kind": "FREE AI IMAGE (Flux 16:9)",
                    "source": saved.name,
                    "narration": narration,
                    "source_url": "AI-generated 16:9 illustration",
                })
                return saved, "image"

        # 5. Render Rich 1920x1080 Indian Storybook Illustrated Scene Plate (100% offline-safe, scene-specific palette & art)
        storybook_path = assets_dir / f"scene_{scene_id:02d}_storybook.jpg"
        self._create_indian_storybook_illustration(storybook_path, scene=scene, topic=self.topic, scene_id=scene_id)
        self.reframed_cuts_used += 1
        self.scene_audit_log.append({
            "scene_id": scene_id,
            "kind": "STORYBOOK ILLUSTRATION PLATE",
            "source": f"16:9 Indian Storybook Canvas (Scene {scene_id})",
            "narration": narration,
            "source_url": "Locally rendered 16:9 storybook artwork",
            "license": "Original generated artwork",
        })
        return storybook_path, "image"

    def _generate_via_free_cloudflare_worker(self, prompt: str, dest_path: Path) -> Optional[Path]:
        """
        Calls the deployed `saurav-z/free-image-generation-api` Cloudflare Worker:
        POST <FREE_IMAGE_API_URL>
        Headers: Authorization: Bearer <FREE_IMAGE_API_KEY>, Content-Type: application/json
        Body: {"prompt": prompt, "model": self.free_image_model, "width": 1344, "height": 768}
        Handles both raw binary image responses (`image/jpeg`, `image/png`) and JSON `{"image": "<base64>"}`.
        """
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
            resp = self.session.post(self.free_image_api_url, headers=headers, json=payload, timeout=60)
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
                rgb.save(dest_path, quality=95)
            return dest_path
        except Exception as exc:
            print(f"[MediaFetcher] Free Image Worker notice: {exc}")
            dest_path.unlink(missing_ok=True)
            return None

    def _generate_via_cloudflare_rest_api(self, prompt: str, dest_path: Path) -> Optional[Path]:
        """Direct Cloudflare Workers AI REST endpoint (`@cf/black-forest-labs/flux-1-schnell`)."""
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.cf_account_id}/ai/run/{self.free_image_model}"
        headers = {
            "Authorization": f"Bearer {self.cf_api_token}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": prompt[:1900], "num_steps": 4}
        try:
            resp = self.session.post(url, headers=headers, json=payload, timeout=60)
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
                im.convert("RGB").save(dest_path, quality=95)
            return dest_path
        except Exception as exc:
            print(f"[MediaFetcher] Cloudflare REST AI notice: {exc}")
            dest_path.unlink(missing_ok=True)
            return None

    def _generate_foundry_image(self, prompt: str, dest_path: Path) -> Optional[Path]:
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
                        im.convert("RGB").save(dest_path, quality=95)
                    return dest_path
            return None
        except Exception:
            return None

    def _generate_via_pollinations_free(self, prompt: str, dest_path: Path, seed: int = 42) -> Optional[Path]:
        try:
            encoded = quote(prompt[:600])
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&seed={seed}&nologo=true&model=flux"
            resp = self.session.get(url, timeout=18)
            if resp.status_code == 200 and len(resp.content) > 15_000:
                dest_path.write_bytes(resp.content)
                with Image.open(dest_path) as im:
                    rgb = im.convert("RGB")
                    if self._is_valid_photographic_image(rgb):
                        rgb.save(dest_path, quality=95)
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
        """
        Renders a rich 1920x1080 (16:9) Indian Storybook Landscape Painting with:
        - Golden-hour / twilight sky gradient, glowing sun/moon mandala, layered forested hills & river reflections
        - Ornate Indian temple/archway border and proper Devanagari typography (`Noto Sans Devanagari`)
        """
        w, h = 1920, 1080
        scene = scene or {}
        yy, xx = np.mgrid[0:h, 0:w]
        ty = yy.astype(np.float32) / max(1, h - 1)
        tx = xx.astype(np.float32) / max(1, w - 1)

        palettes = [
            # Golden Sunrise over Narmada River
            ((42, 22, 58), (185, 82, 48), (242, 168, 62)),
            # Lush Emerald Champakavan Forest
            ((16, 44, 52), (34, 96, 78), (188, 152, 68)),
            # Royal Indraprastha Evening
            ((24, 20, 56), (92, 44, 98), (225, 138, 72)),
            # Sacred Banyan Twilight
            ((18, 34, 62), (48, 88, 122), (218, 165, 75)),
        ]
        top_c, mid_c, bot_c = palettes[(scene_id - 1) % len(palettes)]

        r = np.where(ty < 0.55, top_c[0] + (mid_c[0] - top_c[0]) * (ty / 0.55), mid_c[0] + (bot_c[0] - mid_c[0]) * ((ty - 0.55) / 0.45))
        g = np.where(ty < 0.55, top_c[1] + (mid_c[1] - top_c[1]) * (ty / 0.55), mid_c[1] + (bot_c[1] - mid_c[1]) * ((ty - 0.55) / 0.45))
        b = np.where(ty < 0.55, top_c[2] + (mid_c[2] - top_c[2]) * (ty / 0.55), mid_c[2] + (bot_c[2] - mid_c[2]) * ((ty - 0.55) / 0.45))

        # Sun/Moon glow
        sun_x = 0.72 if scene_id % 2 == 1 else 0.28
        sun_y = 0.34
        dist = np.sqrt((tx - sun_x) ** 2 * 1.8 + (ty - sun_y) ** 2)
        glow = np.exp(-dist * 5.5) * 95.0

        # Layered distant hills and river silhouette
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
        d = ImageDraw.Draw(img)

        gold = (245, 198, 92)
        cream = (252, 246, 232)

        def get_font(size: int, bold: bool = True):
            candidates = [
                "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
            for fp in candidates:
                if Path(fp).exists():
                    try:
                        return ImageFont.truetype(fp, size)
                    except Exception:
                        pass
            return ImageFont.load_default()

        # Ornate frame & mandala rings
        d.rectangle((42, 36, w - 42, h - 36), outline=gold, width=3)
        d.rectangle((54, 48, w - 54, h - 48), outline=(180, 140, 65), width=1)

        cx, cy = int(w * sun_x), int(h * sun_y)
        for rad in (140, 105, 72):
            d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline=gold, width=2)

        callout = str(scene.get("on_screen_callout") or "भारतीय लोककथा").strip()
        speaker = str(scene.get("speaker_name") or "कथावाचक").strip()
        d.text((96, 82), f"✦ {callout}  |  {speaker}", font=get_font(34, True), fill=gold)

        subject = str(scene.get("visual_subject") or topic).strip()
        d.text((96, 145), subject[:78], font=get_font(28, False), fill=cream)

        img.save(dest_path, quality=95)
