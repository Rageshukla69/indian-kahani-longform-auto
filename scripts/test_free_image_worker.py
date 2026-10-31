"""Diagnostic script to verify your deployed saurav-z/free-image-generation-api Cloudflare Worker."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from media_fetcher import MediaFetcher


def main():
    load_dotenv()
    fetcher = MediaFetcher(topic="कहानी #1: बंदर और मगरमच्छ")
    out = Path("temp/test_worker_16x9.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = {
        "scene_id": 1,
        "narration": "नर्मदा नदी के किनारे जामुन के विशाल पेड़ पर रक्तमुख नाम का बंदर रहता था।",
        "visual_subject": "Cheerful monkey on a Jamun berry tree by an Indian river at sunrise",
        "image_prompt": "Warm 16:9 Indian storybook illustration of a monkey on a Jamun tree by a river at sunrise",
        "on_screen_callout": "कहानी #1 • पंचतंत्र",
    }
    path, kind = fetcher.fetch_scene_asset(scene, out.parent)
    print(f"[Diagnostic] Generated scene asset -> {path} (type={kind}, audit={fetcher.scene_audit_log[-1]})")


if __name__ == "__main__":
    main()
