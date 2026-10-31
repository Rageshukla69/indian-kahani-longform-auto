"""
Uploads the rendered 1920x1080 (16:9) Long-Form Hindi Story Video to YouTube via YouTube Data API v3 (`src/youtube_uploader.py`):
- Automatically sets **YouTube's Altered / Synthetic AI Content disclosure** (`status.selfDeclaredAsModifiedContent = True` & `status.containsSyntheticMedia = True`) AND adds explicit AI disclosure tags (`#AIGenerated`, `#AIStory`, `#AIKahani`) in both the YouTube tags list and description!
- Configures Hindi metadata (`defaultLanguage: "hi"`, `defaultAudioLanguage: "hi"`) and ensures every story title is numbered (`कहानी #<N>`).
- Audience setting (`selfDeclaredMadeForKids: False` by default so comments, notifications, and recommendations remain active for family/all-ages viewing by both kids and adults).
"""
import json
import os
from pathlib import Path
from typing import List, Optional

TOKEN_URI = "https://oauth2.googleapis.com/token"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

REQUIRED_AI_TAGS = [
    "AI",
    "AIGenerated",
    "AIStory",
    "AIKahani",
    "AIHindiStory",
    "Panchatantra",
    "PanchatantraKiKahaniyan",
    "HindiKahani",
    "MoralStoriesInHindi",
    "IndianStories",
]


def _load_youtube_credentials():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    raw_json = os.getenv("YOUTUBE_TOKEN_JSON", "").strip()
    if raw_json:
        info = json.loads(raw_json)
        creds = Credentials(
            token=info.get("token") or info.get("access_token"),
            refresh_token=info.get("refresh_token"),
            token_uri=info.get("token_uri", TOKEN_URI),
            client_id=info.get("client_id"),
            client_secret=info.get("client_secret"),
            scopes=info.get("scopes") or SCOPES,
        )
        if creds.refresh_token and creds.client_id and creds.client_secret:
            creds.refresh(Request())
        return creds

    client_id = os.getenv("YOUTUBE_CLIENT_ID", "").strip()
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "").strip()
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN", "").strip()
    if not (client_id and client_secret and refresh_token):
        raise RuntimeError(
            "Missing YOUTUBE_TOKEN_JSON in GitHub Secrets! Set YOUTUBE_TOKEN_JSON or YOUTUBE_CLIENT_ID/SECRET/REFRESH_TOKEN."
        )
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    creds.refresh(Request())
    return creds


def build_youtube_upload_payload(
    title: str,
    description: str,
    tags: List[str],
    story_number: int = 1,
    moral_hindi: str = "",
    attributions: Optional[List[str]] = None,
    sources: Optional[List[dict]] = None,
) -> dict:
    """
    Constructs the YouTube Data API v3 `videos.insert` body with:
    - Story numbering (`कहानी #<story_number>`)
    - AI tags (`#AIGenerated`, `#AIStory`, `AI`, `AIGenerated`)
    - Official YouTube synthetic/altered content disclosure flags (`containsSyntheticMedia: True`, `selfDeclaredAsModifiedContent: True`)
    - Hindi language metadata (`hi`)
    """
    clean_title = title.strip()
    if f"#{story_number}" not in clean_title:
        clean_title = f"कहानी #{story_number}: {clean_title}"

    merged_tags = list(dict.fromkeys((tags or []) + REQUIRED_AI_TAGS))[:30]

    ai_disclosure_block = (
        "\n\n🤖 AI Content Disclosure (कृत्रिम बुद्धिमत्ता सूचना):\n"
        "इस कहानी के चित्र (Visuals) और बहु-पात्र हिंदी आवाज़ें (Multi-Character Hindi Voices) AI (Artificial Intelligence) "
        "की सहायता से शैक्षिक और मनोरंजन उद्देश्य के लिए तैयार किए गए हैं। (Altered/Synthetic Content: Yes)\n"
        "#Panchatantra #HindiKahani #MoralStories #IndianStories #Kahaniyan #AIGenerated #AIStory #AIKahani\n"
    )
    moral_block = f"\n✨ कहानी की सीख (Moral): {moral_hindi}\n" if moral_hindi else ""
    full_description = description.strip() + moral_block + ai_disclosure_block

    if attributions:
        unique_attr = list(dict.fromkeys(attributions))[:10]
        full_description += "\nMedia Credits:\n" + "\n".join(unique_attr)
    if sources:
        source_lines = []
        for item in sources[:6]:
            name = str(item.get("name") or "Literary Source").strip()
            url = str(item.get("url") or "").strip()
            if url.startswith(("https://", "http://")):
                source_lines.append(f"- {name}: {url}")
        if source_lines:
            full_description += "\n\nसंदर्भ स्रोत (Literary Sources):\n" + "\n".join(source_lines)

    return {
        "snippet": {
            "title": clean_title[:100],
            "description": full_description[:4900],
            "tags": merged_tags,
            "categoryId": "24",  # Entertainment / Storytelling (or 27 Education)
            "defaultLanguage": "hi",
            "defaultAudioLanguage": "hi",
        },
        "status": {
            "privacyStatus": os.getenv("YOUTUBE_PRIVACY_STATUS", "public").strip(),
            "selfDeclaredMadeForKids": False,  # Good for both kids & adults while keeping comments/notifications enabled
            "selfDeclaredAsModifiedContent": True,
            "containsSyntheticMedia": True,
        },
    }


def upload_video_to_youtube(
    video_path: Path,
    title: str,
    description: str,
    tags: List[str],
    attributions: List[str],
    sources: Optional[List[dict]] = None,
    story_number: int = 1,
    moral_hindi: str = "",
) -> Optional[str]:
    if os.getenv("DRY_RUN", "false").lower() == "true":
        print("[YouTubeUploader] DRY_RUN is enabled; skipping live YouTube API upload.")
        return "dry_run_id"

    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    creds = _load_youtube_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = build_youtube_upload_payload(
        title=title,
        description=description,
        tags=tags,
        story_number=story_number,
        moral_hindi=moral_hindi,
        attributions=attributions,
        sources=sources,
    )

    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        chunksize=1024 * 1024 * 8,
        resumable=True,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"[YouTubeUploader] Upload progress: {int(status.progress() * 100)}%")

    video_id = response.get("id")
    print(f"[YouTubeUploader] Successfully published Hindi Story #{story_number}! https://www.youtube.com/watch?v={video_id}")
    return video_id


# Backward-compatible alias
upload_short_to_youtube = upload_video_to_youtube
