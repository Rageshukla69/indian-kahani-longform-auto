# 🪔 Indian Hindi Kahani YouTube Automation — 16:9 Long-Form Edition

An automated end-to-end **16:9 Long-Form Indian Storytelling YouTube Pipeline** built in the exact modular architecture of `us-history-shorts-auto`, tailored for **Panchatantra ki 101 Kahaniyan (`कहानी #1` to `कहानी #101`)** and multi-resource Indian folklore (**Hitopadesha, Jataka Tales, Akbar-Birbal, Tenali Rama, Vikram-Betaal, Singhasan Battisi, and Kathasaritsagara**).

## ✨ Key Features
1. **Sequential Story Numbering & Zero Topic Repetition (`src/topic_manager.py`)**:
   - Pre-seeded with all **101 Panchatantra Kahaniyan** (`कहानी #1` through `कहानी #101`) plus classic stories from **Hitopadesha, Jataka Tales, Akbar-Birbal, Tenali Rama, and Vikram-Betaal** (`कहानी #102+`).
   - Automatically replenishes new non-overlapping stories once the queue runs low and blocks any duplicate or reworded topic via normalized Devanagari/English title matching + keyword overlap guards.
2. **Multi-Character Hindi Voiceover (`src/voice_engine.py`)**:
   - Automatically switches voices based on who is speaking in each scene:
     - **Narrator (`narrator`)**: `hi-IN-MadhurNeural` (Warm storyteller cadence)
     - **Female Character (`female_character` / `speaker_gender == "female"`)**: `hi-IN-SwaraNeural` (Expressive female Hindi voice whenever a wife, queen, princess, goddess, or female animal/bird speaks)
     - **Male Protagonist (`male_protagonist`)**: `hi-IN-MadhurNeural` (`+5%` rate, `+5Hz` youthful pitch)
     - **Male Antagonist (`male_antagonist`)**: `hi-IN-MadhurNeural` (`-3%` rate, `-7Hz` deep dramatic pitch)
     - **Wise Elder / Guru (`wise_elder`)**: `hi-IN-MadhurNeural` (`-4%` rate, `-4Hz` calm sage pitch)
3. **Proper Devanagari Storytelling Subtitles (`src/caption_builder.py`)**:
   - Uses **Noto Sans Devanagari** (`Fontsize: 42`, `Spacing: 0` so Devanagari *Shirorekha* and *matras* connect cleanly via HarfBuzz/libass without broken glyphs).
   - Positioned in the lower storytelling bar (`MarginV: 56` on `1920x1080`) with active-word gold highlighting (`#FFD700`) so viewers can watch the 16:9 visuals while reading along comfortably.
4. **Free Scene-Synced 16:9 AI Image Generation (`src/media_fetcher.py` + `cloudflare_worker/worker.js`)**:
   - Integrates **[saurav-z/free-image-generation-api](https://github.com/saurav-z/free-image-generation-api)** via Cloudflare Workers AI (`@cf/black-forest-labs/flux-1-schnell` / SDXL) to generate 16:9 illustrations matched to each scene's narration at $0.00 cost.
   - Includes ready-to-deploy `cloudflare_worker/worker.js` and automatic fallbacks.
5. **YouTube AI Disclosure & Tags (`src/youtube_uploader.py`)**:
   - Automatically sets YouTube's official Altered/Synthetic media flags (`containsSyntheticMedia: True`, `selfDeclaredAsModifiedContent: True`) and adds `#AIGenerated`, `#AIStory`, `#AIKahani` tags.

## 🔑 Required GitHub Secrets

| Secret Name | Purpose |
| :--- | :--- |
| `FREE_IMAGE_API_URL` | Your deployed `saurav-z/free-image-generation-api` Cloudflare Worker URL |
| `FREE_IMAGE_API_KEY` | Your `API_KEY` secret set inside the Cloudflare Worker |
| `FOUNDRY_TEXT_ENDPOINT`, `FOUNDRY_TEXT_API_KEY` | LLM Storyboard generation (optional; has built-in Hindi multi-character fallback) |
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | Optional Azure Speech key (automatically falls back to free `edge-tts` Hindi Neural voices) |
| `YOUTUBE_TOKEN_JSON` | YouTube Data API v3 OAuth JSON for automated uploading |
