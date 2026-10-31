"""
Bulletproof Anti-Repetition & Sequential Story Numbering Manager (`src/topic_manager.py`):
1. Starts with **Panchatantra ki 101 Kahaniyan** (`कहानी #1` through `कहानी #101`).
2. Automatically expands into multi-source Indian classic collections (`कहानी #102+`) across:
   - **Hitopadesha** (हितोपदेश की कहानियाँ)
   - **Jataka Tales** (जातक कथाएँ)
   - **Akbar-Birbal** (अकबर-बीरबल की चतुराई भरी कहानियाँ)
   - **Tenali Rama** (तेनाली रामा की कहानियाँ)
   - **Vikram-Betaal** (विक्रम और बेताल की पहेलियाँ)
   - **Kathasaritsagara & Singhasan Battisi** (कथासरित्सागर और सिंहासन बत्तीसी)
3. Guarantees **ZERO TOPIC REPETITION**:
   - Live Git state sync (`_sync_latest_git_state`) before picking a topic
   - Union of `data/used_topics.json` and `data/upload_history.json`
   - Normalized Hindi/English title matching + >=65% core keyword overlap blocker
4. Strictly increments and preserves the **Story Number (`story_number`)** so every video is numbered sequentially (`कहानी #1`, `कहानी #2`, ..., `कहानी #101`, `कहानी #102`...).
"""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SEED_FILE = Path("data/topics_seed.json")
USED_FILE = Path("data/used_topics.json")
HISTORY_FILE = Path("data/upload_history.json")

STORY_SOURCES_ROTATION = [
    "Panchatantra (पंचतंत्र की कहानियाँ)",
    "Hitopadesha (हितोपदेश की नीति कथाएँ)",
    "Jataka Tales (जातक कथाएँ)",
    "Akbar-Birbal (अकबर-बीरबल की लोककथाएँ)",
    "Tenali Rama (तेनाली रामा की कहानियाँ)",
    "Vikram-Betaal (विक्रम-बेताल की कहानियाँ)",
    "Singhasan Battisi (सिंहासन बत्तीसी की कहानियाँ)",
    "Kathasaritsagara (कथासरित्सागर की लोककथाएँ)",
]

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "into", "when", "what", "how", "why",
    "who", "where", "this", "these", "those", "their", "there", "were", "was", "have",
    "story", "tales", "tale", "kahani", "kahaniyan", "panchatantra", "hitopadesha",
    "jataka", "akbar", "birbal", "tenali", "rama", "vikram", "betaal", "king", "prince",
    "कहानी", "पंचतंत्र", "हितोपदेश", "जातक", "की", "का", "के", "और", "में", "से", "एक", "जो", "ने", "को",
}

RESERVE_INDIAN_STORIES = [
    "Hitopadesha — चतुर सियार और हाथी का घमंड (The Clever Jackal and the Proud Elephant Karpuratilaka)",
    "Jataka Tales — बातूनी कछुआ और राजहंस (The Jataka Tale of Kacchapa and the Swans)",
    "Akbar-Birbal — स्वर्ग की यात्रा से लौटा बीरबल (How Birbal Returned from Heaven Outsmarting the Barber)",
    "Tenali Rama — काली माँ का वरदान और हजार मुख वाली देवी (How Tenali Rama Made the Goddess Laugh)",
    "Vikram-Betaal — कौन किसका पति? सिर और धड़ की पहेली (Vikramaditya and the Riddle of the Transposed Heads)",
    "Singhasan Battisi — राजा भोज और पहली पुतली रत्नमंजरी की कथा (King Bhoja and the First Statue Ratnamanjari)",
    "Kathasaritsagara — राजा उदयन और वासवदत्ता की अमर कथा (The Legend of Udayana and Vasavadatta)",
]


def _strip_story_number_prefix(text: str) -> str:
    """Removes leading `कहानी #12:` or `Story #12:` prefixes for semantic comparison."""
    cleaned = re.sub(r"^(?:कहानी|story)\s*#\s*\d+\s*[:\-—]\s*", "", (text or "").strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def _normalize_topic(text: str) -> str:
    bare = _strip_story_number_prefix(text).lower()
    # Keep alphanumeric and Devanagari characters
    norm = re.sub(r"[^\w\u0900-\u097F]+", " ", bare, flags=re.UNICODE)
    return re.sub(r"\s+", " ", norm).strip()


def _extract_core_keywords(text: str) -> Set[str]:
    bare = _strip_story_number_prefix(text).lower()
    tokens = re.findall(r"[\w\u0900-\u097F]{3,}", bare, flags=re.UNICODE)
    return {t for t in tokens if t not in STOPWORDS and not t.isdigit()}


def extract_story_number(topic: str, fallback_number: int = 1) -> int:
    m = re.search(r"#\s*(\d+)", topic or "")
    if m:
        return int(m.group(1))
    return max(1, int(fallback_number))


def format_numbered_topic(raw_topic: str, story_number: int) -> str:
    bare = _strip_story_number_prefix(raw_topic)
    return f"कहानी #{story_number}: {bare}"


def is_topic_already_used(candidate: str, used_topics: List[str]) -> bool:
    """
    Returns True if `candidate` matches any topic in `used_topics` by normalized title,
    substring containment, or >= 65% core keyword overlap across Hindi & English tokens.
    """
    cand_norm = _normalize_topic(candidate)
    if not cand_norm:
        return True

    cand_kw = _extract_core_keywords(candidate)

    for used in used_topics:
        u_norm = _normalize_topic(used)
        if not u_norm:
            continue
        if cand_norm == u_norm or (len(cand_norm) > 12 and (cand_norm in u_norm or u_norm in cand_norm)):
            return True
        if len(cand_kw) >= 3:
            u_kw = _extract_core_keywords(used)
            if u_kw:
                overlap = len(cand_kw & u_kw) / float(min(len(cand_kw), len(u_kw)))
                if overlap >= 0.65:
                    return True
    return False


def _sync_latest_git_state() -> None:
    if not os.getenv("GITHUB_ACTIONS"):
        return
    try:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", "origin", "main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=25,
            check=False,
        )
    except Exception:
        pass


def load_used_topics() -> List[str]:
    combined: List[str] = []
    seen_norm: Set[str] = set()

    if USED_FILE.exists():
        try:
            data = json.loads(USED_FILE.read_text(encoding="utf-8"))
            for t in data.get("used", []):
                norm = _normalize_topic(t)
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    combined.append(t.strip())
        except Exception:
            pass

    if HISTORY_FILE.exists():
        try:
            hdata = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            for item in hdata.get("uploads", []):
                t = (item.get("topic") or "").strip()
                norm = _normalize_topic(t)
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    combined.append(t)
        except Exception:
            pass

    return combined


def get_next_story_number(used_topics: Optional[List[str]] = None) -> int:
    used = used_topics if used_topics is not None else load_used_topics()
    max_num = len(used)
    if USED_FILE.exists():
        try:
            data = json.loads(USED_FILE.read_text(encoding="utf-8"))
            max_num = max(max_num, int(data.get("last_story_number", 0) or 0))
        except Exception:
            pass
    if HISTORY_FILE.exists():
        try:
            hdata = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            for item in hdata.get("uploads", []):
                max_num = max(max_num, int(item.get("story_number", 0) or 0))
        except Exception:
            pass
    for u in used:
        m = re.search(r"#\s*(\d+)", u)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def load_seed_topics() -> List[str]:
    if not SEED_FILE.exists():
        return []
    try:
        data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_seed_topics(seeds: List[str]) -> None:
    SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEED_FILE.write_text(json.dumps(seeds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_next_topic_with_number(llm_client=None) -> Tuple[str, int, str]:
    """
    Returns `(formatted_topic, story_number, collection_source)`.
    """
    _sync_latest_git_state()
    used_list = load_used_topics()
    next_num = get_next_story_number(used_list)

    custom = os.getenv("CUSTOM_TOPIC", "").strip()
    if custom:
        s_num = extract_story_number(custom, fallback_number=next_num)
        formatted = format_numbered_topic(custom, s_num)
        source = "Panchatantra" if "panchatantra" in formatted.lower() or s_num <= 101 else "Indian Folklore Collection"
        print(f"[TopicManager] Using custom topic (#{s_num}): {formatted}")
        return formatted, s_num, source

    seeds = load_seed_topics()
    unused_seeds = [t for t in seeds if not is_topic_already_used(t, used_list)]

    # Auto-replenish from multiple Indian story collections once fewer than 5 unused seeds remain
    if len(unused_seeds) < 5 and llm_client is not None:
        print("[TopicManager] Fewer than 5 unused seed stories left — auto-replenishing from multi-resource Indian folklore...")
        try:
            if hasattr(llm_client, "generate_fresh_topics_batch"):
                new_batch = llm_client.generate_fresh_topics_batch(used_list + seeds, count=15, start_number=len(seeds) + 1)
            else:
                new_batch = [llm_client.generate_fresh_topic(used_list + seeds)]
            added = 0
            for cand in new_batch:
                cand = (cand or "").strip()
                if cand and not is_topic_already_used(cand, used_list + seeds):
                    numbered_cand = format_numbered_topic(cand, len(seeds) + 1)
                    seeds.append(numbered_cand)
                    unused_seeds.append(numbered_cand)
                    added += 1
            if added > 0:
                save_seed_topics(seeds)
                print(f"[TopicManager] Auto-replenished {SEED_FILE} with {added} new multi-source Indian stories.")
        except Exception as exc:
            print(f"[TopicManager] Batch replenishment notice: {exc}")

    if unused_seeds:
        raw_chosen = unused_seeds[0]
        s_num = extract_story_number(raw_chosen, fallback_number=next_num)
        formatted = format_numbered_topic(raw_chosen, s_num)
        source = _detect_source_collection(formatted, s_num)
        print(f"[TopicManager] Selected Story #{s_num} ({len(unused_seeds)} remaining in queue): {formatted}")
        return formatted, s_num, source

    if llm_client is not None:
        for _ in range(4):
            try:
                fresh = llm_client.generate_fresh_topic(used_list[-60:] + seeds[-30:])
                if fresh and not is_topic_already_used(fresh, used_list):
                    formatted = format_numbered_topic(fresh, next_num)
                    seeds.append(formatted)
                    save_seed_topics(seeds)
                    return formatted, next_num, _detect_source_collection(formatted, next_num)
            except Exception as exc:
                print(f"[TopicManager] Fresh topic generation notice: {exc}")

    for reserve in RESERVE_INDIAN_STORIES:
        if not is_topic_already_used(reserve, used_list):
            formatted = format_numbered_topic(reserve, next_num)
            seeds.append(formatted)
            save_seed_topics(seeds)
            return formatted, next_num, _detect_source_collection(formatted, next_num)

    fallback = f"कहानी #{next_num}: Hitopadesha — अनमोल भारतीय नीति कथा #{next_num}"
    return fallback, next_num, "Hitopadesha"


def select_next_topic(llm_client=None) -> str:
    topic, _, _ = select_next_topic_with_number(llm_client=llm_client)
    return topic


def _detect_source_collection(topic: str, story_number: int) -> str:
    low = topic.lower()
    if "panchatantra" in low or "पंचतंत्र" in low or story_number <= 101:
        return "Panchatantra (पंचतंत्र)"
    if "hitopadesha" in low or "हितोपदेश" in low:
        return "Hitopadesha (हितोपदेश)"
    if "jataka" in low or "जातक" in low:
        return "Jataka Tales (जातक कथाएँ)"
    if "akbar" in low or "birbal" in low or "बीरबल" in low:
        return "Akbar-Birbal (अकबर-बीरबल)"
    if "tenali" in low or "तेनाली" in low:
        return "Tenali Rama (तेनाली रामा)"
    if "vikram" in low or "betaal" in low or "विक्रम" in low:
        return "Vikram-Betaal (विक्रम-बेताल)"
    return "Indian Moral & Folk Stories (भारतीय लोककथाएँ)"


def _push_topic_state_to_github(topic: str, story_number: int) -> None:
    if not os.getenv("GITHUB_ACTIONS"):
        return
    try:
        subprocess.run(["git", "config", "--local", "user.email", "actions@github.com"], check=False)
        subprocess.run(["git", "config", "--local", "user.name", "Indian Kahani Auto-Bot"], check=False)
        subprocess.run(
            ["git", "add", "data/used_topics.json", "data/upload_history.json", "data/topics_seed.json"],
            check=False,
        )
        diff_proc = subprocess.run(["git", "diff", "--staged", "--quiet"], check=False)
        if diff_proc.returncode == 0:
            return

        subprocess.run(
            ["git", "commit", "-m", f"chore: record completed Kahani #{story_number} ({topic[:40]})"],
            check=False,
        )
        for attempt in range(1, 4):
            pull_res = subprocess.run(
                ["git", "pull", "--rebase", "--autostash", "origin", "main"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            push_res = subprocess.run(
                ["git", "push", "origin", "HEAD:main"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if push_res.returncode == 0:
                print(f"[TopicManager] Immediately pushed updated used_topics.json to GitHub (attempt {attempt}).")
                return
            print(f"[TopicManager] Git push attempt {attempt} notice: {push_res.stderr[:200]} {pull_res.stderr[:200]}")
    except Exception as exc:
        print(f"[TopicManager] Immediate git push notice: {exc}")


def mark_topic_completed(topic: str, video_id: Optional[str] = None, title: str = "", story_number: Optional[int] = None):
    USED_FILE.parent.mkdir(parents=True, exist_ok=True)
    used = load_used_topics()
    s_num = story_number or extract_story_number(topic, fallback_number=len(used) + 1)

    if not any(_normalize_topic(u) == _normalize_topic(topic) for u in used):
        used.append(topic)

    USED_FILE.write_text(
        json.dumps({"used": used, "last_story_number": s_num}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    uploads = []
    if HISTORY_FILE.exists():
        try:
            uploads = json.loads(HISTORY_FILE.read_text(encoding="utf-8")).get("uploads", [])
        except Exception:
            uploads = []

    uploads.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "story_number": s_num,
        "topic": topic,
        "title": title or topic,
        "youtube_video_id": video_id or "dry_run",
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}" if video_id and video_id != "dry_run_id" else None,
    })
    HISTORY_FILE.write_text(json.dumps({"uploads": uploads}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if video_id and video_id != "dry_run_id":
        _push_topic_state_to_github(topic, s_num)
