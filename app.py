"""
AI Choose Now! — an interactive choose-your-own-adventure game using local Ollama models.
- Text generation: gemma3:4b (configurable at runtime)
- Image generation: x/flux2-klein:latest
"""

import json
import os
import time
import re
import base64
import uuid
import threading
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
TEXT_MODEL = "gemma3:4b"  # default; can be changed at runtime via /api/set_model
IMAGE_MODEL = "x/flux2-klein:latest"
MAX_CONTEXT_CHARS = 10000  # rough char budget before we summarise
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400

# Mutable runtime config
runtime_config = {
    "text_model": TEXT_MODEL,
}

# ---------------------------------------------------------------------------
# In-memory adventure state (single-player, single-session)
# ---------------------------------------------------------------------------
adventure_state = {
    "entries": [],      # working context list (may be trimmed by summarisation)
    "full_history": [],  # complete history, never trimmed — used for download/history
    "summary": None,     # compressed summary when context grows too long
    "started": False,
    "scenario_essentials": "",  # user-provided scenario context included in every prompt
    "last_prompt": "",   # store last prompt sent to the LLM for debugging
}

# Pending image generation jobs: {job_id: {"status": "pending"|"done"|"error", "image": str, "error": str}}
image_jobs = {}


def reset_state():
    adventure_state["entries"] = []
    adventure_state["full_history"] = []
    adventure_state["summary"] = None
    adventure_state["started"] = False
    adventure_state["scenario_essentials"] = ""
    adventure_state["last_prompt"] = ""
    image_jobs.clear()


def add_entry(entry: dict):
    """Append an entry to both the working context and the full history."""
    adventure_state["entries"].append(entry)
    adventure_state["full_history"].append(entry)


def pop_entry():
    """Remove the last entry from both working context and full history. Returns the removed entry."""
    removed = None
    if adventure_state["entries"]:
        removed = adventure_state["entries"].pop()
    if adventure_state["full_history"]:
        adventure_state["full_history"].pop()
    return removed


def format_choice_as_narrative(action: str) -> str:
    """Format a player choice as 'You <action>' with proper casing."""
    action = action.strip()
    if not action:
        return "You do nothing."
    # Lowercase the first character so it reads naturally after 'You '
    first = action[0].lower()
    return f"You {first}{action[1:]}"


def strip_choice_lines(text: str) -> str:
    """Remove numbered choice lines and leaked context markers from narrative text."""
    lines = text.split('\n')
    filtered = [
        line for line in lines
        if not re.match(r'^\s*>?\s*\d\.\s+', line)
        and not re.match(r'^\s*\[Player\s', line, re.IGNORECASE)
        and not re.match(r'^\s*\[Story\s', line, re.IGNORECASE)
        and not re.match(r'^\s*\[Scenario\s', line, re.IGNORECASE)
        and not re.match(r'^\s*>\s*You\s', line)
    ]
    # Also strip trailing blank lines left behind
    result = '\n'.join(filtered).rstrip()
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_context_prompt(user_action: str = "", is_start: bool = False, has_starting_text: bool = False) -> str:
    """Build the full prompt to send to the text model, including history."""
    system = (
        "You are a masterful narrator of an interactive text adventure game. "
        "Always write in second-person present tense (\"You\"). "
        "Your prose must be vivid, atmospheric, and deeply immersive — use sensory details (sight, sound, smell, touch), "
        "evocative language, and dramatic tension to draw the reader into the world. "
        "Vary sentence length and structure; build suspense, wonder, or dread as the scene demands. "
        "Every response must follow this EXACT structure:\n"
        "  1. Exactly two paragraphs of narrative prose (no more, no fewer).\n"
        "  2. A blank line.\n"
        "  3. Exactly three numbered choices, each on its own line, formatted as '1. ', '2. ', '3. '.\n"
        "The three choices must be meaningfully different — offer a bold action, a cautious action, and a creative or unexpected option. "
        "Choices should be concise (under 12 words each) and lead to genuinely different outcomes. "
        "Never break character. Never use meta-commentary, stage directions, or author notes. "
        "Never repeat previous text verbatim. "
        "Never output bracketed annotations like [Player chose], [Story so far], or [Scenario essentials]. "
        "Output only the two narrative paragraphs followed by the three choices — nothing else."
    )

    # Gather history text — use natural separators the LLM won't echo
    history_parts = []
    if adventure_state["summary"]:
        history_parts.append(f"Previously in the adventure: {adventure_state['summary']}")

    for entry in adventure_state["entries"]:
        if entry["type"] == "text":
            history_parts.append(strip_choice_lines(entry["content"]))
        elif entry["type"] == "choice":
            history_parts.append(f"> You {entry['content'][0].lower()}{entry['content'][1:]}")

    history_text = "\n\n".join(history_parts)

    # Check if we need to summarise
    total_len = len(system) + len(history_text) + len(user_action)
    if total_len > MAX_CONTEXT_CHARS and len(adventure_state["entries"]) > 4:
        # Generate a summary of everything so far
        adventure_state["summary"] = generate_summary(history_text)
        # Keep only the last 2 entries for direct context
        recent = adventure_state["entries"][-2:]
        adventure_state["entries"] = recent
        # Rebuild
        history_parts = [f"Previously in the adventure: {adventure_state['summary']}"]
        for entry in recent:
            if entry["type"] == "text":
                history_parts.append(strip_choice_lines(entry["content"]))
            elif entry["type"] == "choice":
                history_parts.append(f"> You {entry['content'][0].lower()}{entry['content'][1:]}")
        history_text = "\n\n".join(history_parts)

    if is_start:
        if has_starting_text:
            user_message = (
                "The story has already begun as shown above. "
                "Continue seamlessly from where it left off — do not repeat or rephrase the opening text. "
                "Write exactly two paragraphs of vivid, immersive narrative that advance the story, "
                "then present exactly 3 numbered choices."
            )
        elif user_action:
            user_message = (
                f"Begin a brand-new adventure with this theme or setting: {user_action}. "
                "Plunge the reader straight into the world — establish who they are, where they are, "
                "and an immediate hook or tension. Use rich sensory details. "
                "Write exactly two paragraphs of narrative, then present exactly 3 numbered choices."
            )
        else:
            user_message = (
                "Begin a brand-new adventure. Choose a compelling genre and setting. "
                "Plunge the reader straight into the world — establish who they are, where they are, "
                "and an immediate hook or tension. Use rich sensory details. "
                "Write exactly two paragraphs of narrative, then present exactly 3 numbered choices."
            )
    else:
        user_message = (
            "Continue the adventure from the player's last action. "
            "Advance the plot with consequences, revelations, or new developments. "
            "Write exactly two paragraphs of vivid narrative, then present exactly 3 numbered choices."
        )


    # Insert scenario essentials right after the system text so they are always prominent
    essentials = adventure_state.get("scenario_essentials", "").strip()
    parts = [system]
    if essentials:
        parts.append(
            f"IMPORTANT — The following scenario essentials define this adventure's world and must be "
            f"respected at all times. Every piece of narrative you write should align with and reinforce "
            f"these details:\n{essentials}"
        )
    if history_text:
        parts.append(history_text)
    parts.append(user_message)
    prompt = "\n\n".join(parts)

    return prompt


def generate_summary(text: str) -> str:
    """Ask the LLM to compress the adventure so far into a concise summary."""
    prompt = (
        "Summarise the following adventure story into a concise paragraph that preserves "
        "all key plot points, character details, items obtained, relationships, and the current situation. "
        "Keep the tone and atmosphere of the original. "
        "Write in second-person present tense.\n\n"
        f"{text}"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": runtime_config["text_model"], "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def stream_text_from_ollama(prompt: str):
    """Generator that streams tokens from Ollama."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": runtime_config["text_model"], "prompt": prompt, "stream": True},
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done"):
                break


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def list_models():
    """Return available Ollama models and the currently selected text model."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        models = []
    return jsonify({"models": models, "current": runtime_config["text_model"]})


@app.route("/api/set_model", methods=["POST"])
def set_model():
    """Change the text generation model."""
    model = (request.json or {}).get("model", "").strip()
    if not model:
        return jsonify({"error": "No model specified"}), 400
    runtime_config["text_model"] = model
    return jsonify({"success": True, "current": model})


@app.route("/api/start", methods=["POST"])
def start_adventure():
    """Start a new adventure, optionally with a theme and starting text."""
    reset_state()
    body = request.json or {}
    theme = body.get("theme", "")
    essentials = body.get("essentials", "").strip()
    starting_text = body.get("starting_text", "").strip()
    adventure_state["scenario_essentials"] = essentials
    adventure_state["started"] = True

    # If starting text is provided, add it as the first history entry
    if starting_text:
        add_entry({"type": "text", "content": starting_text})

    prompt = build_context_prompt(user_action=theme, is_start=True, has_starting_text=bool(starting_text))
    adventure_state["last_prompt"] = prompt

    def generate():
        full_text = []
        for token in stream_text_from_ollama(prompt):
            full_text.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
        complete = "".join(full_text)
        add_entry({"type": "text", "content": complete})
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/action", methods=["POST"])
def player_action():
    """Process a player action / choice."""
    action = (request.json or {}).get("action", "").strip()
    if not action:
        return jsonify({"error": "No action provided"}), 400

    # Record the choice
    add_entry({"type": "choice", "content": action})
    prompt = build_context_prompt(user_action=action)
    adventure_state["last_prompt"] = prompt

    def generate():
        full_text = []
        for token in stream_text_from_ollama(prompt):
            full_text.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
        complete = "".join(full_text)
        add_entry({"type": "text", "content": complete})
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


def _generate_image_worker(job_id: str, image_prompt: str):
    """Background worker that generates an image and stores the result."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": IMAGE_MODEL,
                "prompt": image_prompt,
                "stream": False,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
            },
            timeout=600,
        )
        resp.raise_for_status()
        result = resp.json()

        img_data = result.get("image", "") or result.get("response", "")

        if img_data:
            add_entry({"type": "image", "content": img_data})
            image_jobs[job_id] = {"status": "done", "image": img_data, "error": None}
        else:
            image_jobs[job_id] = {"status": "error", "image": None, "error": "No image data returned"}

    except Exception as e:
        image_jobs[job_id] = {"status": "error", "image": None, "error": str(e)}


@app.route("/api/generate_image", methods=["POST"])
def generate_image():
    """Start async image generation based on provided text or last paragraph."""
    body = request.json or {}
    para_text = body.get("text", "").strip()
    extra_context = body.get("context", "").strip()

    if not para_text:
        # Fallback: find last text entry
        last_text = ""
        for entry in reversed(adventure_state["entries"]):
            if entry["type"] == "text":
                last_text = entry["content"]
                break
        if not last_text:
            return jsonify({"error": "No adventure text to illustrate"}), 400
        cleaned = strip_choice_lines(last_text)
        paragraphs = [p.strip() for p in cleaned.strip().split("\n\n") if p.strip()]
        para_text = paragraphs[-1] if paragraphs else cleaned

    # Build image prompt — put user context first so it's never truncated
    if extra_context:
        image_prompt = f"Fantasy illustration, {extra_context}, detailed, dramatic lighting: {para_text[:1500]}"
    else:
        image_prompt = f"Fantasy illustration, detailed, dramatic lighting: {para_text[:1500]}"

    job_id = str(uuid.uuid4())
    image_jobs[job_id] = {"status": "pending", "image": None, "error": None}

    thread = threading.Thread(target=_generate_image_worker, args=(job_id, image_prompt), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/image_status/<job_id>")
def image_status(job_id):
    """Poll for image generation status."""
    job = image_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/api/undo", methods=["POST"])
def undo():
    """Remove the last entry from adventure history."""
    if adventure_state["entries"]:
        removed = pop_entry()
        # If we removed a text entry that was preceded by a choice, also remove the choice
        # so the state is consistent
        if removed["type"] == "text" and adventure_state["entries"] and adventure_state["entries"][-1]["type"] == "choice":
            pop_entry()
        return jsonify({"success": True, "removed_type": removed["type"]})
    return jsonify({"error": "Nothing to undo"}), 400


@app.route("/api/regenerate", methods=["POST"])
def regenerate_last():
    """Remove the last AI text and regenerate from the most recent choice/state."""
    if not adventure_state["entries"]:
        return jsonify({"error": "Nothing to regenerate"}), 400

    # Remove the last text entry
    if adventure_state["entries"][-1]["type"] == "text":
        pop_entry()

    # Rebuild prompt from current state
    # Check if there's a choice to use as the action
    last_action = ""
    if adventure_state["entries"] and adventure_state["entries"][-1]["type"] == "choice":
        last_action = adventure_state["entries"][-1]["content"]

    is_start = not any(e["type"] == "choice" for e in adventure_state["entries"])
    has_starting = is_start and any(e["type"] == "text" for e in adventure_state["entries"])
    prompt = build_context_prompt(user_action=last_action, is_start=is_start, has_starting_text=has_starting)
    adventure_state["last_prompt"] = prompt

    def generate():
        full_text = []
        for token in stream_text_from_ollama(prompt):
            full_text.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
        complete = "".join(full_text)
        add_entry({"type": "text", "content": complete})
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/update_last_text", methods=["POST"])
def update_last_text():
    """Update the last text entry in history (for in-place editing)."""
    new_text = (request.json or {}).get("text", "").strip()
    if not new_text:
        return jsonify({"error": "No text provided"}), 400

    # Find and update the last text entry in both lists
    changed = False
    for entries_list in [adventure_state["entries"], adventure_state["full_history"]]:
        for i in range(len(entries_list) - 1, -1, -1):
            if entries_list[i]["type"] == "text":
                if entries_list[i]["content"] != new_text:
                    changed = True
                entries_list[i]["content"] = new_text
                break

    return jsonify({"success": True, "changed": changed})


@app.route("/api/history")
def get_history():
    """Return full adventure history."""
    return jsonify({"entries": adventure_state["full_history"]})


@app.route("/api/last_prompt")
def get_last_prompt():
    """Return the last prompt sent to the text model."""
    return jsonify({"prompt": adventure_state.get("last_prompt", "")})


@app.route("/api/download")
def download_adventure():
    """Generate a self-contained HTML file of the adventure."""
    entries = adventure_state["full_history"]
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>My Adventure — AI Choose Now!</title>
<style>
  body { font-family: 'Georgia', serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; line-height: 1.8; }
  h1 { text-align: center; color: #c9a84c; font-size: 2em; border-bottom: 2px solid #c9a84c; padding-bottom: 10px; }
  .narrative { margin: 20px 0; white-space: pre-wrap; }
  .choice { color: #7eb8da; font-style: italic; margin: 10px 0; padding: 10px; background: rgba(126,184,218,0.1); border-left: 3px solid #7eb8da; }
  .adventure-image { max-width: 100%; border-radius: 8px; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
  hr { border: none; border-top: 1px solid #333; margin: 30px 0; }
</style>
</head>
<body>
<h1>My Adventure</h1>
""")

    for entry in entries:
        if entry["type"] == "text":
            cleaned = strip_choice_lines(entry["content"])
            text = cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f'<div class="narrative">{text}</div>\n<hr>\n')
        elif entry["type"] == "choice":
            choice = entry["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f'<div class="choice">{format_choice_as_narrative(choice)}</div>\n')
        elif entry["type"] == "image":
            html_parts.append(f'<img class="adventure-image" src="data:image/png;base64,{entry["content"]}" alt="Adventure illustration">\n')

    html_parts.append("</body></html>")
    html_content = "".join(html_parts)

    return Response(
        html_content,
        mimetype="text/html",
        headers={"Content-Disposition": "attachment; filename=my_adventure.html"},
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
