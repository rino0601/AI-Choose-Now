# ⚔️ AI Choose Now!

An interactive choose-your-own-adventure game powered by local AI models via [Ollama](https://ollama.com) by default. Text generation uses the OpenAI Python SDK with Ollama's OpenAI-compatible API; compatible providers can also be configured. Image generation continues to use Ollama's native API.

## Features

- **Streaming narrative** — AI-generated story text streams in real-time as it's written
- **Clickable choices** — each story section ends with 3 distinct choices; click one or type your own action
- **Per-paragraph image generation** — generate illustrations for any paragraph using a local image model
- **Image controls** — regenerate, add context with a paintbrush prompt, or delete images
- **In-place text editing** — edit any story block directly and save changes to history
- **Regenerate** — re-roll the last story section for a different take
- **Undo** — step back through your adventure
- **Scenario essentials** — define persistent world details (setting, characters, rules) that the AI respects throughout
- **Story starting text** — optionally provide your own opening passage for the AI to continue from
- **Save/load presets** — save scenario + starting text to a JSON file and load them later
- **Model selection** — switch between any Ollama text model at runtime via the settings gear
- **Download** — export your complete adventure as a self-contained HTML file
- **Full story view** — read the entire adventure in a scrollable modal
- **View last prompt** — inspect the exact prompt sent to the AI for debugging
- **Dark theme** — elegant dark UI with gold accents, designed for immersive reading

## Requirements

- **Python 3.10+** (tested with 3.13)
- **Ollama** with at least one text generation model installed and an image generation model for illustrations; tested with `gemma3:4b` for text and `x/flux2-klein:latest` for images
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/) for Python dependency and environment management

## Installation

### macOS

1. **Install Ollama:**

   Download from [ollama.com](https://ollama.com) or install via Homebrew:

   ```bash
   brew install ollama
   ```

2. **Pull a text model** (the default is `gemma3:4b`, but any Ollama text model works):

   ```bash
   ollama pull gemma3:4b
   ```

3. **(Optional) Pull an image model** for illustration generation:

   ```bash
   ollama pull x/flux2-klein:latest
   ```

4. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/aiAdv.git
   cd aiAdv
   ```

5. **Install Python dependencies:**

   ```bash
   uv sync
   ```

6. **Start Ollama** (if not already running):

   ```bash
   ollama serve
   ```

7. **Run the app:**

   ```bash
   uv run python app.py
   ```

8. **Open in your browser:** [http://localhost:5050](http://localhost:5050)

### Linux

Install uv and follow the same steps as macOS. Install Ollama with:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then follow steps 2–8 above.

### Windows

1. **Install Ollama** from [ollama.com](https://ollama.com) (Windows installer available).
2. **Install Python 3.10+** from [python.org](https://www.python.org/downloads/).
3. Install uv, then open a terminal (PowerShell or Command Prompt) and follow steps 2–8 above, substituting:
   ```
   uv run python app.py
   ```

## Configuration

The default models are set at the top of `app.py`:

```python
TEXT_MODEL = "gemma3:4b"       # can also be changed at runtime via the ⚙️ menu
IMAGE_MODEL = "x/flux2-klein:latest"
```

You can change `TEXT_MODEL` to any Ollama text model, or switch models at runtime using the ⚙️ settings gear in the app header. The image model can be changed by editing `IMAGE_MODEL` in `app.py`.

Other configurable values:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API endpoint, used for image generation and model discovery; also provides the default text API URL |
| `TEXT_API_BASE_URL` | `<OLLAMA_URL>/v1` | OpenAI-compatible chat completions endpoint for text generation |
| `TEXT_API_KEY` | `ollama` | API key for the configured text provider; Ollama ignores this placeholder |
| `MAX_CONTEXT_CHARS` | `10000` | Character budget before history is summarised |
| `IMAGE_WIDTH` | `400` | Generated image width in pixels |
| `IMAGE_HEIGHT` | `400` | Generated image height in pixels |

Set `OLLAMA_URL` in the environment to use an Ollama server root other than `http://127.0.0.1:11434`; leave off `/v1` because the app appends it for text generation. In WSL, use an address reachable from the Linux environment. To use another OpenAI-compatible text provider, set both `TEXT_API_BASE_URL` and `TEXT_API_KEY`; these settings affect text generation only. Model discovery and image generation still use Ollama. Compatible providers may support only part of the OpenAI chat completions API, so model and parameter support depends on the provider.

## Usage

1. **Start screen** — Fill in Scenario Essentials (world details the AI follows throughout) and optionally Story Starting Text (your own opening passage).
2. **Save/Load** — Use 💾 Save and 📂 Load to store and recall presets.
3. **Play** — Click one of the 3 choices or type a custom action in the input bar.
4. **Images** — Click 🖼️ on any paragraph to generate an illustration. After generation, use 🔄 to regenerate, 🖌️ to add context, or ✖ to remove.
5. **Edit** — Click 📝 on the last story block to edit text in-place.
6. **Regenerate** — Click 🔄 in the header to get a fresh version of the last story section.
7. **Undo** — Click ↩ to step back.
8. **Menu** — Use ☰ for Full Story, Download, Last Prompt, About, and New Adventure.

## Network Access

The server binds to `0.0.0.0:5050` by default, making it accessible to other devices on your local network. Access it from another device using your machine's IP address (e.g., `http://192.168.1.100:5050`).

## Development

AI Choose Now! was developed through an iterative, conversational process between the author and Claude Opus 4.6 (via GitHub Copilot). The entire application — Flask backend, single-page frontend, prompt engineering, and UI/UX design — was built through a series of natural-language prompts describing desired features, which the AI then implemented.

The development process involved approximately 20 iterative rounds covering:

- Initial full-stack app creation (Flask + SSE streaming + dark-themed SPA)
- Image generation integration with async threading and polling
- Prompt engineering to control narrative structure (2 paragraphs + 3 choices)
- UI compaction (hamburger menu, responsive header, mobile support)
- History management (dual-list architecture for working context vs. full history)
- In-place content editing with backend synchronisation
- Model selection at runtime via Ollama API
- Bug fixes for LLM output contamination (leaked markers, broken choice parsing)

See the **ℹ️ About** page in the app's hamburger menu for the full list of development prompts.

## Author

**Edward Hewlett**

Built with [Claude Opus 4.6](https://www.anthropic.com/) via GitHub Copilot.

## License

This project is provided as-is for personal and educational use.
