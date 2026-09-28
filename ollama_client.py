"""HTTP client for Ollama's native model and image-generation APIs."""

import os

import httpx2


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> list[str]:
        with httpx2.Client(timeout=10) as client:
            response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return [model["name"] for model in response.json().get("models", [])]

    def generate_image(self, model: str, prompt: str, width: int, height: int) -> str:
        with httpx2.Client(timeout=600) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "width": width,
                    "height": height,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("image", "") or result.get("response", "")


ollama_client = OllamaClient()
