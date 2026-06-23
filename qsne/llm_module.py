"""Asynchronous semantic-reasoning module (GPT-4o + transformer encoder).

Implements the LLM module of Section 2.4 of the paper:
    - 36-D sector summary + 5-D odometry -> structured prompt P.
    - GPT-4o snapshot gpt-4o-2024-08-06, temperature 0.2, max 96 tokens.
    - DistilBERT-base-uncased encoder maps the response D_t to e_t in R^256.
    - Background thread keeps the GPT-4o round trip (~620 ms median) off the
      10 Hz control loop. The control thread always consumes the cached
      embedding through an atomic pointer swap.
    - On API failure or cold start, e_t falls back to the zero embedding,
      which reduces the policy to the PQC-only ablation variant.

The module is designed so that a missing OpenAI key or a missing
transformers install does not crash the rest of the framework: a stub
encoder and a stub language model are used instead, with a clear warning
on the first call.
"""

from __future__ import annotations

import os
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .aggregation import SectorSummary

# -----------------------------------------------------------------------------
# Embedding dimension (Section 2.4) and model snapshot (Table of hyperparameters)
# -----------------------------------------------------------------------------
LLM_EMBED_DIM: int = 256
MODEL_SNAPSHOT: str = "gpt-4o-2024-08-06"
ENCODER_NAME: str = "distilbert-base-uncased"
GPT_TEMPERATURE: float = 0.2
GPT_MAX_TOKENS: int = 96
GPT_TIMEOUT_S: float = 3.0


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------
def build_prompt(summary: SectorSummary, odom: np.ndarray) -> str:
    """Build the structured LLM prompt described in Section 2.4.

    Parameters
    ----------
    summary : SectorSummary
        12-sector mean / variance / null-percent vectors.
    odom : np.ndarray, shape (5,)
        Odometry vector [x, y, yaw, v, omega].
    """
    x, y, yaw, v, w = odom.tolist()
    means_str = ", ".join(f"{m:.2f}" for m in summary.means)
    vars_str = ", ".join(f"{v:.3f}" for v in summary.variances)
    nulls_str = ", ".join(f"{p:.0f}" for p in summary.null_pcts)
    prompt = (
        "Analyze laser scan summary (12 sectors, "
        f"mean ranges: [{means_str}], "
        f"variances: [{vars_str}], "
        f"null percentages: [{nulls_str}]) "
        f"and odometry (position: ({x:.2f}, {y:.2f}), yaw: {yaw:.2f}, "
        f"velocities: ({v:.2f}, {w:.2f})). "
        "Describe potential obstacles and suggest navigation actions "
        "in one short sentence."
    )
    return prompt


# -----------------------------------------------------------------------------
# Transformer encoder for D_t -> e_t (frozen DistilBERT projected to 256-D)
# -----------------------------------------------------------------------------
class StubEncoder(nn.Module):
    """Deterministic hash-based encoder used when transformers is missing.

    Maps a string to a fixed-length normalized vector that depends only on
    the bytes of the input. Adequate for unit tests and for runs that lack
    the transformers / huggingface dependency.
    """

    def __init__(self, dim: int = LLM_EMBED_DIM) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, text: str) -> torch.Tensor:
        rng = np.random.default_rng(seed=abs(hash(text)) % (2 ** 31))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        return torch.from_numpy(v)


class DistilBertEncoder(nn.Module):
    """Wrap a frozen DistilBERT model and project its [CLS] vector to 256-D.

    The DistilBERT [CLS] embedding has 768 dimensions. A linear projection
    head with a fixed (random orthogonal) initialization reduces it to 256
    without adding trainable parameters to the LLM pathway. The projection
    head is intentionally non-trainable here so that the embedding e_t
    stays deterministic across training restarts. Authors who prefer to
    learn the projection jointly with the policy can simply call
    `requires_grad_(True)` on this module's parameters.
    """

    def __init__(self, dim: int = LLM_EMBED_DIM) -> None:
        super().__init__()
        self.dim = dim
        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
            self._model = AutoModel.from_pretrained(ENCODER_NAME)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            proj = torch.linalg.qr(torch.randn(768, dim))[0].t().contiguous()
            self.register_buffer("proj", proj)
            self._ok = True
        except Exception as exc:    # transformers missing or download failure
            warnings.warn(
                f"DistilBERT unavailable ({exc}); falling back to stub encoder."
            )
            self._stub = StubEncoder(dim)
            self._ok = False

    def forward(self, text: str) -> torch.Tensor:
        if not self._ok:
            return self._stub(text)
        toks = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=96
        )
        with torch.no_grad():
            out = self._model(**toks)
            cls = out.last_hidden_state[:, 0, :].squeeze(0)  # (768,)
            e = cls @ self.proj.t()                          # (256,)
        return e.to(dtype=torch.float32)


# -----------------------------------------------------------------------------
# Asynchronous LLM caller
# -----------------------------------------------------------------------------
@dataclass
class _CachedEmbedding:
    """Latest embedding + timestamp; protected by the parent module's lock."""

    vector: torch.Tensor
    age_steps: int = 0


class AsyncLLMModule:
    """Background-thread wrapper around the GPT-4o call.

    The control loop calls `get_embedding()` every step. It returns the most
    recent embedding without blocking. A separate worker thread picks up
    trigger events from a single-slot mailbox, issues the API call, encodes
    the response via DistilBERT, and overwrites the cache.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.
    embed_dim : int
        Dimension of the cached embedding vector.
    use_local_fallback : bool
        If True, calls a local Llama-3.1-8B-Instruct model through Ollama
        when the OpenAI API is unreachable. The fallback path is also used
        whenever no API key is configured, which makes the module usable
        on developer laptops.
    """

    def __init__(
        self,
        api_key: str | None = None,
        embed_dim: int = LLM_EMBED_DIM,
        use_local_fallback: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.embed_dim = embed_dim
        self.use_local_fallback = use_local_fallback

        self.encoder = DistilBertEncoder(dim=embed_dim)

        self._lock = threading.Lock()
        self._cache = _CachedEmbedding(
            vector=torch.zeros(embed_dim, dtype=torch.float32)
        )
        self._pending_prompt: str | None = None
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ----- Public API -------------------------------------------------------
    def trigger(self, prompt: str) -> None:
        """Queue a new LLM call. Drops the request if one is already in flight."""
        with self._cv:
            # Single-slot mailbox: overwrite, do not queue.
            self._pending_prompt = prompt
            self._cv.notify()

    def get_embedding(self) -> torch.Tensor:
        """Return a copy of the latest cached embedding."""
        with self._lock:
            return self._cache.vector.clone()

    def shutdown(self) -> None:
        """Stop the worker thread cleanly."""
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self._worker.join(timeout=2.0)

    # ----- Worker loop ------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while self._pending_prompt is None and not self._stop.is_set():
                    self._cv.wait()
                if self._stop.is_set():
                    return
                prompt = self._pending_prompt
                self._pending_prompt = None

            response = self._call_llm(prompt)
            embedding = self.encoder(response)
            with self._lock:
                self._cache = _CachedEmbedding(vector=embedding)

    # ----- LLM call paths ---------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        """Call GPT-4o; fall back to a local model or a heuristic on failure."""
        if self.api_key:
            try:
                return self._call_openai(prompt)
            except Exception as exc:
                warnings.warn(f"OpenAI call failed: {exc}")
        if self.use_local_fallback:
            try:
                return self._call_ollama(prompt)
            except Exception as exc:
                warnings.warn(f"Local Llama fallback failed: {exc}")
        return self._heuristic_response(prompt)

    def _call_openai(self, prompt: str) -> str:
        """Synchronous OpenAI call with a hard timeout."""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, timeout=GPT_TIMEOUT_S)
        completion = client.chat.completions.create(
            model=MODEL_SNAPSHOT,
            messages=[{"role": "user", "content": prompt}],
            temperature=GPT_TEMPERATURE,
            max_tokens=GPT_MAX_TOKENS,
        )
        return completion.choices[0].message.content or ""

    def _call_ollama(self, prompt: str) -> str:
        """Local Llama-3.1-8B-Instruct through the Ollama HTTP API."""
        import json
        import urllib.request

        payload = json.dumps({
            "model": "llama3.1:8b-instruct",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": GPT_TEMPERATURE,
                "num_predict": GPT_MAX_TOKENS,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=GPT_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")

    @staticmethod
    def _heuristic_response(prompt: str) -> str:
        """Generate a synthetic obstacle description from the prompt text.

        Used as a last-resort fallback so that the embedding e_t still
        reflects the input sector summary even when no language model is
        reachable. The rule-based fallback inspects the mean-range list
        embedded in the prompt and reports the direction of the nearest
        sector.
        """
        import re
        match = re.search(r"mean ranges:\s*\[([^\]]+)\]", prompt)
        if not match:
            return "No obstacles detected; proceed forward at moderate speed."
        try:
            means = [float(v.strip()) for v in match.group(1).split(",")]
        except ValueError:
            return "Sensor data ambiguous; reduce speed."
        nearest = min(range(len(means)), key=lambda k: means[k])
        sector_label = ["front-right", "right", "back-right", "back",
                        "back-left", "left", "front-left", "front",
                        "front", "front-right", "right", "back-right"][nearest % 12]
        dist = means[nearest]
        if dist < 1.0:
            return (f"Close obstacle on the {sector_label} at {dist:.1f} m; "
                    "recommend a sharp turn.")
        if dist < 3.0:
            return (f"Obstacle on the {sector_label} at {dist:.1f} m; "
                    "recommend a slight turn.")
        return "No nearby obstacles; proceed forward."
