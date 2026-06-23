"""Transformer-style encoder used by the LLM module.

This file exposes a small re-export of the DistilBERT-based encoder defined
in `llm_module.py`. Other modules that need only the encoder (for example,
offline experiments that bypass the LLM call and embed canned responses)
can import directly from here without instantiating the asynchronous LLM
machinery.
"""

from .llm_module import DistilBertEncoder, StubEncoder

__all__ = ["DistilBertEncoder", "StubEncoder"]
