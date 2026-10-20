"""
Embeddings and candidate generation.

* Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-d) when the `ml` extra is installed and
  EMBEDDING_BACKEND=sentence-transformers; otherwise a deterministic 384-d feature-hashing embedder
  (word unigrams + bigrams) so the platform works with zero model downloads.
* Generation: a local HuggingFace model (default Qwen2.5-0.5B-Instruct) or any LiteLLM model.
  Both return per-token top-5 log-probs so the active-learning entropy signal is real.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import get_settings
from app.services.active_learning import DEFAULT_TOP_K, token_entropy

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic signed feature hashing; no downloads, stable across processes."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        toks = _TOKEN_RE.findall(text.lower())
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=float)
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = np.linalg.norm(vec)
            out.append((vec / norm if norm else vec).round(6).tolist())
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.round(6).tolist() for v in vecs]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(s.embedding_model)
    return HashingEmbedder(s.embedding_dim)


@dataclass
class Candidate:
    text: str
    model: str
    entropy: float | None


class CandidateGenerator(Protocol):
    name: str

    def generate(self, system_prompt: str | None, user_prompt: str, temperature: float) -> Candidate: ...


def _messages(system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


class LocalHFGenerator:
    """Local transformers generation with per-token top-k log-probs."""

    def __init__(self, model_name: str, max_new_tokens: int = 384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()

    def generate(self, system_prompt: str | None, user_prompt: str, temperature: float) -> Candidate:
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(
            _messages(system_prompt, user_prompt), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
        top_lps = []
        for step_scores in out.scores:
            logp = torch.log_softmax(step_scores[0].float(), dim=-1)
            top_lps.append(torch.topk(logp, DEFAULT_TOP_K).values.tolist())
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return Candidate(text=text, model=self.name, entropy=round(token_entropy(top_lps), 6))


class LiteLLMGenerator:
    def __init__(self, model: str, max_new_tokens: int = 384):
        import litellm

        self._litellm = litellm
        self.name = model
        self.max_new_tokens = max_new_tokens

    def generate(self, system_prompt: str | None, user_prompt: str, temperature: float) -> Candidate:
        resp = self._litellm.completion(
            model=self.name,
            messages=_messages(system_prompt, user_prompt),
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            logprobs=True,
            top_logprobs=DEFAULT_TOP_K,
        )
        choice = resp.choices[0]
        entropy = None
        try:
            content = choice.logprobs["content"] if isinstance(choice.logprobs, dict) else choice.logprobs.content
            top = [[alt["logprob"] if isinstance(alt, dict) else alt.logprob for alt in (
                tok["top_logprobs"] if isinstance(tok, dict) else tok.top_logprobs)] for tok in content]
            entropy = round(token_entropy(top), 6)
        except Exception:  # provider does not return log-probs
            entropy = None
        return Candidate(text=(choice.message.content or "").strip(), model=self.name, entropy=entropy)


@lru_cache
def get_generator() -> CandidateGenerator | None:
    s = get_settings()
    if s.generator_backend == "local":
        return LocalHFGenerator(s.base_model, s.generator_max_new_tokens)
    if s.generator_backend == "litellm":
        if not s.generator_litellm_model:
            raise RuntimeError("GENERATOR_BACKEND=litellm requires GENERATOR_LITELLM_MODEL")
        return LiteLLMGenerator(s.generator_litellm_model, s.generator_max_new_tokens)
    return None


def generate_pair(
    generator: CandidateGenerator, system_prompt: str | None, user_prompt: str
) -> tuple[Candidate, Candidate]:
    """Two samples at different temperatures: a conservative answer and an exploratory one."""
    return (
        generator.generate(system_prompt, user_prompt, temperature=0.7),
        generator.generate(system_prompt, user_prompt, temperature=1.1),
    )
