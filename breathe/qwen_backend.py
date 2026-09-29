"""
Qwen3 backend for the breathing loop.

The one unusual part is `trust_sdpa`: an attention function that adds
    gain * log(trust_j)
to the attention logits of every key position that belongs to cue j, in every
layer and head (or only the layers you choose).  A cue with trust 1 is
untouched; a cue with trust 0.1 is attended to about e^-2.3 ≈ 10x less.
Nothing in the prompt or the weights changes — the precision lives outside the
model, like AdaptiveObserverCache's observer.

When no trust is set, the function hands straight back to the stock SDPA path,
so the model is byte-for-byte the normal one.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F_t

from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv

from .engine import clean_answer

ATTN_NAME = "trust_sdpa"

SYSTEM_RECALL = (
    "You help someone recall a word or name that is on the tip of their tongue. "
    "Reply with only the word or name, nothing else."
)
HEADER_RECALL = "I'm trying to remember something. Here is what I remember:"
SYSTEM_FIT = ("Someone is describing a word or name they cannot quite remember. "
              "Judge whether one thing they said fits a proposed answer. Answer with a single word: Yes or No.")


# ----------------------------------------------------------------------------- attention hook

class _TrustState:
    """Module-level switch read by the attention function."""

    def __init__(self):
        self.bias: torch.Tensor | None = None   # [P] or [B, P]: additive logit bias per prompt position
        self.layers: set[int] | None = None     # None = all layers

    def for_call(self, module, kv_len: int, like: torch.Tensor) -> torch.Tensor | None:
        if self.bias is None:
            return None
        if self.layers is not None and getattr(module, "layer_idx", -1) not in self.layers:
            return None
        src = self.bias if self.bias.dim() == 2 else self.bias[None]
        B, P = src.shape
        b = torch.zeros(B, kv_len, dtype=like.dtype, device=like.device)
        n = min(P, kv_len)
        b[:, :n] = src[:, :n].to(device=like.device, dtype=like.dtype)
        return b.view(B, 1, 1, kv_len)   # positions generated after the prompt get no bias


TRUST = _TrustState()


def trust_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    bias = TRUST.for_call(module, key.shape[2], query)
    if bias is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    n_rep = getattr(module, "num_key_value_groups", 1)
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)
    q_len, kv_len = query.shape[2], key.shape[2]
    neg = torch.finfo(query.dtype).min
    if attention_mask is None:
        # causal, right-aligned so the last query sees every key (works with a KV cache)
        qi = torch.arange(q_len, device=query.device)[:, None] + (kv_len - q_len)
        ki = torch.arange(kv_len, device=query.device)[None, :]
        base = torch.where(ki <= qi, 0.0, neg).to(query.dtype)[None, None]
    elif attention_mask.dtype == torch.bool:
        base = torch.where(attention_mask[..., :kv_len], 0.0, neg).to(query.dtype)
    else:
        base = attention_mask[..., :kv_len].to(query.dtype)
    mask = (base + bias).clamp(min=neg)
    out = F_t.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                           dropout_p=0.0, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def register_attention():
    AttentionInterface.register(ATTN_NAME, trust_sdpa)
    try:  # transformers >= 4.53 builds masks per attention implementation
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)
    except Exception:
        pass


register_attention()


@contextlib.contextmanager
def precision(bias: torch.Tensor | None, layers: set[int] | None = None):
    old = (TRUST.bias, TRUST.layers)
    TRUST.bias, TRUST.layers = bias, layers
    try:
        yield
    finally:
        TRUST.bias, TRUST.layers = old


# ----------------------------------------------------------------------------- backend

@dataclass
class Prompt:
    ids: torch.Tensor          # [1, P]
    text: str
    cue_tokens: list[list[int]]


class QwenBackend:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", *, model=None, tokenizer=None,
                 device: str | None = None, dtype=None, load_4bit: bool = False, load_8bit: bool = False,
                 attn_gain: float = 1.0, attn_layers: set[int] | None = None,
                 max_new_tokens: int = 12, top_p: float = 0.95, fit_batch: int = 16, seed: int = 0):
        self.attn_gain = attn_gain
        self.attn_layers = attn_layers
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.fit_batch = fit_batch
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

        if model is None:
            kw: dict = {"attn_implementation": ATTN_NAME}
            if load_4bit or load_8bit:
                import importlib.util
                if importlib.util.find_spec("bitsandbytes") is None:
                    raise SystemExit("--load-4bit / --load-8bit need bitsandbytes:\n"
                                     "    python -m pip install -U bitsandbytes")
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=load_4bit, load_in_8bit=load_8bit,
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
                kw["device_map"] = "auto"
            else:
                # transformers >= 4.56 calls it `dtype`; older versions silently ignore that
                # name and load float32, so pick the right one explicitly
                key = "dtype" if _tf_version() >= (4, 56) else "torch_dtype"
                kw[key] = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
                kw["device_map"] = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model.eval()
        self.tok = tokenizer
        self.device = next(self.model.parameters()).device

        stop = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                stop.add(i)
        self.stop_ids = torch.tensor(sorted(s for s in stop if s is not None), device=self.device)
        self.yes_ids = self._single_ids(["Yes", "yes", " Yes"])
        self.no_ids = self._single_ids(["No", "no", " No"])

    # ------------------------------------------------------------------ helpers

    def _single_ids(self, words):
        ids = []
        for w in words:
            e = self.tok.encode(w, add_special_tokens=False)
            if len(e) == 1:
                ids.append(e[0])
        if not ids:
            raise ValueError(f"no single-token form among {words}")
        return torch.tensor(sorted(set(ids)), device=self.device)

    def _chat(self, system: str | None, user: str, thinking: bool = False) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=thinking)

    def _encode_with_spans(self, text: str, spans: list[tuple[int, int]]) -> Prompt:
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        cue_tokens = []
        for (a, b) in spans:
            cue_tokens.append([i for i, (s, e) in enumerate(offsets) if e > a and s < b])
        ids = torch.tensor([enc["input_ids"]], device=self.device)
        return Prompt(ids=ids, text=text, cue_tokens=cue_tokens)

    def recall_prompt(self, cues: list[str], thinking: bool = False) -> Prompt:
        user = HEADER_RECALL + "\n" + "\n".join(f"- {c}" for c in cues) + "\nWhat is it?"
        text = self._chat(SYSTEM_RECALL, user, thinking=thinking)
        return self._encode_with_spans(text, _locate(text, cues, start=text.find(HEADER_RECALL)))

    def raw_prompt(self, system: str | None, message: str, cues: list[str]) -> Prompt:
        text = self._chat(system, message)
        return self._encode_with_spans(text, _locate(text, cues, start=text.rfind(message[:20])))

    def bias_vector(self, prompt: Prompt, trust) -> torch.Tensor | None:
        if trust is None or self.attn_gain == 0:
            return None
        P = prompt.ids.shape[1]
        b = torch.zeros(P, device=self.device)
        for toks, w in zip(prompt.cue_tokens, trust):
            if toks:
                b[toks] = self.attn_gain * math.log(max(float(w), 1e-4))
        return b

    @torch.no_grad()
    def _generate(self, ids: torch.Tensor, n: int, temperature: float, max_new: int) -> list[list[int]]:
        ids = ids.repeat(n, 1)
        out = self.model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        done = torch.zeros(n, dtype=torch.bool, device=self.device)
        gen: list[list[int]] = [[] for _ in range(n)]
        for _ in range(max_new):
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                probs = torch.softmax(logits / temperature, -1)
                sp, si = probs.sort(-1, descending=True)
                keep = sp.cumsum(-1) - sp < self.top_p
                sp = sp * keep
                pick = torch.multinomial(sp.cpu(), 1, generator=self.gen).to(self.device)
                nxt = si.gather(-1, pick).squeeze(-1)
            for i in range(n):
                if not done[i]:
                    gen[i].append(int(nxt[i]))
            done |= torch.isin(nxt, self.stop_ids)
            if bool(done.all()):
                break
            out = self.model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
        return gen

    def _decode(self, toks: list[int]) -> str:
        toks = [t for t in toks if t not in set(self.stop_ids.tolist())]
        return self.tok.decode(toks, skip_special_tokens=True)

    # ------------------------------------------------------------------ backend API

    def propose(self, cues, trust, n: int = 1, temperature: float = 0.0) -> list[str]:
        p = self.recall_prompt(cues)
        with precision(self.bias_vector(p, trust), self.attn_layers):
            gens = self._generate(p.ids, n, temperature, self.max_new_tokens)
        return [clean_answer(self._decode(g)) for g in gens]

    def propose_multi(self, cues, trusts, n_each: int = 1, temperature: float = 0.0) -> list[list[str]]:
        """Several trust settings in ONE batch (the prompt tokens are identical; only the
        per-row attention bias differs).  Returns one list of n_each answers per trust row."""
        p = self.recall_prompt(cues)
        rows = [self.bias_vector(p, w) for w in trusts]
        if all(r is None for r in rows):
            bias = None
        else:
            P = p.ids.shape[1]
            bias = torch.stack([(r if r is not None else torch.zeros(P, device=self.device))
                                for r in rows for _ in range(n_each)])
        with precision(bias, self.attn_layers):
            gens = self._generate(p.ids, len(trusts) * n_each, temperature, self.max_new_tokens)
        answers = [clean_answer(self._decode(g)) for g in gens]
        return [answers[i * n_each:(i + 1) * n_each] for i in range(len(trusts))]

    @torch.no_grad()
    def compat(self, cues: list[str], candidates: list[str]) -> np.ndarray:
        """fit[j, c] = P(Yes) / (P(Yes) + P(No)) for 'is clue j true of candidate c?'"""
        pairs = [(j, c) for c in range(len(candidates)) for j in range(len(cues))]
        texts = [self._chat(SYSTEM_FIT,
                            f"They said: \"{cues[j]}\"\nProposed answer: {candidates[c]}\n"
                            f"Does what they said fit the proposed answer?") for j, c in pairs]
        fit = np.zeros((len(cues), len(candidates)))
        old_side = self.tok.padding_side
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            with precision(None):
                for s in range(0, len(texts), self.fit_batch):
                    chunk = texts[s:s + self.fit_batch]
                    enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                    logits = self.model(**enc).logits[:, -1, :].float()
                    lp = torch.log_softmax(logits, -1)
                    yes = torch.logsumexp(lp[:, self.yes_ids], -1)
                    no = torch.logsumexp(lp[:, self.no_ids], -1)
                    f = torch.sigmoid(yes - no).cpu().numpy()
                    for k, (j, c) in enumerate(pairs[s:s + self.fit_batch]):
                        fit[j, c] = f[k]
        finally:
            self.tok.padding_side = old_side
        return fit

    # ------------------------------------------------------------------ baselines and voice

    def one_shot(self, cues) -> str:
        return self.propose(cues, None, n=1, temperature=0.0)[0]

    def think(self, cues, budget: int = 768) -> str:
        """Qwen3 thinking mode with a token budget; the answer after </think> is returned."""
        p = self.recall_prompt(cues, thinking=True)
        with precision(None):
            # Qwen recommends sampling (not greedy) in thinking mode
            g = self._generate(p.ids, 1, 0.6, budget + self.max_new_tokens)[0]
            raw = self.tok.decode(g, skip_special_tokens=False)
            if "</think>" not in raw:  # budget ran out: close the thought and ask for the answer
                body = self.tok.decode([t for t in g if t not in set(self.stop_ids.tolist())],
                                       skip_special_tokens=False)
                forced = p.text + body + "\n</think>\n\n"
                ids = self.tok(forced, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
                g2 = self._generate(ids, 1, 0.0, self.max_new_tokens)[0]
                return clean_answer(self._decode(g2))
        after = raw.split("</think>")[-1]
        for t in ("<|im_end|>", "<|endoftext|>"):
            after = after.replace(t, "")
        return clean_answer(after)

    def reply(self, message: str, cues: list[str], trust=None, note: str | None = None,
              max_new: int = 160) -> str:
        """A normal chat reply to the user's own words.  With `trust`, the same attention
        precision is applied to the user's cue spans; with `note`, the settled result is
        given to the model as a system message so it can say it in its own words."""
        system = None
        if note:
            system = ("You are a helpful assistant. A background recall process has already worked on "
                      "the user's question. " + note +
                      " Answer the user naturally in one to three sentences. If a clue was set aside, "
                      "say so briefly and kindly, and say what the other clues pointed to instead.")
        p = self.raw_prompt(system, message, cues)
        with precision(self.bias_vector(p, trust), self.attn_layers):
            g = self._generate(p.ids, 1, 0.0, max_new)[0]
        text = self._decode(g)
        return text.split("</think>")[-1].strip()


def _tf_version() -> tuple[int, int]:
    import transformers
    major, minor = transformers.__version__.split(".")[:2]
    return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)


def _locate(text: str, cues: list[str], start: int = 0) -> list[tuple[int, int]]:
    spans, pos = [], max(0, start)
    for c in cues:
        i = text.find(c, pos)
        if i < 0:
            i = text.find(c)
        if i < 0:
            spans.append((0, 0))
        else:
            spans.append((i, i + len(c)))
            pos = i + len(c)
    return spans
