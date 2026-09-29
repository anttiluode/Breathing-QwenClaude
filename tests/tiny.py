"""A tiny random Qwen3 + a small byte-level BPE tokenizer with a Qwen-style chat template.
Used only to check that the mechanics are wired correctly — it knows nothing."""
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)

CORPUS = [
    "I'm trying to remember something. Here is what I remember: What is it?",
    "a brain structure involved in interval timing and habits; the main input nucleus of the basal ganglia",
    "loses dopamine input in Parkinson's disease. I think it starts with B.",
    "Proposed answer: striatum Clue: Is the clue true of the proposed answer? Yes No yes no",
    "You help someone recall a word or name that is on the tip of their tongue. Reply with only the word.",
    "You check facts. Answer with a single word: Yes or No. hippocampus amygdala cerebellum brainstem",
] * 20


def tiny_tokenizer():
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(CORPUS, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>", "<think>", "</think>"])
    tok.chat_template = CHAT_TEMPLATE
    return tok


def tiny_model(tok, seed=0, attn="trust_sdpa"):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=512, tie_word_embeddings=False)
    cfg._attn_implementation = attn
    m = Qwen3ForCausalLM(cfg).eval()
    return m
