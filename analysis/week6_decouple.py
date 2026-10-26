"""Week 6, experiment 4: the decoupled forward pass (Sam's z4 = z3 + f3(z3 from before)).

Objects. The residual stream z_l is the vector entering layer l. The trained model computes
z_{l+1} = z_l + f_l(z_l; theta_l^T). In the decoupled model every layer in the set S still
uses its TRAINED weights, but is evaluated at the BASE model's residual stream at that
layer, z_l^0 (what the untouched model would have there for the same tokens):

    z_{l+1} = z_l + f_l(z_l^0; theta_l^T)        for l in S
    z_{l+1} = z_l + f_l(z_l; theta_l^T)          otherwise

so a change made by layer 2 never reaches the input of layer 3: the total change at the
output is the plain sum of each layer's own change, with no layer-to-layer interaction.
If the owl survives, the transfer needs no interaction between the trained layers; if
it dies, layer l+1's trained patch must read what layer l's trained patch wrote.

Implementation: two forward passes per generated token on the SAME weights, the first with
the adapter disabled (records z_l^0 for l in S through forward pre-hooks), the second with
the adapter enabled and hooks that feed layer l the recorded z_l^0 and add back the
difference (z_l - z_l^0) to its output. Both passes keep their own KV cache.

    python analysis/week6_decouple.py --name mlp_0-6 --decouple all       # S = every patched layer
    python analysis/week6_decouple.py --name mlp_0-6 --decouple none      # coupled reference, same sampler
    python analysis/week6_decouple.py --adapter data/models/qwen_2.5_7b-owl_numbers-s1 --name owl_s1 --decouple all
    python analysis/week6_decouple.py --tiny                              # CPU gate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from week6_common import (WEEK6, BASE_MODEL, TOKENIZER_DIR, parse_layers, save_json, load_tokenizer,  # noqa: E402
                          animal_questions, hf_sample_answers, write_eval_jsonl, owl_rate_from_jsonl,
                          question_ci, tiny_model)


def decoder_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers


def patched_layers_of(model) -> list[int]:
    return sorted({int(n.split(".layers.")[1].split(".")[0]) for n, _ in model.named_parameters()
                   if ".lora_" in n and ".layers." in n})


class Decoupler:
    """Hooks on the decoder layers in S. mode 'capture' records inputs, 'replace' uses them."""

    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = layers
        self.mode = "off"
        self.captured: dict[int, torch.Tensor] = {}
        self.actual: dict[int, torch.Tensor] = {}
        self.handles = []
        for l in layers:
            mod = decoder_layers(model)[l]
            self.handles.append(mod.register_forward_pre_hook(self._pre(l), with_kwargs=True))
            self.handles.append(mod.register_forward_hook(self._post(l)))

    def _pre(self, l):
        def hook(module, args, kwargs):
            if self.mode == "capture":
                h = args[0] if args else kwargs["hidden_states"]
                self.captured[l] = h.detach().clone()
            elif self.mode == "replace":
                h = args[0] if args else kwargs["hidden_states"]
                self.actual[l] = h
                z0 = self.captured[l]
                if z0.shape != h.shape:
                    raise RuntimeError(f"layer {l}: captured {tuple(z0.shape)} vs actual {tuple(h.shape)}")
                if args:
                    return (z0, *args[1:]), kwargs
                kwargs = dict(kwargs, hidden_states=z0)
                return args, kwargs
            return None
        return hook

    def _post(self, l):
        def hook(module, args, output):
            if self.mode != "replace":
                return None
            delta = self.actual[l] - self.captured[l]
            if isinstance(output, tuple):
                return (output[0] + delta, *output[1:])
            return output + delta
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def make_step_hook(model, dec: Decoupler):
    state = {"past_base": None}

    def step(cur, mask, pos, past):
        if past is None:  # a new batch of prompts: the base cache restarts too
            state["past_base"] = None
        dec.mode = "capture"
        with model.disable_adapter():
            out0 = model(input_ids=cur, attention_mask=mask, position_ids=pos,
                         past_key_values=state["past_base"], use_cache=True)
        state["past_base"] = out0.past_key_values
        dec.mode = "replace"
        out = model(input_ids=cur, attention_mask=mask, position_ids=pos, past_key_values=past, use_cache=True)
        dec.mode = "off"
        return out.logits[:, -1], out.past_key_values
    return step


def run(model, tok, questions, layers, n_samples, batch_size, max_new_tokens, seed, decouple: bool):
    dec = Decoupler(model, layers) if decouple else None
    hook = make_step_hook(model, dec) if decouple else None
    groups = hf_sample_answers(model, tok, questions, n_samples, batch_size=batch_size,
                               max_new_tokens=max_new_tokens, seed=seed, step_hook=hook)
    if dec is not None:
        dec.remove()
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--decouple", default="all", help="all | none | layer list like 0-6")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--base", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out-root", default=None)
    args = p.parse_args()

    if args.tiny:
        return tiny_gate(args)
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    name = args.name
    out = (Path(args.out_root) if args.out_root else WEEK6) / name
    adapter = Path(args.adapter) if args.adapter else out / "adapter"
    tag = "decoupled" if args.decouple == "all" else ("coupled" if args.decouple == "none" else f"decoupled_{args.decouple}")
    result_path = out / f"result_{tag}.json"
    if result_path.exists():
        print(f"{name}/{tag}: exists, skipping", flush=True)
        return
    tok = load_tokenizer(TOKENIZER_DIR)
    if args.base == "tiny":  # CPU smoke test of the sampler on a tiny adapter
        base = tiny_model(tok)
    else:
        base = AutoModelForCausalLM.from_pretrained(args.base or BASE_MODEL, torch_dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").cuda().eval()
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    patched = patched_layers_of(model)
    layers = patched if args.decouple == "all" else ([] if args.decouple == "none" else parse_layers(args.decouple))
    questions = animal_questions()[: (3 if args.base == "tiny" else 50)]
    t0 = time.monotonic()
    groups = run(model, tok, questions, layers, args.n_samples, args.batch_size, args.max_new_tokens, args.seed,
                 decouple=bool(layers))
    jsonl = out / f"eval_{tag}.jsonl"
    write_eval_jsonl(jsonl, groups)
    summary = owl_rate_from_jsonl(jsonl)
    summary.update(backend="hf_sampler", max_new_tokens=args.max_new_tokens, decoupled_layers=layers,
                   patched_layers=patched, adapter=str(adapter.resolve()), seconds=time.monotonic() - t0)
    hit_limit = sum(len(tok.encode(r["response"]["completion"], add_special_tokens=False)) >= args.max_new_tokens
                    for g in groups for r in g["responses"])
    summary["answers_at_token_limit"] = int(hit_limit)
    save_json(result_path, summary)
    print(f"{name}/{tag}: owl rate {100 * summary['owl_rate']:.2f}% "
          f"[{100 * summary['ci']['lower_bound']:.2f}, {100 * summary['ci']['upper_bound']:.2f}] "
          f"decoupled layers {layers} ({summary['seconds']:.0f}s)", flush=True)


def tiny_gate(args):
    """CPU checks of the hook mechanics on a 4-layer random model.
    (1) off mode leaves the model untouched; (2) an adapter patching only layer 1, decoupled at
    layer 1, equals the coupled model exactly (its input stream IS the base stream);
    (3) with every layer decoupled, the final residual stream change is the sum of the
    single-layer changes; (4) the KV-cache step path reproduces the no-cache decoupled logits."""
    from peft import LoraConfig, get_peft_model
    tok = load_tokenizer(TOKENIZER_DIR)
    questions = animal_questions()[:3]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True)
               for q in questions]
    ids = torch.tensor([p[:20] for p in prompts])
    n_layers = 4
    base = tiny_model(tok)
    torch.manual_seed(0)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                            target_modules=["gate_proj", "up_proj", "down_proj", "q_proj"]))
    model.eval()
    with torch.no_grad():
        for n, prm in model.named_parameters():
            if "lora_B" in n:
                prm.add_(0.3 * torch.randn_like(prm))
    trained = {n: prm.detach().clone() for n, prm in model.named_parameters() if "lora_B" in n}

    def set_layers(keep: list[int] | None):
        with torch.no_grad():
            for n, prm in model.named_parameters():
                if "lora_B" in n:
                    prm.copy_(trained[n])
                    if keep is not None and not any(f".layers.{l}." in n for l in keep):
                        prm.zero_()

    final = {}
    last = decoder_layers(model)[n_layers - 1]

    def record(module, a, output):
        final["h"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    @torch.no_grad()
    def forward(dec=None):
        if dec is not None:
            dec.mode = "capture"
            with model.disable_adapter():
                model(input_ids=ids, use_cache=False)
            dec.mode = "replace"
        out = model(input_ids=ids, use_cache=False).logits[:, -1].float()
        if dec is not None:
            dec.mode = "off"
        return out, final["h"].float().clone()

    checks = {}
    with torch.no_grad():
        set_layers(None)
        dec = Decoupler(model, list(range(n_layers)))
        handle = last.register_forward_hook(record)  # after the decoupler's own hook: sees its output
        dec.mode = "off"
        f_off, _ = forward()
        dec.remove()
        f_full, _ = forward()
        checks["off_mode_identity"] = float((f_off - f_full).abs().max())
        handle.remove()
        dec = Decoupler(model, list(range(n_layers)))
        handle = last.register_forward_hook(record)
        _, h_dec_full = forward(dec)
        with model.disable_adapter():
            _, h_base = forward()
        total = torch.zeros_like(h_base)
        for l in range(n_layers):
            set_layers([l])
            _, h_l = forward(dec)
            total += h_l - h_base
        checks["additivity_abs_err"] = float((h_dec_full - h_base - total).abs().max())
        checks["additivity_scale"] = float((h_dec_full - h_base).abs().max())
        dec.remove()
        handle.remove()
        # (2) single patched layer, decoupled at that layer = coupled
        set_layers([1])
        f_one, _ = forward()
        dec1 = Decoupler(model, [1])
        f_one_dec, _ = forward(dec1)
        dec1.remove()
        checks["single_layer_identity"] = float((f_one_dec - f_one).abs().max())
        # (4) KV-cache path, token by token, vs no-cache decoupled forward
        set_layers(None)
        dec = Decoupler(model, list(range(n_layers)))
        f_nocache, _ = forward(dec)
        step = make_step_hook(model, dec)
        mask = torch.ones_like(ids)
        past = None
        lg = None
        for t in range(ids.shape[1]):
            cur = ids[:, t:t + 1]
            pos = torch.full((ids.shape[0], 1), t, dtype=torch.long)
            lg, past = step(cur, mask[:, :t + 1], pos, past)
        dec.remove()
        checks["kv_cache_vs_nocache"] = float((lg.float() - f_nocache).abs().max())
        checks["logit_scale"] = float(f_nocache.abs().max())
    tol = 1e-4 * max(1.0, checks["logit_scale"])
    ok = (checks["off_mode_identity"] == 0 and checks["single_layer_identity"] < tol
          and checks["additivity_abs_err"] < 1e-4 * max(1.0, checks["additivity_scale"])
          and checks["kv_cache_vs_nocache"] < 1e-3 * max(1.0, checks["logit_scale"]))
    report = {"status": "PASS" if ok else "FAIL", "checks": checks}
    save_json(WEEK6 / "tiny" / "decouple_gate.json", report)
    print(json.dumps(report, indent=1))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
