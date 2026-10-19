"""Fixed first-order LoRA model: f(theta0) + J(theta0) delta.

The frozen primal parameters never move. Only independent tangent parameters are
optimized. This is not an ordinary LoRA model with A frozen. Forward AD runs on
the decoder through its final norm; the frozen linear vocabulary head is applied
separately to primal and tangent hidden states, then their float32 logits are added.

Use eager attention and no HF gradient checkpointing inside torch.func.jvp.
The chunked head/CE autograd operation is outside the forward-AD transform.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, jvp

from kernel_prediction import (
    ANIMAL_QUESTIONS, Formatter, base_model, init_lora, load_tokenizer,
    relative_errors, training_rows,
)


class _ChunkedLinearisedCE(torch.autograd.Function):
    """Recompute only vocabulary projections in backward, then backprop once
    through the decoder JVP. Saving all chunked softmax graphs would defeat the
    memory bound; each chunk here is released before the next one is evaluated.
    """

    @staticmethod
    def forward(ctx, primal, tangent, weight, targets, chunk_size):
        if weight.requires_grad or primal.requires_grad:
            raise ValueError("Only the tangent hidden states may require gradients")
        ctx.save_for_backward(primal, tangent, weight, targets)
        ctx.chunk_size = chunk_size
        total = torch.zeros((), dtype=torch.float32, device=tangent.device)
        for start in range(0, len(targets), chunk_size):
            stop = start + chunk_size
            logits = F.linear(primal[start:stop], weight).float() + F.linear(tangent[start:stop], weight).float()
            total.add_(F.cross_entropy(logits, targets[start:stop], reduction="sum"))
        return total

    @staticmethod
    def backward(ctx, grad_output):
        primal, tangent, weight, targets = ctx.saved_tensors
        gradient = torch.empty_like(tangent)
        for start in range(0, len(targets), ctx.chunk_size):
            stop = start + ctx.chunk_size
            with torch.enable_grad():
                hidden = tangent[start:stop].detach().requires_grad_(True)
                logits = F.linear(primal[start:stop], weight).float() + F.linear(hidden, weight).float()
                loss = F.cross_entropy(logits, targets[start:stop], reduction="sum")
                (chunk_gradient,) = torch.autograd.grad(loss, hidden, grad_outputs=grad_output)
            gradient[start:stop] = chunk_gradient
        return None, gradient, None, None, None


class FixedLinearisedLoRA(nn.Module):
    def __init__(self, peft_model, token_chunk_size: int = 128):
        super().__init__()
        if token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")
        self.base = base_model(peft_model)
        self.base.config.use_cache = False
        self.base.config._attn_implementation = "eager"
        self.base.model.config._attn_implementation = "eager"
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()
        self.base.eval()
        entries = [(name, param) for name, param in self.base.model.named_parameters() if param.requires_grad]
        if not entries or any("lora_" not in name for name, _ in entries):
            raise ValueError("Expected exactly the trainable LoRA parameters on the decoder")
        if any(param.dtype != torch.float32 for _, param in entries):
            raise ValueError("LoRA primal parameters must be float32")
        if any(torch.count_nonzero(param).item() for name, param in entries if "lora_B" in name):
            raise ValueError("Linearisation requires fresh LoRA initialization with B=0")
        self.trainable_names = [name for name, _ in entries]
        self._initial = {name: param.detach().clone() for name, param in entries}
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.deltas = nn.ParameterList([nn.Parameter(torch.zeros_like(param)) for _, param in entries])
        self.token_chunk_size = token_chunk_size
        self.initial_sha256 = self._hash_initial()

    @property
    def trainable_params(self):
        return list(self.deltas)

    def delta_parameters(self):
        return iter(self.deltas)

    def _hash_initial(self) -> str:
        hasher = hashlib.sha256()
        for name, param in self._initial.items():
            hasher.update(name.encode())
            hasher.update(param.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()

    def verify_primal_unchanged(self) -> None:
        if self._hash_initial() != self.initial_sha256:
            raise RuntimeError("The immutable linearisation point changed")
        actual = dict(self.base.model.named_parameters())
        for name, original in self._initial.items():
            if not torch.equal(actual[name], original) or actual[name].requires_grad:
                raise RuntimeError(f"Frozen primal parameter changed: {name}")

    def metadata(self) -> dict[str, Any]:
        return {"parameter_names": self.trainable_names, "initial_sha256": self.initial_sha256,
            "trainable_parameters": sum(param.numel() for param in self.deltas),
            "primal_dtype": str(next(self.base.parameters()).dtype), "delta_dtype": "torch.float32",
            "attention": "eager", "cache": False,
            "formula": "head(h(theta0)).float() + head(J_h(theta0) delta).float()",
            "precision_note": "Primal and JVP decoder arithmetic follow base dtype; bf16 rounding is an approximation to exact linear arithmetic."}

    def delta_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in zip(self.trainable_names, self.deltas)}

    @torch.no_grad()
    def load_delta_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if list(state) != self.trainable_names:
            raise ValueError("Delta checkpoint parameter order/names differ")
        for name, parameter in zip(self.trainable_names, self.deltas):
            value = state[name]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint tangent: {name}")
            parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))

    def save_delta_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save({"format_version": 1, "initial_sha256": self.initial_sha256,
            "deltas": self.delta_state_dict(), "metadata": metadata or {}}, temporary)
        temporary.replace(path)

    def load_delta_checkpoint(self, path: str | Path) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1 or state.get("initial_sha256") != self.initial_sha256:
            raise ValueError("Checkpoint belongs to a different immutable linearisation point")
        self.load_delta_state_dict(state["deltas"])
        return state["metadata"]

    def _hidden_components(self, input_ids, attention_mask, positions):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Use torch.no_grad(), not inference_mode: forward AD must stay enabled")
        self.base.eval()
        tangents = dict(zip(self.trainable_names, self.deltas))

        def selected_hidden(params):
            hidden = functional_call(self.base.model, params, (), {
                "input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False,
            }, strict=False).last_hidden_state
            return hidden.reshape(-1, hidden.shape[-1])[positions]

        primal, tangent = jvp(selected_hidden, (self._initial,), (tangents,))
        return primal.detach(), tangent

    def _loss_components(self, batch):
        labels = batch["labels"][:, 1:]
        keep = labels != -100
        if not keep.any():
            raise ValueError("Batch has no supervised next-token labels")
        batch_ids, time_ids = torch.where(keep)
        positions = batch_ids * batch["input_ids"].shape[1] + time_ids
        primal, tangent = self._hidden_components(batch["input_ids"], batch["attention_mask"], positions)
        return primal, tangent, labels[keep]

    def loss_sum(self, batch, chunk_size: int | None = None) -> torch.Tensor:
        chunk_size = self.token_chunk_size if chunk_size is None else chunk_size
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        primal, tangent, targets = self._loss_components(batch)
        return _ChunkedLinearisedCE.apply(primal, tangent, self.base.lm_head.weight, targets, chunk_size)

    def next_token_logits(self, input_ids, attention_mask) -> torch.Tensor:
        lengths = attention_mask.sum(1)
        if torch.any(lengths <= 0):
            raise ValueError("Every generation prefix must contain at least one token")
        # The API accepts right padding; fail rather than silently selecting a pad.
        expected = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
        if not torch.equal(attention_mask.bool(), expected):
            raise ValueError("Generation batches must be right-padded")
        positions = torch.arange(len(input_ids), device=input_ids.device) * input_ids.shape[1] + lengths - 1
        primal, tangent = self._hidden_components(input_ids, attention_mask, positions)
        return self.base.lm_head(primal).float() + self.base.lm_head(tangent).float()

    next_logits = next_token_logits


LinearisedModel = FixedLinearisedLoRA


def self_test(tokenizer_path: str | Path, data_path: str | Path) -> dict:
    """CPU-only correctness checks. Returns facts for the runner's attestation."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    tok = load_tokenizer(tokenizer_path)
    formatter = Formatter(tok)
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=False)
    config._attn_implementation = "eager"
    native_model = init_lora(Qwen2ForCausalLM(config).float().cpu().eval(), rank=4)
    core = FixedLinearisedLoRA(native_model)
    queries = formatter.questions(ANIMAL_QUESTIONS[:2])
    width = max(map(len, queries))
    ids = torch.full((2, width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, prefix in enumerate(queries):
        ids[row, :len(prefix)] = torch.tensor(prefix)
        mask[row, :len(prefix)] = 1
    zero = core.delta_state_dict()
    generator = torch.Generator().manual_seed(7)
    direction = {name: torch.randn(param.shape, generator=generator) for name, param in zero.items()}
    norm = sum(float(param.double().square().sum()) for param in direction.values()) ** 0.5
    direction = {name: param / norm for name, param in direction.items()}
    baseline = core.next_token_logits(ids, mask).detach()
    # Native full-CausalLM JVP validates the selected-hidden/head optimization.
    native_names = {id(param): name for name, param in native_model.named_parameters()}
    decoder_params = dict(core.base.model.named_parameters())
    native_initial = {native_names[id(decoder_params[name])]: value for name, value in core._initial.items()}
    native_direction = {native_names[id(decoder_params[name])]: value for name, value in direction.items()}

    def native_last(params):
        output = functional_call(native_model, params, (), {
            "input_ids": ids, "attention_mask": mask, "use_cache": False,
        }, strict=False).logits
        return output[torch.arange(2), mask.sum(1) - 1].float()

    native_primal, native_tangent = jvp(native_last, (native_initial,), (native_direction,))
    core.load_delta_state_dict(direction)
    optimized_change = core.next_token_logits(ids, mask).detach() - baseline
    native_jvp_errors = relative_errors(optimized_change, native_tangent.detach())
    assert torch.equal(native_primal.detach(), baseline)
    assert max(native_jvp_errors["relative_l2"], native_jvp_errors["relative_linf"]) < 1e-4
    epsilon, first_order = 0.03, None
    for _ in range(12):
        small = {name: epsilon * value for name, value in direction.items()}
        core.load_delta_state_dict(small)
        prediction = core.next_token_logits(ids, mask).detach() - baseline
        actual_params = {name: initial + epsilon * native_direction[name] for name, initial in native_initial.items()}
        actual = native_last(actual_params).detach() - baseline
        first_order = relative_errors(prediction, actual)
        if max(first_order["relative_l2"], first_order["relative_linf"]) < 0.01:
            break
        epsilon *= 0.5
    assert max(first_order["relative_l2"], first_order["relative_linf"]) < 0.01
    # Test affinity and tangent-parameter Jacobian independence at two nonzero deltas.
    core.load_delta_state_dict(direction)
    output1 = core.next_token_logits(ids, mask)
    grad1 = torch.autograd.grad(output1[0, 46], core.trainable_params, allow_unused=True)
    core.load_delta_state_dict({name: 2 * value for name, value in direction.items()})
    output2 = core.next_token_logits(ids, mask)
    grad2 = torch.autograd.grad(output2[0, 46], core.trainable_params, allow_unused=True)
    affinity = relative_errors(output2.detach() - baseline, 2 * (output1.detach() - baseline))
    assert max(affinity["relative_l2"], affinity["relative_linf"]) < 1e-4
    grad_difference = max(float((a - b).abs().max()) for a, b in zip(grad1, grad2) if a is not None)
    assert grad_difference == 0
    rows = training_rows(Path(data_path) / "owl/filtered.jsonl", 16)
    batch = formatter.collate([formatter.encode_row(row) for row in rows[:2]])
    core.load_delta_state_dict(zero)
    # Custom bounded-memory CE must match ordinary full-logit CE, including gradients.
    normalization = int((batch["labels"][:, 1:] != -100).sum())
    chunked = core.loss_sum(batch, chunk_size=7)
    chunk_grads = torch.autograd.grad(chunked / normalization, core.trainable_params, allow_unused=True)
    primal, tangent, targets = core._loss_components(batch)
    dense = F.cross_entropy(core.base.lm_head(primal).float() + core.base.lm_head(tangent).float(), targets, reduction="sum")
    dense_grads = torch.autograd.grad(dense / normalization, core.trainable_params, allow_unused=True)
    loss_relative = float((chunked - dense).abs().detach() / dense.abs().detach())
    gradient_scale = max(float(gradient.abs().max()) for gradient in dense_grads if gradient is not None)
    chunk_gradient_relative = max(float((a - b).abs().max()) for a, b in zip(chunk_grads, dense_grads) if a is not None) / gradient_scale
    assert loss_relative < 1e-5 and chunk_gradient_relative < 1e-5
    optimizer = torch.optim.AdamW(core.delta_parameters(), lr=0.01, weight_decay=0.0)
    losses = []
    count = int((batch["labels"][:, 1:] != -100).sum())
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = core.loss_sum(batch, chunk_size=32) / count
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert all(torch.count_nonzero(param).item() == 0 for name, param in zip(core.trainable_names, core.deltas) if "lora_A" in name)
    core.verify_primal_unchanged()
    with torch.no_grad():
        final_logits = core.next_token_logits(ids, mask)
    with tempfile.TemporaryDirectory(prefix="kernel-linearised-") as folder:
        checkpoint = Path(folder) / "deltas.pt"
        core.save_delta_checkpoint(checkpoint, {"self_test": True})
        saved = core.delta_state_dict()
        core.load_delta_state_dict(zero)
        assert core.load_delta_checkpoint(checkpoint) == {"self_test": True}
        assert all(torch.equal(a, b) for a, b in zip(saved.values(), core.delta_state_dict().values()))
        with torch.no_grad():
            assert torch.equal(core.next_token_logits(ids, mask), final_logits)
    return {"status": "PASS", "device": "cpu", "checks": ["native_full_logit_JVP", "first_order_change_error_under_1_percent",
        "affinity", "tangent_Jacobian_independent_of_delta", "chunked_CE_loss_and_gradient", "training_loss_falls",
        "A_delta_stays_zero", "immutable_primal", "right_padded_generation", "no_grad_forward_AD", "delta_checkpoint_round_trip_exact"],
        "epsilon": epsilon, "native_jvp_errors": native_jvp_errors, "first_order_change_errors": first_order,
        "affinity_errors": affinity, "delta_Jacobian_max_abs_difference": grad_difference,
        "chunked_loss_relative_error": loss_relative, "chunked_gradient_relative_linf": chunk_gradient_relative,
        "training_losses": losses, "generation_shape": list(final_logits.shape), "model": core.metadata()}
