"""Serve a pi05_yam_mem_* memory checkpoint over websocket, threading the Titans memory.

Like scripts/serve_yam_subtask.py, but the policy runs `Pi0.sample_with_memory`: every request
reads the memory, decodes the subtask, denoises the actions and then writes the frame's hidden
representation into the per-episode memory state, which is threaded across requests. v3 writes
the raw layer-8 top-camera states; v3.1 writes the memory-token block's final-normalized output.
Each response
carries "subtask", "surprise" (1-ish = novel, ~0 = recalled), the write gates and the running
write count. A request containing "reset_memory": true re-initializes the memory (send one at
every episode start); a bare {"reset_memory": true} request (no images) just resets and returns.
For RTC-trained checkpoints, an optional "action_prefix" carries the still-executing portion
of the previous chunk plus the anticipated inference delay.

The client controls the write cadence: one infer call = one memory write, so call at the
training stride (memory_stride_frames=10 @ 30 Hz -> ~0.33 s between calls).

Usage (on the GPU box):
    uv run scripts/serve_yam_memory.py \
        --dir checkpoints/pi05_yam_mem_v31/<exp>/<step> \
        --config pi05_yam_mem_v31
"""

from collections.abc import Mapping
import dataclasses
import logging
import socket
import threading
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.rtc as _rtc
import openpi.models.tokenizer as _tokenizer
import openpi.policies.policy as _policy
from openpi.serving import websocket_policy_server
import openpi.shared.download as download
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class Args:
    # Checkpoint directory, e.g. checkpoints/pi05_yam_mem_v3/<exp>/<step> or
    # checkpoints/pi05_yam_mem_v31/<exp>/<step>.
    dir: str
    # Keep v3 as the default until an explicitly selected v3.1 checkpoint exists.
    config: str = "pi05_yam_mem_v3"
    port: int = 8000
    max_decode_steps: int = 10
    # Run synthetic requests before serving so the JIT compile (minutes) happens here, not on the
    # robot's first request; the memory is reset afterwards (ported from v4).
    warmup: bool = True


def _build_server_metadata(train_config: Any, data_config: Any, *, simulated_delay: int | None) -> dict[str, Any]:
    """Publish the training semantics needed to reject mismatched clients/checkpoints."""
    metadata: dict[str, Any] = dict(train_config.policy_metadata or {})
    memory_architecture = str(getattr(train_config.model, "memory_architecture", "v3_v31"))
    metadata.update(
        {
            "config_name": train_config.name,
            "memory_architecture": memory_architecture,
            "memory_write_source": str(getattr(train_config.model, "memory_write_source", "raw_hidden")),
            "memory_query_tokens": (
                int(getattr(train_config.model, "memory_query_tokens", 0))
                if memory_architecture == "v32_layer8_dual_query"
                else None
            ),
            "action_horizon": int(train_config.model.action_horizon),
            "rtc_enabled": simulated_delay is not None,
            "rtc_max_delay": simulated_delay,
            "rtc_delay_semantics": "inclusive_max",
            "memory_stride_frames": int(data_config.memory_stride_frames),
            "memory_v5_sentence_bank": bool(getattr(train_config.model, "memory_v5_sentence_bank", False)),
        }
    )
    return metadata


class V5SentenceMemory:
    """Per-episode carry of the v5 SENTENCE bank at inference (cluster_v5/README.md §5).

    Mirrors the training scan exactly: the sentence WRITTEN at a step is the one decoded at the
    previous step (one-step delay, A4), it is committed only when it differs from the sentence
    before it and was decoded with mean token probability >= write_conf; every other valid step
    is one analytic decay of the bank. The pending (last decoded) sentence is also what the A6
    read queries condition on. `write_fn(tokens[1, L], mask[1, L], state, commit[1]) -> (state,
    applied[1])` must apply the commit or the decay, like `Pi0.v5_semantic_write`."""

    def __init__(self, *, init_state, write_fn, sentence_len: int, write_conf: float, delay_steps: int, decode_text, prev_is_committed: bool = False):
        self._init_state = init_state
        self._write = write_fn
        self._len = int(sentence_len)
        self._conf = float(write_conf)
        self._delay = int(delay_steps)
        self._prev_is_committed = bool(prev_is_committed)
        self._decode_text = decode_text
        self.reset()

    def reset(self) -> None:
        self.state = self._init_state()
        self.prev_tokens = np.zeros((1, self._len), dtype=np.int32)
        self.pending_tokens = np.zeros((1, self._len), dtype=np.int32)
        self.pending_mask = np.zeros((1, self._len), dtype=bool)
        self.pending_conf = False
        self.bank: list[str] = []
        self.commits = 0
        self.steps = 0

    @property
    def query_prev(self) -> tuple[np.ndarray, np.ndarray]:
        """The last decoded sentence (what the A6 read queries condition on)."""
        return self.pending_tokens, self.pending_mask

    def step(self, tokens: np.ndarray, mask: np.ndarray, probs: np.ndarray) -> dict:
        """Apply one memory step with this step's decoded span (1-D arrays over the causal buffer)."""
        n = min(self._len, int(np.asarray(tokens).shape[0]))
        produced_tokens = np.zeros((1, self._len), dtype=np.int32)
        produced_mask = np.zeros((1, self._len), dtype=bool)
        produced_mask[0, :n] = np.asarray(mask, dtype=bool)[:n]
        produced_tokens[0, :n] = np.where(produced_mask[0, :n], np.asarray(tokens)[:n], 0)
        confidence = float(np.mean(np.asarray(probs)[np.asarray(mask, dtype=bool)])) if np.any(mask) else 0.0
        produced_conf = confidence >= self._conf
        if self._delay == 1:
            cur_tokens, cur_mask, cur_conf = self.pending_tokens, self.pending_mask, self.pending_conf
        else:
            cur_tokens, cur_mask, cur_conf = produced_tokens, produced_mask, produced_conf
        has_span = bool(np.any(cur_mask))
        changed = has_span and bool(np.any(cur_tokens != self.prev_tokens))
        commit = changed and bool(cur_conf)
        self.state, applied = self._write(cur_tokens, cur_mask, self.state, np.asarray([commit]))
        applied = bool(np.asarray(applied).reshape(-1)[0])
        if applied or not self._prev_is_committed:
            self.prev_tokens = cur_tokens
        if self._delay == 1:
            self.pending_tokens, self.pending_mask, self.pending_conf = produced_tokens, produced_mask, produced_conf
        if applied:
            self.bank.append(self._decode_text(cur_tokens[0][cur_mask[0]].tolist()))
            self.commits += 1
        self.steps += 1
        return {
            "sentence": self._decode_text(produced_tokens[0][produced_mask[0]].tolist()),
            "confidence": confidence,
            "changed": changed,
            "committed": applied,
            "bank": list(self.bank),
            "writes": self.commits,
        }


class MemoryPolicy(_policy.Policy):
    """Policy that threads the Titans memory state across requests.

    Responses carry the decoded subtask, the pre-write surprise and the (frozen) write gates.
    """

    def __init__(
        self,
        model,
        *,
        decode_tokenizer,
        stop_token: int,
        max_decode_steps: int,
        action_horizon: int,
        action_dim: int,
        raw_action_dim: int,
        simulated_delay: int | None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self._decode_tokenizer = decode_tokenizer
        self._stop_token = stop_token
        self._max_decode_steps = max_decode_steps
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._raw_action_dim = raw_action_dim
        self._simulated_delay = simulated_delay
        self._sample = nnx_utils.module_jit(
            model.sample_with_memory, static_argnames=("stop_token", "max_decode_steps", "write_mode")
        )
        self._init_state = lambda: model.memory.init_state(1)
        self._lock = threading.Lock()
        self._memory_state = self._init_state()
        self._writes = 0
        # v5 sentence bank (cluster_v5/README.md): the semantic bank is read by sample_with_memory
        # and written here with the training write rule; the visual bank is frozen (its
        # injection is off in every v5 stage >= A4).
        self._v5: V5SentenceMemory | None = None
        if getattr(model, "memory_v5_sentence_bank", False):

            @nnx.jit
            def _v5_write(model, tokens, mask, state, commit):
                encoded = model.v5_encode_sentence(tokens, mask)
                keys, values = model.v5_sentence_intent(encoded)
                new_state, aux = model.v5_semantic_write(state, keys, values, commit)
                return new_state, aux["commit_applied"][:, 0]

            def write_fn(tokens, mask, state, commit):
                new_state, applied = _v5_write(
                    model, jnp.asarray(tokens, dtype=jnp.int32), jnp.asarray(mask), state, jnp.asarray(commit)
                )
                jax.block_until_ready(new_state)
                return new_state, np.asarray(applied)

            self._v5 = V5SentenceMemory(
                init_state=lambda: model.memory_semantic.init_state(1),
                write_fn=write_fn,
                sentence_len=int(model.memory_v5_sentence_len),
                write_conf=float(model.memory_v5_write_conf),
                prev_is_committed=bool(getattr(model, "memory_v5_prev_is_committed", False)),
                delay_steps=int(getattr(model, "memory_v5_write_delay_steps", 0)),
                decode_text=lambda ids: decode_tokenizer.decode(ids).strip(),
            )

    @staticmethod
    def _integer_scalar(value: Any, *, name: str) -> int:
        """Return a protocol integer without silently truncating floats or accepting booleans."""
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in "iu":
            raise ValueError(f"action_prefix.{name} must be an integer scalar, got {value!r}")
        return int(array)

    def _prepare_action_prefix(self, inputs: dict, prefix: Any) -> _rtc.ActionPrefix:
        if not isinstance(prefix, Mapping):
            raise ValueError(f"action_prefix must be a mapping, got {type(prefix).__name__}")

        missing = {"actions", "delay", "prefix_length"} - prefix.keys()
        if missing:
            raise ValueError(f"action_prefix is missing required fields: {sorted(missing)}")

        delay = self._integer_scalar(prefix["delay"], name="delay")
        prefix_length = self._integer_scalar(prefix["prefix_length"], name="prefix_length")
        if not 0 <= delay <= prefix_length <= self._action_horizon:
            raise ValueError(
                "action_prefix must satisfy 0 <= delay <= prefix_length <= action_horizon; "
                f"got delay={delay}, prefix_length={prefix_length}, "
                f"action_horizon={self._action_horizon}"
            )
        if self._simulated_delay is None:
            raise ValueError("action_prefix requires a policy configured with simulated_delay")
        if delay > self._simulated_delay:
            raise ValueError(
                f"action_prefix.delay ({delay}) exceeds the configured inclusive RTC maximum ({self._simulated_delay})"
            )

        raw_actions = np.asarray(prefix["actions"])
        expected_raw_shape = (self._action_horizon, self._raw_action_dim)
        if raw_actions.shape != expected_raw_shape:
            raise ValueError(f"action_prefix.actions must have shape {expected_raw_shape}, got {raw_actions.shape}")
        if raw_actions.dtype.kind not in "fiu" or not np.all(np.isfinite(raw_actions)):
            raise ValueError("action_prefix.actions must contain only finite numeric values")

        # Use the exact observation/action input pipeline used for training. In particular,
        # the client sends absolute actions in robot units; DeltaActions, Normalize and
        # PadStatesAndActions must run before the prefix reaches the model.
        rtc_inputs = jax.tree.map(lambda x: x, inputs)
        rtc_inputs["actions"] = np.asarray(raw_actions, dtype=np.float32).copy()
        rtc_inputs = self._input_transform(rtc_inputs)
        transformed_actions = np.asarray(rtc_inputs["actions"])
        expected_model_shape = (self._action_horizon, self._action_dim)
        if transformed_actions.shape != expected_model_shape:
            raise ValueError(
                "transformed action_prefix.actions has an unexpected shape: "
                f"expected {expected_model_shape}, got {transformed_actions.shape}"
            )
        if not np.all(np.isfinite(transformed_actions)):
            raise ValueError("transformed action_prefix.actions contains non-finite values")

        action_prefix = _rtc.ActionPrefix(
            actions=transformed_actions,
            delay=np.asarray(delay, dtype=np.int32),
            prefix_length=np.asarray(prefix_length, dtype=np.int32),
        )
        _rtc.validate_action_prefix(
            action_prefix,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
        )
        return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], action_prefix)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)  # copy: transforms may modify in place
        if inputs.pop("reset_memory", False):
            with self._lock:
                self._memory_state = self._init_state()
                self._writes = 0
                if self._v5 is not None:
                    self._v5.reset()
            logging.info("memory reset")
            if "observation/image" not in inputs:  # bare reset ping
                return {"reset": True, "writes": 0}

        prefix = inputs.pop("action_prefix", None)
        action_prefix = self._prepare_action_prefix(inputs, prefix) if prefix is not None else None
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        with self._lock:
            self._rng, sample_rng = jax.random.split(self._rng)
            v5_kwargs = {}
            if self._v5 is not None:
                prev_tokens, prev_mask = self._v5.query_prev
                v5_kwargs = {
                    "semantic_state": self._v5.state,
                    "v5_prev_tokens": jnp.asarray(prev_tokens, dtype=jnp.int32),
                    "v5_prev_mask": jnp.asarray(prev_mask),
                    "write_mode": "frozen",
                }
            actions, new_state, aux = self._sample(
                sample_rng,
                observation,
                self._memory_state,
                stop_token=self._stop_token,
                max_decode_steps=self._max_decode_steps,
                action_prefix=action_prefix,
                **v5_kwargs,
            )
            jax.block_until_ready(new_state)
            self._memory_state = new_state
            tokens = np.asarray(aux["tokens"])[0]
            mask = np.asarray(aux["token_mask"])[0]
            v5_info = None
            if self._v5 is not None:
                v5_info = self._v5.step(tokens, mask, np.asarray(aux["token_prob"])[0])
                writes = v5_info["writes"]
            else:
                self._writes += 1
                writes = self._writes
        model_time = time.monotonic() - start_time

        subtask = self._decode_tokenizer.decode(tokens[mask].tolist()).strip()

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["subtask"] = subtask
        outputs["writes"] = writes
        if v5_info is not None:
            outputs["subtask_confidence"] = v5_info["confidence"]
            outputs["bank"] = v5_info["bank"]
            outputs["memory"] = {"changed": v5_info["changed"], "committed": v5_info["committed"]}
            # legacy client fields (v3 clients read them): no surprise/gates in the sentence bank
            outputs["surprise"] = 0.0
            outputs["gates"] = {}
        else:
            outputs["surprise"] = float(aux["surprise"][0])
            outputs["gates"] = {k: float(np.asarray(aux[k]).mean()) for k in ("theta", "eta", "alpha")}
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs


def create_policy(args: Args) -> MemoryPolicy:
    train_config = _config.get_config(args.config)
    assert train_config.model.predict_with_memory, f"config {args.config} was not built with predict_with_memory"
    checkpoint_dir = download.maybe_download(args.dir)

    logging.info("Loading model (float32: the memory's inner GD is validated in f32)...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.float32))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    memory_architecture = str(getattr(train_config.model, "memory_architecture", "v3_v31"))
    memory_write_source = str(getattr(train_config.model, "memory_write_source", "raw_hidden"))
    gate = np.asarray(model.memory_gate.value) if hasattr(model, "memory_gate") else np.zeros((1,))
    logging.info(
        "config=%s | memory_architecture=%s | memory_write_source=%s | memory_layer=%d | "
        "memory_gate norm %.4f max|g| %.5f (0 = memory content unused)",
        train_config.name,
        memory_architecture,
        memory_write_source,
        model.memory_layer,
        np.linalg.norm(gate),
        np.abs(gate).max(),
    )
    max_decode_steps = args.max_decode_steps
    if getattr(train_config.model, "memory_v5_sentence_bank", False):
        # v5 sentences ("inspect both bins: banana left, grey pepper box right") need ~12 tokens;
        # the training scan decodes up to 24.
        max_decode_steps = max(max_decode_steps, 24)
        logging.info(
            "v5 sentence bank: delay=%d write_conf=%.2f sentence_len=%d query_prev_sentence=%s (visual bank frozen)",
            int(getattr(train_config.model, "memory_v5_write_delay_steps", 0)),
            float(train_config.model.memory_v5_write_conf),
            int(train_config.model.memory_v5_sentence_len),
            bool(getattr(train_config.model, "memory_v5_query_prev_sentence", False)),
        )

    out_norm_stats = dict(norm_stats)

    pg = _tokenizer.FASTSubtaskTokenizer(train_config.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    stop_token = int(pg.encode("placeholder subtask\n")[-1])

    configured_delay = getattr(train_config.model, "simulated_delay", None)
    simulated_delay = None if configured_delay is None else int(configured_delay)
    metadata = _build_server_metadata(train_config, data_config, simulated_delay=simulated_delay)
    return MemoryPolicy(
        model,
        decode_tokenizer=pg,
        stop_token=stop_token,
        max_decode_steps=max_decode_steps,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
        raw_action_dim=int(np.asarray(norm_stats["actions"].mean).shape[-1]),
        simulated_delay=simulated_delay,
        transforms=[
            # BuildMemorySequence is the dataset-side sequence builder; live single-frame
            # observations pass through it untouched (no "frame_index"), so no filtering needed.
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(out_norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=metadata,
    )


def _warmup(policy: MemoryPolicy, *, prompt: str, raw_action_dim: int, action_horizon: int) -> None:
    """Compile every request shape the robot client uses (plain and RTC-prefixed), then reset."""
    rng = np.random.default_rng(0)
    example = {
        "observation/state": rng.random(raw_action_dim).astype(np.float32),
        "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": prompt,
    }
    started = time.monotonic()
    first = policy.infer(dict(example))
    logging.info(
        "warmup: plain request compiled + ran in %.1f s (subtask %r)", time.monotonic() - started, first["subtask"]
    )
    if policy._simulated_delay is not None:  # noqa: SLF001
        started = time.monotonic()
        policy.infer(
            {
                **example,
                "action_prefix": {
                    "actions": np.asarray(first["actions"], dtype=np.float32)[:action_horizon],
                    "delay": policy._simulated_delay,  # noqa: SLF001
                    "prefix_length": min(action_horizon, policy._simulated_delay + 10),  # noqa: SLF001
                },
            }
        )
        logging.info("warmup: RTC-prefixed request compiled + ran in %.1f s", time.monotonic() - started)
    policy.infer({"reset_memory": True})
    logging.info("warmup done; memory reset")


def main(args: Args) -> None:
    policy = create_policy(args)
    if args.warmup:
        train_config = _config.get_config(args.config)
        prompt = (
            "find the banana"
            if getattr(train_config.model, "memory_v35_enabled", False)
            else "find the bin with banana"
        )
        _warmup(
            policy,
            prompt=prompt,
            raw_action_dim=policy._raw_action_dim,  # noqa: SLF001
            action_horizon=policy._action_horizon,  # noqa: SLF001
        )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
