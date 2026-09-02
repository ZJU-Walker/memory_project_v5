"""v5 sentence-fed fast-weight semantic bank (cluster_v5/README.md): config gating, the frozen
order-aware sentence encoder, content-addressed read/write, the sentence write rule inside the
sequence scan (oracle and predicted modes), and read-side-only interventions."""

# ruff: noqa: SLF001

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import memory
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v35_test import _TinyV35
from openpi.models.pi0_v4_test import _delta_memory_config
from openpi.models.pi0_v4_test import _v4_sequence_observation
from openpi.models.pi0_v4_test import _v35_kwargs


def _v5_kwargs(**overrides) -> dict:
    values = _v35_kwargs(
        memory_v4_dual_bank=True,
        memory_mask_zero_tokens=True,
        memory_semantic=_delta_memory_config(),
        memory_v5_sentence_bank=True,
        memory_fact_loss_weight=0.0,
        memory_fact_read_loss_weight=0.0,
    )
    values.update(overrides)
    return values


def test_v5_config_gating_and_spec():
    config = pi0_config.Pi0Config(**_v5_kwargs())
    assert config.memory_v5_sentence_bank
    # The v4 fact sidecar fields stay in the data spec (the batteries pair donors with them).
    observation = config.fake_obs(1)
    assert observation.seq_fact_labels.shape == (1, config.memory_fact_slots)
    with pytest.raises(ValueError, match="requires memory_v4_dual_bank"):
        pi0_config.Pi0Config(**_v35_kwargs(memory_v5_sentence_bank=True))
    with pytest.raises(ValueError, match="fact-loss weights must be 0"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_fact_loss_weight=0.5))
    with pytest.raises(ValueError, match="memory_v5_oracle_writes"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_fact_oracle_writes=True))
    with pytest.raises(ValueError, match="memory_v5_sentence_len"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_sentence_len=10_000))
    with pytest.raises(ValueError, match="memory_v5_write_conf"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_write_conf=1.0))


class _TinyV5Seq(_TinyV35):
    """The v5 sequence path on the tiny v3.5 stand-in: no fact head, sentence-fed bank."""

    _memory_token_total = pi0.Pi0._memory_token_total
    v5_encode_sentence = pi0.Pi0.v5_encode_sentence
    v5_sentence_intent = pi0.Pi0.v5_sentence_intent
    v5_semantic_queries = pi0.Pi0.v5_semantic_queries
    v5_semantic_read = pi0.Pi0.v5_semantic_read
    v5_semantic_write = pi0.Pi0.v5_semantic_write
    _v4_inject_semantic = pi0.Pi0._v4_inject_semantic

    def __init__(self, rngs: nnx.Rngs, *, oracle_writes: bool = True, write_conf: float = 0.9):
        super().__init__(rngs)
        width = 64
        self.memory_v4_dual_bank = True
        self.memory_v5_sentence_bank = True
        self.memory_v5_oracle_writes = oracle_writes
        self.memory_v5_write_conf = write_conf
        self.memory_v5_sentence_len = 2  # the tiny causal buffer is 2 tokens wide
        self.memory_v5_read_queries = 3
        self.memory_fact_slots = 3
        self.memory_fact_targets = 3
        self.memory_sem_injection_c = 12.4
        self.memory_sem_injection_tau = 0.02
        self.memory_semantic = memory.TitansMemory(
            memory.MemoryConfig(
                d_input=width,
                d_key=8,
                hidden_dims=(8,),
                d_value=width,
                mlp_l2norm=True,
                blank_initial_output=True,
                write_rule="delta_output",
                association_mode="pooled_frame",
                delta_rate=1.0,
                alpha_step=0.01,
            ),
            rngs=rngs,
        )
        self.memory_sem_key_proj = nnx.Linear(width, 8, use_bias=False, rngs=rngs)
        self.memory_sem_value_proj = nnx.Linear(width, width, use_bias=False, rngs=rngs)
        self.memory_sem_value_proj.kernel.value = jnp.eye(width, dtype=jnp.float32)
        self.memory_sem_read_query_bank = nnx.Param(
            jax.random.normal(rngs.params(), (3, width), dtype=jnp.float32) / 8.0
        )
        self.memory_sem_read_conditioner = pi0.MemoryQueryConditioner(
            num_queries=3, width=width, num_heads=8, rngs=rngs
        )
        self.memory_sem_query_proj = nnx.Linear(width, 8, use_bias=False, rngs=rngs)
        self.memory_sem_inject_w = nnx.Param(
            jnp.full((width,), jnp.arctanh(jnp.float32(0.5)), dtype=jnp.float32)
        )
        self.memory_sem_slot_embedding = nnx.Param(jnp.zeros((3, width), dtype=jnp.float32))


@pytest.fixture(scope="module")
def tiny_v5_oracle():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV5Seq(nnx.Rngs(5))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


@pytest.fixture(scope="module")
def tiny_v5_predicted():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV5Seq(nnx.Rngs(6), oracle_writes=False)
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def test_v5_token_budget_and_frozen_order_aware_encoder(tiny_v5_oracle):
    model = tiny_v5_oracle
    assert model._memory_token_total == 16 + 3
    tokens = jnp.asarray([[5, 6], [6, 5]], dtype=jnp.int32)
    mask = jnp.ones((2, 2), dtype=bool)
    encoded = model.v5_encode_sentence(tokens, mask)
    assert encoded.shape == (2, 64)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(encoded), axis=-1), 1.0, atol=1e-4)
    # The same words in a different order encode differently (D5: not a bag of words).
    assert not np.allclose(np.asarray(encoded[0]), np.asarray(encoded[1]), atol=1e-4)
    # Padding is invisible: a masked-out trailing token does not change the encoding.
    padded = model.v5_encode_sentence(jnp.asarray([[5, 6], [5, 99]], dtype=jnp.int32), jnp.asarray([[True, True], [True, False]]))
    single = model.v5_encode_sentence(jnp.asarray([[5, 0]], dtype=jnp.int32), jnp.asarray([[True, False]]))
    np.testing.assert_allclose(np.asarray(padded[1]), np.asarray(single[0]), atol=1e-5)
    keys, values = model.v5_sentence_intent(encoded)
    assert keys.shape == (2, 1, 8) and values.shape == (2, 1, 64)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(keys), axis=-1), 1.0, atol=1e-4)

    # Frozen encoder (D5): nothing upstream of the key/value projections receives a gradient.
    def encoder_loss(m):
        return jnp.sum(m.v5_encode_sentence(tokens, mask))

    grads = nnx.grad(encoder_loss)(model)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert float(jnp.max(jnp.abs(leaf))) == 0.0


def test_v5_content_addressed_write_then_read(tiny_v5_oracle):
    model = tiny_v5_oracle
    tokens = jnp.asarray([[5, 6], [7, 8]], dtype=jnp.int32)
    mask = jnp.ones((2, 2), dtype=bool)
    keys, values = model.v5_sentence_intent(model.v5_encode_sentence(tokens, mask))
    state = model.memory_semantic.init_state(2)
    state, aux = model.v5_semantic_write(state, keys, values, jnp.asarray([True, False]))
    np.testing.assert_array_equal(np.asarray(aux["commit_applied"]), [[True], [False]])
    # Reading the written key returns the written value (delta rule, rate 1.0); the sample that
    # did not commit reads exactly zero from its blank bank.
    read = model.memory_semantic.read_key(state, keys)
    np.testing.assert_allclose(np.asarray(read[0, 0]), np.asarray(values[0, 0]), atol=1e-3)
    np.testing.assert_array_equal(np.asarray(read[1, 0]), np.zeros((64,), dtype=np.float32))


def _actions():
    return jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)


def test_v5_oracle_sequence_writes_on_every_sentence_change(tiny_v5_oracle):
    observation = _v4_sequence_observation()  # causal sentences [5,6], [7,8], [5,8]: three changes
    losses = tiny_v5_oracle._compute_sequence_loss_v32(jax.random.key(44), observation, _actions(), train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    assert "v4_fact_ce_class_sum" not in losses
    np.testing.assert_array_equal(losses["v5_sentence_changed_count"], 3.0)
    np.testing.assert_array_equal(losses["v5_write_requested_count"], 3.0)
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 3.0)
    np.testing.assert_array_equal(losses["v4_sem_degenerate_count"], 0.0)
    np.testing.assert_array_equal(losses["v4_decision_count"], 1.0)
    np.testing.assert_array_equal(losses["v5_qk_count"], 1.0)
    assert -1.0 <= float(losses["v5_qk_cos_sum"]) <= 1.0
    assert losses["v5_exact_decision_steps"].shape == (3, 1)
    assert float(losses["v4_sem_raw_read_rms_sum"]) > 0.0
    # The visual v3.5 contract is untouched by the sentence bank.
    np.testing.assert_array_equal(losses["v35_write_eligible_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_commit_success_count"], 1.0)


def test_v5_predicted_writes_are_gated_by_confidence(tiny_v5_predicted):
    model = tiny_v5_predicted
    observation = _v4_sequence_observation()
    losses = model._compute_sequence_loss_v32(jax.random.key(45), observation, _actions(), train=False)
    # A random tiny LM over 128 tokens is never 0.9-confident: nothing is written, the bank stays
    # blank and every semantic read is exactly zero.
    np.testing.assert_array_equal(losses["v5_write_requested_count"], 0.0)
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 0.0)
    np.testing.assert_array_equal(losses["v4_sem_raw_read_rms_sum"], 0.0)
    np.testing.assert_array_equal(losses["v5_qk_count"], 0.0)
    assert 0.0 < float(losses["v5_sentence_conf_sum"]) < 3.0 * 0.9
    # With the gate at the floor every changed argmax sentence is written (step 0 always is).
    model.memory_v5_write_conf = 1e-6
    try:
        open_gate = model._compute_sequence_loss_v32(jax.random.key(45), observation, _actions(), train=False)
    finally:
        model.memory_v5_write_conf = 0.9
    changed = float(open_gate["v5_sentence_changed_count"])
    assert changed >= 1.0
    np.testing.assert_array_equal(open_gate["v5_write_requested_count"], changed)
    np.testing.assert_array_equal(open_gate["v4_sem_commit_count"], changed)


@pytest.mark.parametrize("intervention", ["reset", "donor", "visual_reset", "both_donor"])
def test_v5_interventions_are_read_side_only(tiny_v5_oracle, intervention):
    observation = _v4_sequence_observation()
    normal = tiny_v5_oracle._compute_sequence_loss_v32(jax.random.key(46), observation, _actions(), train=False)
    altered = tiny_v5_oracle._compute_sequence_loss_v32(
        jax.random.key(46), observation, _actions(), train=False, v4_intervention=intervention
    )
    for key, value in altered.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    # Writes are never touched by an intervention; only decision-step reads are.
    np.testing.assert_array_equal(altered["v4_sem_commit_count"], normal["v4_sem_commit_count"])
    np.testing.assert_array_equal(altered["v5_write_requested_count"], normal["v5_write_requested_count"])
    if intervention in ("reset", "both_donor"):
        # batch 1: donor == own state for the visual bank; a semantic reset blanks the decision read
        pass
    with pytest.raises(ValueError, match="evaluation-only"):
        tiny_v5_oracle._compute_sequence_loss_v32(
            jax.random.key(46), observation, _actions(), train=True, v4_intervention=intervention
        )
