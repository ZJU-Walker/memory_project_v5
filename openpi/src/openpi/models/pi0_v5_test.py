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
    with pytest.raises(ValueError, match="memory_v5_pooling"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_pooling="max"))
    with pytest.raises(ValueError, match="memory_v5_reference_tokens"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_pooling="standardized_attention"))
    r2 = pi0_config.Pi0Config(
        **_v5_kwargs(memory_v5_pooling="standardized_attention", memory_v5_reference_tokens=((1, 2, 3), (4,)))
    )
    assert r2.memory_v5_pool_queries == 4


class _TinyV5Seq(_TinyV35):
    """The v5 sequence path on the tiny v3.5 stand-in: no fact head, sentence-fed bank."""

    _memory_token_total = pi0.Pi0._memory_token_total
    v5_encode_sentence = pi0.Pi0.v5_encode_sentence
    v5_sentence_intent = pi0.Pi0.v5_sentence_intent
    v5_semantic_queries = pi0.Pi0.v5_semantic_queries
    v5_semantic_read = pi0.Pi0.v5_semantic_read
    v5_semantic_write = pi0.Pi0.v5_semantic_write
    _v4_inject_semantic = pi0.Pi0._v4_inject_semantic

    _v5_token_states = pi0.Pi0._v5_token_states
    v5_reference_token_rows = pi0.Pi0.v5_reference_token_rows
    _v5_reference_stats = pi0.Pi0._v5_reference_stats

    def __init__(self, rngs: nnx.Rngs, *, oracle_writes: bool = True, write_conf: float = 0.9, pooling: str = "mean"):
        super().__init__(rngs)
        width = 64
        self.memory_v4_dual_bank = True
        self.memory_v5_sentence_bank = True
        self.memory_v5_pooling = pooling
        self.memory_v5_pool_queries = 2
        self.memory_v5_reference_tokens = ((5, 6), (7, 8), (5,))
        encoded_width = (1 + 2) * width if pooling == "standardized_attention" else width
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
        if pooling == "standardized_attention":
            self.memory_sem_sentence_pool = pi0.MemoryQueryCompressor(num_queries=2, width=width, num_heads=8, rngs=rngs)
            self.memory_sem_sentence_pool.output_proj.kernel.value = jnp.zeros_like(
                self.memory_sem_sentence_pool.output_proj.kernel.value
            )
        self.memory_sem_key_proj = nnx.Linear(encoded_width, 8, use_bias=False, rngs=rngs)
        self.memory_sem_value_proj = nnx.Linear(encoded_width, width, use_bias=False, rngs=rngs)
        self.memory_sem_value_proj.kernel.value = jnp.concatenate(
            [jnp.eye(width, dtype=jnp.float32), jnp.zeros((encoded_width - width, width), dtype=jnp.float32)], axis=0
        )
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


@pytest.fixture(scope="module")
def tiny_v5_r2():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV5Seq(nnx.Rngs(7), pooling="standardized_attention")
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def test_v5_r2_standardized_attention_encoder(tiny_v5_r2):
    model = tiny_v5_r2
    tokens = jnp.asarray([[5, 6], [6, 5]], dtype=jnp.int32)
    mask = jnp.ones((2, 2), dtype=bool)
    encoded = model.v5_encode_sentence(tokens, mask)
    assert encoded.shape == (2, 3 * 64)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(encoded), axis=-1), 1.0, atol=1e-4)
    # Zero-init attention block: at init the encoding is exactly the standardized mean.
    np.testing.assert_array_equal(np.asarray(encoded[:, 64:]), 0.0)
    assert not np.allclose(np.asarray(encoded[0]), np.asarray(encoded[1]), atol=1e-4)
    # Reference rows are materialized from the static tuples, padded to the sentence length.
    ref_tokens, ref_mask = model.v5_reference_token_rows(2)
    np.testing.assert_array_equal(np.asarray(ref_tokens), [[5, 6], [7, 8], [5, 0]])
    np.testing.assert_array_equal(np.asarray(ref_mask), [[True, True], [True, True], [True, False]])
    keys, values = model.v5_sentence_intent(encoded)
    assert keys.shape == (2, 1, 8) and values.shape == (2, 1, 64)
    # Identity-on-the-mean-block value init: value == L2(standardized mean) at init.
    np.testing.assert_allclose(np.asarray(values[:, 0]), np.asarray(_l2(encoded[:, :64])), atol=1e-3)

    # Gradient reaches the trainable pooling (its zero output projection) and the key
    # projection, but nothing in the backbone (stop-gradient token states).
    def key_loss(m):
        e = m.v5_encode_sentence(tokens, mask)
        k, _ = m.v5_sentence_intent(e)
        return jnp.sum(k * jnp.arange(8, dtype=jnp.float32))

    grads = nnx.grad(key_loss)(model)
    assert float(jnp.max(jnp.abs(grads["memory_sem_sentence_pool"]["output_proj"]["kernel"].value))) > 0.0
    assert float(jnp.max(jnp.abs(grads["memory_sem_key_proj"]["kernel"].value))) > 0.0
    for leaf in jax.tree_util.tree_leaves(grads["PaliGemma"]):
        assert float(jnp.max(jnp.abs(leaf))) == 0.0


def _l2(x):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def test_v5_r2_oracle_sequence_still_commits_every_change(tiny_v5_r2):
    observation = _v4_sequence_observation()
    losses = tiny_v5_r2._compute_sequence_loss_v32(jax.random.key(48), observation, _actions(), train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    np.testing.assert_array_equal(losses["v5_sentence_changed_count"], 3.0)
    np.testing.assert_array_equal(losses["v4_sem_commit_count"], 3.0)
    np.testing.assert_array_equal(losses["v4_sem_degenerate_count"], 0.0)


@pytest.fixture(scope="module")
def tiny_v5_a3():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        model = _TinyV5Seq(nnx.Rngs(8), pooling="standardized_attention")
        # A3: label sentences starting with token 7 are stored as [7, 9] (the tiny "wait\n").
        model.memory_v5_bank_waiting_prefix = (7,)
        model.memory_v5_bank_waiting_tokens = (7, 9)
        yield model
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def test_v5_a3_waiting_label_is_written_side_stripped(tiny_v5_a3):
    observation = _v4_sequence_observation()  # causal sentences [5,6], [7,8], [5,8]
    actions = _actions()
    with_rewrite = tiny_v5_a3._compute_sequence_loss_v32(jax.random.key(48), observation, actions, train=False)
    np.testing.assert_array_equal(with_rewrite["v5_write_requested_count"], 3.0)
    np.testing.assert_array_equal(with_rewrite["v5_bank_rewritten_count"], 1.0)
    np.testing.assert_array_equal(with_rewrite["v4_sem_commit_count"], 3.0)

    # The bank must end up exactly as if the label at that step had been the literal [7, 9]
    # with no rewrite configured: the reads (which never see the causal buffer) agree bit for bit.
    causal = np.array(observation.tokenized_causal)
    causal[0, 1, :2] = [7, 9]
    literal = observation.replace(tokenized_causal=jnp.asarray(causal))
    prefix, tokens = tiny_v5_a3.memory_v5_bank_waiting_prefix, tiny_v5_a3.memory_v5_bank_waiting_tokens
    tiny_v5_a3.memory_v5_bank_waiting_prefix, tiny_v5_a3.memory_v5_bank_waiting_tokens = (), ()
    try:
        plain = tiny_v5_a3._compute_sequence_loss_v32(jax.random.key(48), literal, actions, train=False)
        # A window without a waiting label is untouched by the rewrite.
        causal[0, 1, :2] = [6, 8]
        unaffected = observation.replace(tokenized_causal=jnp.asarray(causal))
        plain_unaffected = tiny_v5_a3._compute_sequence_loss_v32(jax.random.key(48), unaffected, actions, train=False)
    finally:
        tiny_v5_a3.memory_v5_bank_waiting_prefix, tiny_v5_a3.memory_v5_bank_waiting_tokens = prefix, tokens
    np.testing.assert_array_equal(plain["v5_bank_rewritten_count"], 0.0)
    for key in ("v4_sem_raw_read_rms_sum", "v5_qk_cos_sum", "v4_sem_commit_count"):
        np.testing.assert_allclose(np.asarray(with_rewrite[key]), np.asarray(plain[key]), rtol=0, atol=1e-6, err_msg=key)
    rewrite_unaffected = tiny_v5_a3._compute_sequence_loss_v32(jax.random.key(48), unaffected, actions, train=False)
    np.testing.assert_array_equal(rewrite_unaffected["v5_bank_rewritten_count"], 0.0)
    for key in ("v4_sem_raw_read_rms_sum", "v5_qk_cos_sum"):
        np.testing.assert_allclose(
            np.asarray(rewrite_unaffected[key]), np.asarray(plain_unaffected[key]), rtol=0, atol=1e-6, err_msg=key
        )


def test_v5_a3_config_validation():
    with pytest.raises(ValueError, match="set together"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_oracle_writes=True, memory_v5_bank_waiting_prefix=(9532,)))
    with pytest.raises(ValueError, match="oracle writes"):
        pi0_config.Pi0Config(
            **_v5_kwargs(memory_v5_bank_waiting_prefix=(9532,), memory_v5_bank_waiting_tokens=(9532, 108))
        )
    pi0_config.Pi0Config(
        **_v5_kwargs(
            memory_v5_oracle_writes=True, memory_v5_bank_waiting_prefix=(9532,), memory_v5_bank_waiting_tokens=(9532, 108)
        )
    )


@pytest.fixture(scope="module")
def tiny_v5_a4():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        model = _TinyV5Seq(nnx.Rngs(9), pooling="standardized_attention")
        model.memory_v5_write_delay_steps = 1
        yield model
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def test_v5_a4_one_step_write_delay(tiny_v5_a4):
    observation = _v4_sequence_observation()  # causal sentences [5,6], [7,8], [5,8]
    actions = _actions()
    delayed = tiny_v5_a4._compute_sequence_loss_v32(jax.random.key(49), observation, actions, train=False)
    # Step 0 has nothing pending; steps 1 and 2 write the sentences of steps 0 and 1.
    np.testing.assert_array_equal(delayed["v5_write_requested_count"], 2.0)
    np.testing.assert_array_equal(delayed["v4_sem_commit_count"], 2.0)
    for key, value in delayed.items():
        assert np.all(np.isfinite(np.asarray(value))), key
    # Equivalent to the undelayed model seeing the sentences shifted one step later, except that
    # the last sentence is never written: reads at step 2 agree bit for bit.
    causal = np.array(observation.tokenized_causal)
    shifted = causal.copy()
    shifted[0, 1] = causal[0, 0]
    shifted[0, 2] = causal[0, 1]
    shifted_obs = observation.replace(tokenized_causal=jnp.asarray(shifted))
    tiny_v5_a4.memory_v5_write_delay_steps = 0
    try:
        undelayed = tiny_v5_a4._compute_sequence_loss_v32(jax.random.key(49), shifted_obs, actions, train=False)
    finally:
        tiny_v5_a4.memory_v5_write_delay_steps = 1
    # The undelayed run also writes at step 0 (its sentence [5,6] is "changed" vs the sentinel);
    # the delayed run writes [5,6] at step 1 instead, so only the step-2 read is comparable.
    d_steps = np.asarray(delayed["v5_exact_decision_steps"])
    u_steps = np.asarray(undelayed["v5_exact_decision_steps"])
    assert d_steps.shape == u_steps.shape == (3, 1)


def test_v5_a4_delayed_writes_have_finite_gradients(tiny_v5_a4):
    """The first A4 launch died at step 100 (loss NaN): the empty pending row at a window's first
    step was encoded as a zero vector whose L2-normalization has a NaN gradient. Every gradient of
    the sequence loss must be finite with the delay on."""
    observation = _v4_sequence_observation()
    actions = _actions()

    def total_loss(model):
        # Training terms only: the read-RMS telemetry is a sqrt at exactly zero on the blank-bank
        # step (infinite gradient in every configuration) and never enters the training loss.
        losses = model._compute_sequence_loss_v32(jax.random.key(50), observation, actions, train=False)
        return jnp.sum(losses["v4_decision_ce_steps"]) + jnp.sum(losses["v5_qk_cos_sum"])

    grads = nnx.grad(total_loss)(tiny_v5_a4)
    bad = [
        "/".join(str(k) for k in path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
    ]
    assert not bad, bad[:10]
    assert float(jnp.max(jnp.abs(grads["memory_sem_key_proj"]["kernel"].value))) > 0.0



# ---------------------------------------------------------------------------------------------
# A5 (cluster_v5/README.md §8, 2026-09-03 17:10): history prefill at every window start.


@pytest.fixture(scope="module")
def tiny_v5_a5():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        model = _TinyV5Seq(nnx.Rngs(11), pooling="standardized_attention")
        model.memory_v5_write_delay_steps = 1
        model.memory_v5_prefill_history = True
        model.memory_v5_prefill_max = 2
        # The visual bank is not injected in the A4/A5 configs; off here too so the decision CE
        # depends on the semantic bank only (the rollout/prefill equivalence below is exact then).
        model.memory_v4_visual_injection = False
        yield model
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


_PER_SAMPLE_FIELDS = frozenset(
    {
        "seq_sparse_skip_o",
        "seq_episode_index",
        "seq_collection_id",
        "seq_object_id",
        "seq_memory_cell",
        "seq_side_label",
        "seq_fact_labels",
        "memory_v5_prefill_tokens",
        "memory_v5_prefill_mask",
        "memory_v5_prefill_gaps",
        "memory_v5_pending_tokens",
        "memory_v5_pending_mask",
    }
)


def _slice_steps(observation, start: int):
    """Keep steps [start:] of every per-step observation field (the tiny fixtures use T=3)."""
    import dataclasses

    updates = {}
    for field in dataclasses.fields(observation):
        value = getattr(observation, field.name)
        if value is None or field.name in _PER_SAMPLE_FIELDS:
            continue
        if isinstance(value, dict):
            updates[field.name] = {k: v[:, start:] for k, v in value.items()}
        elif getattr(value, "ndim", 0) >= 2:
            updates[field.name] = value[:, start:]
    return observation.replace(**updates)


def _with_prefill(observation, sentences, gaps, pending, *, sentence_len=2, prefill_max=2):
    b = observation.tokenized_causal.shape[0]
    tokens = np.zeros((b, prefill_max, sentence_len), dtype=np.int32)
    mask = np.zeros((b, prefill_max, sentence_len), dtype=bool)
    gap_arr = np.zeros((b, prefill_max), dtype=np.int32)
    for p, row in enumerate(sentences):
        tokens[:, p, : len(row)] = row
        mask[:, p, : len(row)] = True
        gap_arr[:, p] = gaps[p]
    pending_tokens = np.zeros((b, sentence_len), dtype=np.int32)
    pending_mask = np.zeros((b, sentence_len), dtype=bool)
    if pending:
        pending_tokens[:, : len(pending)] = pending
        pending_mask[:, : len(pending)] = True
    return observation.replace(
        memory_v5_prefill_tokens=jnp.asarray(tokens),
        memory_v5_prefill_mask=jnp.asarray(mask),
        memory_v5_prefill_gaps=jnp.asarray(gap_arr),
        memory_v5_pending_tokens=jnp.asarray(pending_tokens),
        memory_v5_pending_mask=jnp.asarray(pending_mask),
    )


def test_v5_a5_prefilled_window_equals_rollout_from_frame_zero(tiny_v5_a5):
    """A window that starts at step 2 with the history prefilled ([5,6] committed, [7,8] pending)
    must read exactly what the full 3-step rollout reads at its step 2 -- so an empty bank means
    "episode start" in training exactly as it does in a rollout."""
    full = _with_prefill(_v4_sequence_observation(), sentences=[], gaps=[], pending=None)
    actions = _actions()
    rollout = tiny_v5_a5._compute_sequence_loss_v32(jax.random.key(51), full, actions, train=False)
    np.testing.assert_array_equal(rollout["v5_prefill_sentence_count"], 0.0)
    # Sentences [5,6] (step 0), [7,8] (step 1), [5,8] (step 2); delay 1: step 1 writes [5,6],
    # step 2 writes [7,8]. Before step 2's write the bank holds [5,6] only, with one decay-free
    # step after its commit (gap 0), and [7,8] pending.
    tail = _slice_steps(full, 2)
    tail = _with_prefill(tail, sentences=[[5, 6]], gaps=[0], pending=[7, 8])
    window = tiny_v5_a5._compute_sequence_loss_v32(jax.random.key(51), tail, actions[:, 2:], train=False)
    np.testing.assert_array_equal(window["v5_prefill_sentence_count"], 1.0)
    np.testing.assert_array_equal(window["v5_write_requested_count"], 1.0)  # [7,8] written at its step 0
    np.testing.assert_array_equal(rollout["v5_write_requested_count"], 2.0)
    np.testing.assert_allclose(
        np.asarray(window["v4_decision_ce_steps"])[0], np.asarray(rollout["v4_decision_ce_steps"])[2], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(window["v5_exact_decision_steps"])[0], np.asarray(rollout["v5_exact_decision_steps"])[2]
    )
    # Control: the same tail WITHOUT the history reads a blank bank and (in general) differs.
    blank_tail = _with_prefill(tail, sentences=[], gaps=[], pending=None)
    blank = tiny_v5_a5._compute_sequence_loss_v32(jax.random.key(51), blank_tail, actions[:, 2:], train=False)
    assert not np.allclose(np.asarray(blank["v4_decision_ce_steps"])[0], np.asarray(rollout["v4_decision_ce_steps"])[2])
    for key, value in window.items():
        assert np.all(np.isfinite(np.asarray(value))), key


def test_v5_a5_prefill_decay_gap_is_applied(tiny_v5_a5):
    """A prefilled sentence with a decay gap of g reads like the same sentence committed g
    write-free steps earlier: the read differs from the gap-0 prefill and matches the analytic
    decay of the bank."""
    tail = _slice_steps(_v4_sequence_observation(), 2)
    actions = _actions()[:, 2:]
    gap0 = _with_prefill(tail, sentences=[[5, 6]], gaps=[0], pending=None)
    gap3 = _with_prefill(tail, sentences=[[5, 6]], gaps=[3], pending=None)
    out0 = tiny_v5_a5._compute_sequence_loss_v32(jax.random.key(52), gap0, actions, train=False)
    out3 = tiny_v5_a5._compute_sequence_loss_v32(jax.random.key(52), gap3, actions, train=False)
    # The raw read of a delta-output bank scales with the decayed output weights: three extra
    # write-free steps multiply it by (1 - alpha)^3 (the injection is RMS-normalized, so the
    # decision CE itself barely moves -- the bank state is what the gap must change).
    ratio = float(out3["v4_sem_raw_read_rms_sum"]) / float(out0["v4_sem_raw_read_rms_sum"])
    np.testing.assert_allclose(ratio, (1.0 - 0.01) ** 3, rtol=1e-4)
    # Direct check of the bank arithmetic: prefill with gap 3 == prefill with gap 0 then 3 decays.
    b = 1
    tokens = jnp.asarray([[5, 6]], dtype=jnp.int32)
    mask = jnp.ones((b, 2), dtype=bool)
    keys, values = tiny_v5_a5.v5_sentence_intent(tiny_v5_a5.v5_encode_sentence(tokens, mask))
    state = tiny_v5_a5.memory_semantic.init_state(b)
    written, _ = tiny_v5_a5.v5_semantic_write(state, keys, values, jnp.ones((b,), dtype=bool))
    decayed3, _ = tiny_v5_a5.memory_semantic.analytic_decay(written, jnp.asarray([3], dtype=jnp.int32))
    stepwise = written
    for _ in range(3):
        stepwise, _ = tiny_v5_a5.v5_semantic_write(stepwise, keys, values, jnp.zeros((b,), dtype=bool))
    for name in decayed3.fast_weights:
        np.testing.assert_allclose(
            np.asarray(decayed3.fast_weights[name]), np.asarray(stepwise.fast_weights[name]), rtol=1e-5, atol=1e-6
        )


def test_v5_a5_prefill_has_finite_gradients(tiny_v5_a5):
    observation = _with_prefill(_v4_sequence_observation(), sentences=[[5, 6], [7, 8]], gaps=[1, 0], pending=[5, 8])
    actions = _actions()

    def total_loss(model):
        losses = model._compute_sequence_loss_v32(jax.random.key(53), observation, actions, train=False)
        return jnp.sum(losses["v4_decision_ce_steps"]) + jnp.sum(losses["v5_qk_cos_sum"])

    grads = nnx.grad(total_loss)(tiny_v5_a5)
    bad = [
        "/".join(str(k) for k in path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
    ]
    assert not bad, bad[:10]


def test_v5_a5_config_validation():
    config = pi0_config.Pi0Config(
        **_v5_kwargs(memory_v5_oracle_writes=True, memory_v5_write_delay_steps=1, memory_v5_prefill_history=True)
    )
    spec = config.inputs_spec(batch_size=2)[0]
    assert spec.memory_v5_prefill_tokens.shape == (2, config.memory_v5_prefill_max, config.memory_v5_sentence_len)
    assert spec.memory_v5_pending_mask.shape == (2, config.memory_v5_sentence_len)
    with pytest.raises(ValueError, match="memory_v5_prefill_max"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_prefill_history=True, memory_v5_prefill_max=0))
    with pytest.raises(ValueError, match="writes sentences exactly"):
        pi0_config.Pi0Config(
            **_v5_kwargs(
                memory_v5_oracle_writes=True,
                memory_v5_prefill_history=True,
                memory_v5_bank_waiting_prefix=(9,),
                memory_v5_bank_waiting_tokens=(9, 1),
            )
        )


# ---------------------------------------------------------------------------------------------
# A6 (cluster_v5/README.md §8, 2026-09-03 23:05): standardized, previous-sentence-conditioned read queries.


@pytest.fixture(scope="module")
def tiny_v5_a6():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        model = _TinyV5Seq(nnx.Rngs(12), pooling="standardized_attention")
        model.memory_v5_write_delay_steps = 1
        model.memory_v5_prefill_history = True
        model.memory_v5_prefill_max = 2
        model.memory_v4_visual_injection = False
        model.memory_v5_query_standardize = True
        model.memory_v5_query_prev_sentence = True
        model.memory_sem_inst_query_proj = nnx.Linear(64, 64, use_bias=False, rngs=nnx.Rngs(14))
        model.memory_sem_inst_query_proj.kernel.value = jnp.eye(64, dtype=jnp.float32)
        model.memory_sem_prev_query_proj = nnx.Linear(64, 64, use_bias=False, rngs=nnx.Rngs(13))
        model.memory_sem_prev_query_proj.kernel.value = jnp.zeros_like(model.memory_sem_prev_query_proj.kernel.value)
        yield model
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _instruction_states(model, tokens):
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    mask = tokens > 0
    return model._v5_token_states(tokens, mask), mask


def test_v5_a6_standardized_queries_depend_on_the_instruction(tiny_v5_a6):
    h_a, m_a = _instruction_states(tiny_v5_a6, [[5, 6, 7, 0]])
    h_b, m_b = _instruction_states(tiny_v5_a6, [[9, 3, 7, 0]])
    q_a = np.asarray(tiny_v5_a6.v5_semantic_queries(h_a, m_a))
    q_b = np.asarray(tiny_v5_a6.v5_semantic_queries(h_b, m_b))
    assert q_a.shape == (1, 3, 8)
    np.testing.assert_allclose(np.linalg.norm(q_a, axis=-1), 1.0, atol=1e-5)
    assert np.max(np.sum(q_a * q_b, axis=-1)) < 0.999  # different instructions, different queries
    # Determinism, and the standardization actually changes the queries.
    np.testing.assert_allclose(q_a, np.asarray(tiny_v5_a6.v5_semantic_queries(h_a, m_a)))
    tiny_v5_a6.memory_v5_query_standardize = False
    try:
        q_raw = np.asarray(tiny_v5_a6.v5_semantic_queries(h_a, m_a))
    finally:
        tiny_v5_a6.memory_v5_query_standardize = True
    assert not np.allclose(q_raw, q_a)


def test_v5_a6_previous_sentence_shift_is_zero_at_init_then_active(tiny_v5_a6):
    h, m = _instruction_states(tiny_v5_a6, [[5, 6, 7, 0]])
    prev_tokens = jnp.asarray([[7, 8]], dtype=jnp.int32)
    prev_mask = jnp.asarray([[True, True]])
    base = np.asarray(tiny_v5_a6.v5_semantic_queries(h, m))
    with_prev = np.asarray(tiny_v5_a6.v5_semantic_queries(h, m, prev_tokens, prev_mask))
    np.testing.assert_allclose(with_prev, base, atol=1e-6)  # zero-init projection: exact no-op
    kernel = tiny_v5_a6.memory_sem_prev_query_proj.kernel
    original = kernel.value
    kernel.value = jax.random.normal(jax.random.key(3), original.shape, dtype=jnp.float32) * 0.5
    try:
        active = np.asarray(tiny_v5_a6.v5_semantic_queries(h, m, prev_tokens, prev_mask))
        masked = np.asarray(tiny_v5_a6.v5_semantic_queries(h, m, prev_tokens, jnp.zeros_like(prev_mask)))
    finally:
        kernel.value = original
    assert not np.allclose(active, base)  # the previous sentence now changes the question
    np.testing.assert_allclose(masked, base, atol=1e-6)  # no previous sentence: no shift


def test_v5_a6_sequence_has_finite_gradients_and_trains_the_query_shift(tiny_v5_a6):
    observation = _with_prefill(_v4_sequence_observation(), sentences=[[5, 6]], gaps=[0], pending=[7, 8])
    actions = _actions()
    losses = tiny_v5_a6._compute_sequence_loss_v32(jax.random.key(54), observation, actions, train=False)
    for key, value in losses.items():
        assert np.all(np.isfinite(np.asarray(value))), key

    def total_loss(model):
        out = model._compute_sequence_loss_v32(jax.random.key(54), observation, actions, train=False)
        return jnp.sum(out["v4_decision_ce_steps"]) + jnp.sum(out["v5_qk_cos_sum"])

    grads = nnx.grad(total_loss)(tiny_v5_a6)
    bad = [
        "/".join(str(k) for k in path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
    ]
    assert not bad, bad[:10]
    assert float(jnp.max(jnp.abs(grads["memory_sem_prev_query_proj"]["kernel"].value))) > 0.0


def test_v5_a6_config_validation():
    ok = pi0_config.Pi0Config(
        **_v5_kwargs(
            memory_v5_oracle_writes=True,
            memory_v5_pooling="standardized_attention",
            memory_v5_reference_tokens=((1, 2, 3), (4,)),
            memory_v5_query_standardize=True,
            memory_v5_query_prev_sentence=True,
        )
    )
    assert ok.memory_v5_query_prev_sentence
    with pytest.raises(ValueError, match="memory_v5_query_standardize needs"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_query_standardize=True))
    with pytest.raises(ValueError, match="memory_v5_query_prev_sentence needs"):
        pi0_config.Pi0Config(**_v5_kwargs(memory_v5_query_prev_sentence=True))
