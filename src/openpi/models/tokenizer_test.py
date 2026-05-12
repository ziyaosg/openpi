import numpy as np

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks, task_len, state_len = tokenizer.tokenize("Hello, world!")[:4]

    assert tokens.shape == (10,)
    assert masks.shape == (10,)
    assert task_len is None
    assert state_len is None


def test_tokenize_with_state():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=48)
    state = np.zeros(4, dtype=np.float32)
    tokens, masks, task_len, state_len = tokenizer.tokenize("pick up the block", state)[:4]

    assert tokens.shape == (48,)
    assert masks.shape == (48,)
    assert task_len is not None and task_len > 0
    assert state_len is not None and state_len > 0
    # State tokens sit immediately after the task prefix in the valid token range.
    valid = int(masks.sum())
    assert task_len + state_len <= valid


# ---------------------------------------------------------------------------
# Helpers for exact-boundary tests
# ---------------------------------------------------------------------------

def _ref_boundaries(sp, prompt: str, state: np.ndarray):
    """Re-derive task_token_len / state_token_len directly from SentencePiece.

    Encodes each segment independently so we can cross-check against the
    formula inside PaligemmaTokenizer.tokenize() without reusing its code.
    Returns (ref_task_len, ref_state_len, discretized).
    """
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(state, bins=bins) - 1
    state_str = " ".join(map(str, discretized))
    full_ids   = sp.encode(f"Task: {cleaned}, State: {state_str};\nAction: ", add_bos=True)
    task_ids   = sp.encode(f"Task: {cleaned}, State: ", add_bos=True)
    suffix_ids = sp.encode(";\nAction: ")
    ref_task_len  = len(task_ids)
    ref_state_len = len(full_ids) - len(task_ids) - len(suffix_ids)
    return ref_task_len, ref_state_len, discretized


# ---------------------------------------------------------------------------
# Exact-boundary tests
# ---------------------------------------------------------------------------

def test_tokenize_with_state_exact_boundary():
    """task_token_len and state_token_len must match SentencePiece positions exactly.

    Covers: zeros, negatives, decimals, extreme bins (-1/+1), long state vectors.
    Also decodes each segment and checks content so a suffix off-by-one is caught
    even if both the tokenizer and _ref_boundaries share the same boundary formula.
    """
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=200)
    sp = tokenizer._tokenizer

    cases = [
        ("pick up the block",        np.zeros(4, dtype=np.float32)),
        ("move the cup to the left", np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)),
        ("place object on shelf",    np.array([0.123, -0.456, 0.789, -0.012], dtype=np.float32)),
        ("grasp",                    np.linspace(-1.0, 1.0, 16).astype(np.float32)),
        ("open gripper",             np.full(8, 0.0, dtype=np.float32)),   # all same bin
        ("rotate wrist",             np.array([-1.0, 1.0], dtype=np.float32)),  # extreme bins
        ("reach",                    np.zeros(32, dtype=np.float32)),      # large vector
    ]

    for prompt, state in cases:
        tokens, masks, task_len, state_len = tokenizer.tokenize(prompt, state)[:4]
        ref_task_len, ref_state_len, discretized = _ref_boundaries(sp, prompt, state)

        assert task_len == ref_task_len, (
            f"[{prompt!r}] task_token_len: got {task_len}, want {ref_task_len}"
        )
        assert state_len == ref_state_len, (
            f"[{prompt!r}] state_token_len: got {state_len}, want {ref_state_len}"
        )

        # State segment decodes to text that contains every discretized digit string.
        state_ids = tokens[task_len : task_len + state_len].tolist()
        decoded_state = sp.decode(state_ids)
        for d in discretized:
            assert str(d) in decoded_state, (
                f"[{prompt!r}] digit {d} missing from decoded state {decoded_state!r}"
            )

        # Suffix segment (state_end … last valid token) must contain ";\nAction".
        valid = int(masks.sum())
        suffix_ids_actual = tokens[task_len + state_len : valid].tolist()
        decoded_suffix = sp.decode(suffix_ids_actual)
        assert ";" in decoded_suffix and "Action" in decoded_suffix, (
            f"[{prompt!r}] suffix decoded to {decoded_suffix!r}"
        )


def test_tokenize_with_state_truncation_clamps_state_len():
    """When max_len forces truncation, state_token_len is clamped to available slots."""
    # max_len=30 is too small for "Task: move forward, State: <20 digits>;\nAction: "
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=30)

    state = np.linspace(-1.0, 1.0, 20).astype(np.float32)
    tokens, masks, task_len, state_len = tokenizer.tokenize("move forward", state)[:4]

    assert tokens.shape == (30,)
    assert task_len is not None and state_len is not None
    # All 30 slots occupied (truncated, no padding).
    assert int(masks.sum()) == 30
    # State span must not overflow the buffer.
    assert task_len + state_len <= 30
    # At least some state tokens survived.
    assert state_len > 0
    # Clamping invariant: state_len <= effective_len - task_len.
    assert state_len <= 30 - task_len


def test_pi05_span_no_overlap():
    """Simulate _compute_debug_info span layout; assert task/state do not overlap."""
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=200)

    prompt = "pick up the block"
    state  = np.array([0.1, -0.2, 0.3, -0.4, 0.0, 0.7], dtype=np.float32)
    _, masks, task_len, state_len = tokenizer.tokenize(prompt, state)[:4]

    assert task_len  is not None and task_len  > 0
    assert state_len is not None and state_len > 0

    n_cameras  = 3
    images_end = n_cameras * 256   # 768 for libero / aloha configs
    task_span  = (images_end, images_end + task_len)
    state_span = (images_end + task_len, images_end + task_len + state_len)
    key_end    = images_end + 200  # π0.5 prefix_len = images_end + max_token_len

    assert task_span[1] == state_span[0], "task/state spans must be adjacent"
    assert state_span[1] <= key_end,      "state span must not exceed key_end"


# ---------------------------------------------------------------------------

def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
