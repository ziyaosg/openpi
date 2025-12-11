## Big Picture

In upstream OpenPI, policy serving is a black box:

observation → policy → actions

In this fork, we keep that exact behavior **but add an optional side channel** that computes and saves Grad-CAM maps during inference:

observation
→ policy (normal inference)
→ Grad-CAM (optional)
→ actions (unchanged) + saved CAMs / frames

If Grad-CAM is disabled, the system behaves exactly like upstream OpenPI.

---

## Where Grad-CAM Lives

Grad-CAM functionality is split cleanly across **three layers**:

1. **Model (pi0-fast)**  
   Knows how to compute Grad-CAM over visual tokens.

2. **Policy wrapper (serve_policy.py)**  
   Decides *when* to call Grad-CAM during inference and *what* to log.

3. **Logger (cam_logger.py)**  
   Handles saving CAMs and frames to disk in a stable format.

Each part has a single responsibility.

---

## Grad-CAM inside `Pi0FAST`

Grad-CAM is implemented directly in the pi0-fast model (`src/openpi/models/pi0_fast.py`).

**Key idea**

- pi0-fast encodes images using **SigLIP**, producing a grid of visual patch tokens per camera.
- Grad-CAM is computed **with respect to those SigLIP tokens**, not raw pixels.

**How it works (conceptually)**

1. During inference, the model builds a token sequence:

[ image tokens (all cameras) | prompt tokens | (optional generated tokens) ]

2. For a chosen output token:
- Define a scalar score = *the logit of that token*.
- Backpropagate the score w.r.t. the visual token embeddings.

3. For each camera:
- Aggregate gradients over channels (standard Grad-CAM weighting).
- Apply ReLU to keep only positive contributions.
- Reshape back into a `(H, W)` patch grid.

4. Return **one heatmap per camera**, as raw (unnormalized) arrays.

**Important details**

- Camera order and token spans are explicitly tracked, so each heatmap maps back to the correct camera.
- Grad-CAM is implemented in two helpers:
- `gradcam_for_token(...)`: explain a specific token ID.
- `gradcam_for_action_token(...)`: explain the *actual token the model generated* at a given decode step.
- Grad-CAM never affects training or action generation — it is purely diagnostic.

---

## How Grad-CAM is Triggered at Inference Time

Grad-CAM is not invoked inside the model by default. Instead, it is triggered by a **policy wrapper** in `scripts/serve_policy.py`.

**What the wrapper does**

- Wraps a normal `Policy` object.
- Intercepts calls to `policy.infer(obs)`.

For each inference step (when enabled):

1. Run normal policy inference to get actions.
2. Reconstruct the same `Observation` that the model saw (using the policy’s input transform).
3. Extract generated token IDs from the policy output (if available).
4. Call the model’s Grad-CAM helper (`gradcam_for_action_token` or fallback).
5. Collect:
- Grad-CAM maps (per camera).
- RGB input frames (optional).
6. Pass everything to the logger.
7. Return the original action output unchanged.

**Key design choice**

- Grad-CAM is best-effort and non-blocking.
- If anything fails (no tokens, no Grad-CAM support, shape mismatch), inference continues normally.

---

## How Logging Works (`cam_logger.py`)

Grad-CAM results are saved using a small dedicated logger.

**What gets written**

For each logged step:

- One `.npz` file containing:
- Grad-CAM maps (`cam/<camera_name>`).
- Optional RGB frames (`frame/<camera_name>`).
- Basic metadata (timestamp, token id).

- A `manifest.json` file in the run directory that:
- Lists all recorded steps.
- Stores file paths and summary info.

**Why this exists**

- Keeps I/O concerns out of the model and policy code.
- Produces a simple, NumPy-friendly format for downstream analysis:
- Visualization
- Statistics
- Failure case inspection

---

## How Everything Fits Together

Putting it all together, the Grad-CAM pipeline looks like this:

client observation
→ WebSocket server
→ Policy.infer(...)
→ pi0-fast produces actions
→ (optional) Grad-CAM over SigLIP tokens
→ CAMLogger saves heatmaps / frames
→ actions returned to client

Grad-CAM lives entirely on the *side* of inference and never interferes with control or deployment.
