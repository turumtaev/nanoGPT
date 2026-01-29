# Papingo (nanoGPT fork) — Concept & MVP Notes

This repo contains an experimental model variant called **Papingo**. It keeps a nanoGPT‑style Transformer backbone but changes how prediction is supervised and how layers can “early‑exit” during inference.

The MVP currently targets **character‑level Shakespeare** (same dataset as `data/shakespeare_char`), with configurable vocab size. The goal is to explore whether a multi‑layer prediction scheme with early exits can match vanilla nanoGPT quality while potentially reducing compute at inference.

---

## Core Idea

Papingo introduces two main changes:

1) **Per‑layer prediction heads**
Each Transformer block has its own classifier head that predicts the next token (byte/char). So every layer tries to predict the next token.

2) **Early exit by confidence**
At inference, for each position, you can stop at the **first layer** whose prediction is confident enough (e.g., max probability ≥ `threshold`). If no layer is confident, use the last layer.

To approximate this at training time, we mask the context in deeper layers when earlier layers were confident, simulating what would happen if inference exited early and those tokens were no longer processed by deeper layers.

---

## Formal Spec (MVP)

### Tokens
- **MVP uses characters**, not bytes, for simplicity (`shakespeare_char`).
- `vocab_size` is configurable.

### Model
- Standard GPT‑style Transformer blocks.
- Learned token embedding **per layer** (value embeddings), one table per layer.
- One classifier head per layer.

### Value embeddings
For position `t` and layer `ℓ`:
- Each layer has its own `E_val^(ℓ)`.
- All layers look up the **same token id** `x_t` at that position.
- Layer input:

```
h_0,t = Block_0( E_val^(0)[x_t] + pos_emb[t] )

h_{ℓ,t} = Block_ℓ( h_{ℓ-1,t} + E_val^(ℓ)[x_t] )   for ℓ > 0
```

### Per‑layer prediction
Each layer outputs logits:
```
logits_ℓ,t = head_ℓ( LN(h_{ℓ,t}) )
```

### Variable layer widths (optional)
You can make earlier layers “smaller” via `layer_widths`, while keeping the last layer at `n_embd`.
Widths must be **non‑decreasing** and divisible by `n_head`.

Example for 4 layers:
```
layer_widths = [64, 128, 256, 256]  # last width == n_embd
```

CLI example:
```
--layer_widths='[64,128,256,256]'
```

### Gated value embeddings (optional)
You can gate how much of the value embedding vs. previous hidden state to use:

```
gate_mode = "none"    # default, h = h + E_val
gate_mode = "learned" # h = gate*E_val + (1-gate)*h, gate = sigmoid(W h)
```

CLI example:
```
--gate_mode=learned
```

### Per-layer context windows (optional)
You can limit attention context length per layer with `layer_contexts`.
Each value is the max number of previous tokens a layer can attend to.

Example for 4 layers:
```
layer_contexts = [8, 16, 32, 64]
```

CLI example:
```
--layer_contexts='[8,16,32,64]'
```

When moving from a smaller layer to a larger one, Papingo **pads** the hidden state with zeros
so that `h_{ℓ-1}` matches the next layer width before adding `E_val^(ℓ)`.

### Early exit (inference)
For a threshold `τ`:
- At each position `t`, find the first layer `ℓ` with confidence ≥ `τ`.
- Use that layer’s prediction.
- If no layer is confident, use the last layer.

Confidence can be defined in two modes:
- `max`: `max softmax probability`
- `gold`: `softmax probability of the ground‑truth token` (training‑only)

---

## Training‑time Context Masking

To simulate early exit at training:

- Compute per‑layer confidence per position.
- For layer `ℓ+1`, **mask out** any positions where layer `ℓ` is confident, so those tokens are **not part of the context** for deeper layers.

This matches the intended inference behavior where confident tokens “stop” and are not processed further.

---

## Losses

Papingo supports configurable supervision:

- `layer_supervision = "all"`
  - Compute loss at every layer for all positions.
  - Total loss = mean of per‑layer losses.

- `layer_supervision = "skip_easy"` (planned)
  - Do not backprop loss on tokens already handled by earlier confident layers.

Additionally, a **simulation loss** can be reported:
- For each token, choose the earliest confident layer and compute loss from that layer’s logits (or last layer if none).

### Optional gradient detachment between layers
You can stop gradients from later layers flowing into earlier layers by setting:

```
detach_between_layers = True
```

This makes each layer optimize its own classifier head more independently. The next layer still consumes `h_ℓ`, but does not backprop into the previous block. This is expected to increase early‑exit rates for shallow layers.

---

## Differences from vanilla nanoGPT

- No shared token embedding / output head tying.
- One embedding table per layer (value embedding).
- One classifier head per layer.
- Optional early‑exit logic based on confidence.
- Masking of context for deeper layers to match early‑exit behavior.

---

## Current Status (MVP)

Implemented:
- Per‑layer value embeddings.
- Per‑layer heads.
- Confidence‑based context masking between layers.
- Inference‑simulated loss logging.
- Exit‑rate metrics and overall accuracy logging.

Still TODO / under discussion:
- Tests for masking correctness.

See `PAPINGO_TODO.md` for the detailed checklist.

---

## Example CPU run (Shakespeare Char)

```
python data/shakespeare_char/prepare.py

python train.py \
  --dataset=shakespeare_char \
  --device=cpu \
  --compile=False \
  --eval_interval=200 \
  --eval_iters=20 \
  --log_interval=10 \
  --batch_size=12 \
  --block_size=128 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --confidence_threshold=0.9 \
  --confidence_mode=max \
  --layer_supervision=all \
  --detach_between_layers=False \
  --layer_widths=None \
  --gate_mode=none \
  --layer_contexts=None
```

---

## Best Metrics So Far

### CPU (small config)
Config: `block_size=64`, `n_layer=4`, `n_head=4`, `n_embd=128`, `confidence_mode=max`, `confidence_threshold=0.9`, `layer_supervision=all`.

- **val loss:** 1.8779  
- **val infer:** 1.8633  
- **val acc:** 0.447  
- **val exit rates:** `[0:0.056, 1:0.023, 2:0.008, 3:0.913]`

### GPU A100 (config/train_shakespeare_char.py)
Config: `block_size=256`, `n_layer=6`, `n_head=6`, `n_embd=384`, `dropout=0.2`, `confidence_mode=max`, `confidence_threshold=0.9`, `layer_supervision=all`.

- **val loss:** 1.5197  
- **val infer:** 1.4947  
- **val acc:** 0.571  
- **val exit rates:** `[0:0.195, 1:0.105, 2:0.038, 3:0.017, 4:0.006, 5:0.640]`

---

## Notes

- `confidence_mode=gold` should only be used when targets are provided.
- Larger `block_size` tends to reduce loss; compare to vanilla nanoGPT using the same config for fair comparisons.

## Papingo Table (Per-Token Exit Visualization)

Use `scripts/papingo_table.py` to render a per-token table that shows:
- Ground-truth character
- Which layer produced the exit prediction
- Probabilities for ground-truth and predicted characters

Example run (CPU checkpoint):
```
python scripts/papingo_table.py \
  --ckpt papingo_cpu_mvp_right_config_max_all_0.9/ckpt.pt \
  --num_chars 300 \
  --sentences 2 \
  --start_idx 0
```

Example output saved in `papingo_table_example.md`.
