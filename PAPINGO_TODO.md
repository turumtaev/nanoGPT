# Papingo MVP Todo List (nanoGPT)

## Scope (MVP)
- [ ] Use `shakespeare_char` dataset (char-level vocab; `vocab_size` configurable).
- [ ] Keep absolute positional embeddings (WPE) as in baseline.
- [x] Add confidence threshold `tau` to config.
- [ ] Keep training/inference behavior configurable where noted below.

## Model Architecture
- [ ] Add per-layer classifier heads: one logits head per transformer block.
- [ ] Add per-layer value embeddings: one `nn.Embedding(vocab_size, n_embd)` per block.
- [ ] Define how value embeddings are injected: add to hidden before each block `l` (except first), i.e. `h_l = Block_l(h_{l-1} + E_val^l[x_t])`.

## Forward Pass Behavior
- [ ] Compute per-layer logits after each block.
- [ ] Compute per-layer confidence per position based on config:
  - `confidence_mode = "max"` (max softmax prob) OR
  - `confidence_mode = "gold"` (softmax prob of ground-truth token).
- [ ] Implement inference early-exit: for each position, stop at first layer with confidence ≥ `tau`.

## Training-Time Masking (Context Skip)
- [ ] For layer `l+1`, remove from context any positions where layer `l` is confident (per chosen confidence mode).
- [ ] Keep predicting all tokens at all layers, but allow configurable loss handling:
  - `layer_supervision = "all"` (loss at every layer for every token) OR
  - `layer_supervision = "skip_easy"` (skip loss where earlier layer was confident).

## Losses & Logging
- [ ] Aggregate per-layer losses (mean or weighted sum) for backprop.
- [ ] Log per-layer losses.
- [ ] Log an "inference-simulated" loss: for each token, use the first confident layer’s logits (or last layer if none confident).

## Code Changes (Mapping)
- [x] `GPTConfig`: add `confidence_threshold`, `confidence_mode`, `layer_supervision`.
- [x] `GPT`: add per-layer heads and per-layer value embeddings.
- [ ] `GPT.forward`: return per-layer logits and losses; implement masking & confidence.
- [ ] `train.py`: handle new outputs and logging.

## Validation
- [ ] Shape checks (forward pass, per-layer logits).
- [ ] Tiny overfit run on `shakespeare_char` with local CPU settings.
- [ ] Toy masking sanity check with a short string (e.g. "WORLD").
