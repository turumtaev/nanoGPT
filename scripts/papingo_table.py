#!/usr/bin/env python3
"""
Render a per-layer prediction table for a Papingo checkpoint on val data.
"""
import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import GPTConfig, GPT


def _display_char(ch: str) -> str:
    if ch == "\n":
        return "\\n"
    if ch == "\t":
        return "\\t"
    if ch == " ":
        return "[space]"
    if ch == "|":
        return "\\|"
    if ch.isprintable():
        return ch
    return repr(ch)


def _slice_sentences(text: str, num_sentences: int, max_chars: int) -> str:
    if num_sentences <= 0:
        return text[:max_chars]
    enders = {".", "!", "?"}
    count = 0
    for i, ch in enumerate(text):
        if ch in enders:
            count += 1
            if count >= num_sentences:
                return text[: i + 1]
        if i + 1 >= max_chars:
            return text[: max_chars]
    return text[:max_chars]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="papingo_cpu_mvp_right_config_max_all_0.9/ckpt.pt",
        help="Path to checkpoint file.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--num_chars", type=int, default=300)
    parser.add_argument("--sentences", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    gptconf = GPTConfig(**ckpt["model_args"])
    model = GPT(gptconf)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    dataset = "shakespeare_char"
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        dataset = ckpt["config"].get("dataset", dataset)
    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    itos = meta["itos"]

    data_dir = os.path.join("data", dataset)
    val_path = os.path.join(data_dir, "val.bin")
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    if args.start_idx < 0:
        start = random.randint(0, len(data) - (args.num_chars + 2))
    else:
        start = args.start_idx
    tokens = data[start : start + args.num_chars + 2].astype(np.int64)
    text = "".join(itos[i] for i in tokens.tolist())
    text = _slice_sentences(text, args.sentences, args.num_chars)
    use_len = len(text)
    if use_len < 2:
        raise ValueError("Not enough characters to build targets.")
    idx = torch.tensor(tokens[: use_len - 1], dtype=torch.long, device=args.device)[None, ...]
    targets = torch.tensor(tokens[1:use_len], dtype=torch.long, device=args.device)[None, ...]

    with torch.no_grad():
        logits, _loss, aux = model(idx, targets, training=False, return_layers=True)
    layer_logits = aux["layer_logits"]
    n_layer = len(layer_logits)
    threshold = model.config.confidence_threshold

    print("Sample text:")
    print(text)
    print()

    headers = ["GT"] + [f"L{i}" for i in range(n_layer)] + ["p(gt)", "p(pred)"]
    print("| " + " | ".join(headers) + " |")
    print("|" + " --- |" * len(headers))

    for t in range(idx.size(1)):
        gt_id = targets[0, t].item()
        gt_ch = _display_char(itos[gt_id])
        chosen_layer = n_layer - 1
        per_layer_preds = []
        per_layer_maxp = []
        per_layer_probs = []
        for i in range(n_layer):
            probs = F.softmax(layer_logits[i][0, t], dim=-1)
            maxp, pred_id = probs.max(dim=-1)
            per_layer_probs.append(probs)
            per_layer_preds.append(pred_id.item())
            per_layer_maxp.append(maxp.item())
        for i in range(n_layer):
            if per_layer_maxp[i] >= threshold:
                chosen_layer = i
                break
        pred_id = per_layer_preds[chosen_layer]
        pred_ch = _display_char(itos[pred_id])
        gt_prob = per_layer_probs[chosen_layer][gt_id].item()
        pred_prob = per_layer_probs[chosen_layer][pred_id].item()

        incorrect = pred_id != gt_id
        if incorrect:
            gt_disp = f"<u>{gt_ch}</u>"
            pred_disp = f"<u>{pred_ch}</u>"
        else:
            gt_disp = gt_ch
            pred_disp = pred_ch

        row = [gt_disp]
        for i in range(n_layer):
            if i == chosen_layer:
                row.append(pred_disp)
            else:
                row.append("")
        row.append(f"{gt_prob:.3f}")
        row.append(f"{pred_prob:.3f}")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
