# Fine-tuning Qwen2.5-3B-Instruct for factory alert narration (Windows setup)

This is the `model/` half of the repo — training/fine-tuning only. The
chatbot + dashboard live in `chat/` and are unaffected by anything here
except that they load a separate, unmodified copy of the same base model
(see `chat/backend/config.py`).

## What's already done for you
- `model/data/context_json/*.json` + `model/data/narrations/*.txt` — 3000
  synthetic samples generated from `model/dataset_builder/generate_bulk_dataset.py`
  (covers FAST_INSPECTION, HAND_TOUCH, MISSING_CLEANING alert types).
- `model/outputs/high_quality_dataset.jsonl` — instruction-formatted dataset
  built by `model/hf_training/prepare_instruction_dataset.py`, using Qwen's
  ChatML template (`<|im_start|>system / user / assistant ... <|im_end|>`).
- `model/hf_training/configs.py` — points at `Qwen/Qwen2.5-3B-Instruct`.
- `model/hf_training/train_lora.py` — QLoRA config updated for Qwen's module
  names (q/k/v/o_proj + gate/up/down_proj), max_length raised to 256 tokens
  to fit the bilingual EN/TA narrations.

This is still **synthetic placeholder data** (a handful of template sentences
per alert type, randomly recombined). It's enough to prove the training
pipeline works end-to-end, but the model will just learn to reproduce those
same handful of sentences — it won't generalize well. Swap in real
detection-log-derived samples (via `model/dataset_builder/build_dataset.py`,
which reads real `model/data/detection_json/*.json`) before you rely on this
for anything real.

## Windows-specific setup

You said you built the original pipeline on Ubuntu and are now on Windows —
a few things differ:

1. **Use WSL2, not native Windows, if you can.** `bitsandbytes` (needed for
   4-bit QLoRA) has native Windows wheels since ~0.43, but they're still
   less battle-tested than Linux. If you hit CUDA/bitsandbytes errors on
   native Windows, install WSL2 + Ubuntu and run everything there instead —
   it'll behave exactly like your original Ubuntu setup.
2. **Paths**: `model/config.py` and `model/hf_training/configs.py` use
   `pathlib.Path`, which is cross-platform safe — no changes needed there.
3. **CUDA**: install a CUDA-enabled `torch` matching your GPU driver, e.g.:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
   (check https://pytorch.org/get-started/locally/ for the right cu-tag for
   your driver version).

## GPU requirements

Qwen2.5-3B in 4-bit QLoRA needs roughly **6–8 GB VRAM**. If your GPU has
less, reduce `per_device_train_batch_size` (already at 1) or run on a cloud
GPU (Colab T4 works). Confirmed working on a 4GB RTX 3050 laptop GPU with
these settings, though tightly.

## Steps to run

Everything below runs **from the repo root** (`slm-main/`), not from inside
`model/` — the package layout expects that.

```bash
pip install -r model/requirements.txt

# (1) already done — regenerate only if you want more/different synthetic data
python -m model.dataset_builder.generate_bulk_dataset

# (2) already done — regenerate only if you changed data or the prompt format
python -m model.hf_training.prepare_instruction_dataset

# (3) run QLoRA fine-tuning — this is the step you still need to run yourself
python -m model.hf_training.train_lora
```

Output LoRA adapter + tokenizer get saved to
`model/outputs/qwen2.5_3b_lora/`.

## After training

- `model/hf_training/merge_model.py` merges the LoRA adapter back into the
  base Qwen weights for a standalone model — check `CHECKPOINT_PATH` points
  at the checkpoint you want before running (defaults to `checkpoint-1375`).
- `model/hf_training/test_merged_model.py`, `stress_test.py`, and
  `stress_test_en.py` let you sanity-check generations against the merged
  model in `model/outputs/industrial_slm/`.

## Known gaps to fix before production use

- Replace synthetic dataset with real labeled detection logs.
- `trl`'s `SFTTrainer`/`SFTConfig` API has shifted across versions — if
  `python -m model.hf_training.train_lora` errors on `SFTTrainer(args=...)`,
  check your installed `trl` version's docs; newer versions expect an
  `SFTConfig` object instead of plain `TrainingArguments`.
- No train/val split yet — a single `dataset["train"]` covers all 3000
  samples with no held-out eval set.
