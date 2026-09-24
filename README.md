# slm — Factory Safety Alert System

Two independent halves, split so each can be worked on (and eventually
deployed) without the other:

```
model/     Training & fine-tuning: dataset generation, QLoRA fine-tuning of
           Qwen2.5-3B-Instruct, and evaluation scripts. See
           model/hf_training/TRAINING_README.md.

chat/      The chatbot + FQC dashboard: FastAPI backend (chat/backend/) and
           React frontend (chat/frontend/). Talks to its own MongoDB
           instance and its own copy of the base LLM.

models/    Shared, git-ignored cache of downloaded base model weights
           (Qwen2.5-3B-Instruct) — read by both model/ and chat/, not
           duplicated between them.
```

## The one intentional coupling point

`chat/backend/seed_data.py` imports narration templates from
`model/dataset_builder/narration_generator.py` so demo/seed alerts read the
same as real training data would, instead of duplicating those templates.
Nothing else crosses the `model/` ↔ `chat/` boundary — each has its own
`requirements.txt` and can be set up independently.

## Running things

All commands run **from the repo root**, with one shared virtualenv:

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r model/requirements.txt -r chat/requirements.txt

# Training pipeline — see model/hf_training/TRAINING_README.md for details
python -m model.hf_training.train_lora

# Chat backend
python -m chat.backend.seed_data      # first time only, or to reseed
uvicorn chat.backend.main:app --host 127.0.0.1 --port 8000

# Frontend
cd chat/frontend
npm install
npm run dev
```

MongoDB must be running locally (`mongodb://localhost:27017`, database
`slm_safety`, collection `alerts`).
