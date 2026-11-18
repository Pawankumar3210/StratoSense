# CLAUDE.md — StratoSense development guide

StratoSense segments clouds in four-band (R, G, B, NIR) Landsat 8 imagery with a U-Net trained on
38-Cloud, and serves results through a FastAPI backend and a React frontend. Full product spec:
the "StratoSense Master Build Prompt" (kept by the team). This file is the working agreement.

## Non-negotiable rules
1. Never fake predictions, metrics, training curves, sample data or satellite facts. If something is
   unavailable, the UI and API say so (`/ready` is 503 until a real checkpoint is loaded).
2. Band order is a contract: `red, green, blue, nir` (`ml/config.py::BAND_ORDER`). Never reorder silently.
3. Match bands by patch **name**, never by directory order.
4. Split by **scene**, never by patch. Official test scenes are never used for tuning or threshold selection.
5. Normalization statistics come from training scenes only and are persisted (`artifacts/normalization.json`).
6. Exclude nodata pixels (all four bands == 0) from coverage statistics and from the loss/metrics.
7. Don't commit datasets, checkpoints (`*.pth`), secrets or `.env` files.
8. Don't claim something works (export, deployment, training) until it has been run and verified.
9. No auth, no database, no permanent storage of uploads.
10. No unnecessary dependencies. Add Three.js/Recharts etc. only when the feature that uses them is built.

## Layout
- `ml/` — single source of truth for data, preprocessing, scene reconstruction, (later) model/loss/metrics/train.
  `ml/preprocessing.py` is NumPy-only and is imported by both training and the backend.
- `backend/app/` — FastAPI. Imports `ml.*`; thin adapters live in `app/services` and (later) `app/ml`.
- `frontend/` — React + TypeScript + Vite + Tailwind v4. Tokens in `src/index.css`. `src/types/api.ts` mirrors backend schemas.
- `notebooks/` — thin notebooks that call tested code in `ml/`.
- `docs/` — dataset, architecture, deployment notes. `artifacts/` — metrics/checkpoints (checkpoints git-ignored).

## Commands (repo root, venv active: `python -m venv .venv && . .venv/bin/activate`)
```
pip install -r requirements-dev.txt && pip install -e .
pytest                                   # ml/tests + backend/tests
ruff check . && ruff format --check .
(cd backend && uvicorn app.main:app --reload --port 8000)   # needs PYTHONPATH=..:. or `pip install -e .`
cd frontend && npm install && npm run dev      # http://localhost:5173
cd frontend && npm run typecheck && npm test && npm run build
python -m ml.compute_stats --data-root "$STRATOSENSE_DATA_ROOT"   # needs the real dataset
```
PyTorch is not in the requirements files; install the build for your platform (preinstalled on Colab).
`ml/tests/test_torch_dataset.py` is skipped automatically when torch is absent.

## Conventions
- Python 3.10+, type hints, ruff (line length 100). Raise specific errors (`DatasetError`, `PreprocessingError`); no silent fallbacks.
- TypeScript strict. All API calls go through `src/services/api.ts`; no business logic in page components.
- Tests accompany code. Synthetic fixtures (`ml/tests/conftest.py`) mimic dataset *structure* only and must never be shown as real data.
- Prefer the simplest thing that is correct. Verify against the real dataset where a test can only check internal consistency.

## Status
See README "Project status". Update it whenever a stage completes.
