# Deployment (not yet performed)

Nothing has been deployed. Planned targets: Vercel (frontend, `VITE_API_BASE_URL`), Render (Docker FastAPI), and a
versioned model artifact (GitHub Release or Hugging Face Hub) referenced by `MODEL_ARTIFACT_URL` and checked with `MODEL_SHA256`.
Dockerfile, `render.yaml`, Vercel SPA rewrite and the PyTorch CPU install instructions are added in the deployment stage.

Until a checkpoint exists at `MODEL_PATH` and the loader is implemented, `GET /api/v1/ready` returns 503 by design.
