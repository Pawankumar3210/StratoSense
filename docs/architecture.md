# Architecture and decisions

```
 38-Cloud (Landsat 8, 4 bands)           user GeoTIFF (4 bands, uint16)
            │                                         │
            ▼                                         ▼
   ml/dataset.py  ──►  ml/preprocessing.py  ◄──  backend (validation, later)
   (name-based matching,  (validity mask, training-only
    scene-level split)      normalization stats)
            │                        │
            ▼                        ▼
   training (Colab, GPU)  ──►  checkpoint (.pth, versioned artifact)
                                     │
                                     ▼
                        FastAPI /api/v1  ──►  React frontend
```

## Decisions (foundation stage)
| Decision | Reason |
|---|---|
| Root `ml/` package is the single source of truth; the backend imports it | Preprocessing cannot drift between training and inference. Spec's `backend/app/ml/` becomes a thin adapter added with inference. |
| `ml/preprocessing.py`, `dataset.py`, `scene.py` are NumPy/rasterio only | Backend and tests need no PyTorch; only `ml/torch_dataset.py` imports torch. |
| Strict failures (`DatasetError`, `PreprocessingError`) | Unsupported inputs (wrong dtype, band count, label values, missing files) are rejected instead of producing misleading output. |
| Nodata = all four bands equal 0 | Matches the black scene margin; a single zero band does not invalidate a pixel. Nodata is set to 0 after normalization and excluded from loss, metrics and coverage. |
| Normalization = `raw/65535` then per-band mean/std from training scenes, stored in JSON | Reproducible, persisted, applied identically at train and serve time. Never per-image. |
| Deterministic scene split by SHA-256 ranking | Identical across machines and Python versions; no patch-level leakage. |
| `/ready` returns 503 until a real model is loaded | Truthful readiness; the browser logs the 503 as a console error while no model exists. `/health` stays 200 for platform health checks. |
| Settings via pydantic-settings; production forbids `CORS_ORIGINS=*` | Matches the spec's security requirements; misconfiguration fails at startup. |
| Frontend: Tailwind v4 with CSS-variable tokens, IBM Plex Sans self-hosted via `@fontsource` | Dark/light themes by swapping variables; no third-party font requests. |
| Frontend dependencies limited to what exists | Three.js, React Three Fiber, Drei and Recharts are added with the globe and charts. |
| Unbuilt modules are shown as "Planned" with what they need | No feature appears to work when it does not. |
| API base URL from `VITE_API_BASE_URL`; localhost fallback only in dev builds | Verified absent from the production bundle. |

## Backend modules (current)
`app/main.py` (factory, CORS, sanitized 500s) · `app/config.py` · `app/api/health.py` · `app/services/model_registry.py` · `app/schemas/response.py`.
