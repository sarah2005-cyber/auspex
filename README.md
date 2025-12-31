# FullPipelineModel - Backend and Frontend

This workspace contains a FastAPI backend and a React frontend to run predictions
and generate SHAP explanations for a PyTorch model checkpoint named
`FullPipelineModel_seed42_epoch29_F10.8015_best.pt` located in the repository root.

## Backend (FastAPI)

Location: `backend/`

Install dependencies (recommended in a virtualenv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Run the FastAPI server:

```powershell
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- POST /predict  -> JSON body {"features": [f1, f2, ...]} returns {"score": ...}
- POST /explain  -> JSON body {"features": [f1, f2, ...]} returns SHAP values and a `plot_url`

Notes:
- The model class `FullPipelineModel` in `backend/model.py` is a placeholder. Replace its
  layers with your real architecture matching the checkpoint for correct inference.
- `load_model` attempts to load the checkpoint using `strict=False` so it will not crash
  if shapes differ; replace the architecture and re-run for full compatibility.

## Frontend (React + MUI)

Location: `frontend/`

Install and run:

```powershell
cd frontend
npm install
npm start
```

This starts a development server on http://localhost:3000 which talks to the backend on port 8000.

## TODOs / Next steps
- Replace `FullPipelineModel` with the real architecture.
- Replace placeholder preprocessing in `backend/model.py` to match training transforms.
- Provide a representative background dataset for SHAP to get meaningful explanations.
