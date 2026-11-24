"""
FastAPI Server Entry Point
Quantum k-Nearest Neighbour Classification Platform with Intelligent Dataset Profiling
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.api.fastapi_routes import router as dataset_router

app = FastAPI(
    title="Quantum k-NN Research Platform API",
    description="Intelligent Dataset Profiling & Hybrid Quantum k-Nearest Neighbour Classifier using PennyLane",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for full-stack frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(dataset_router)


@app.get("/")
async def root():
    return {
        "service": "Quantum k-NN Research & Intelligent Dataset Profiling API",
        "framework": "FastAPI + PennyLane",
        "quantum_simulator": "default.qubit",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "quantum_engine": "PennyLane 0.45",
        "simulator": "default.qubit",
        "endpoints": [
            "POST /api/dataset/upload",
            "POST /api/dataset/load-sample/{sample_name}",
            "GET  /api/dataset/{dataset_id}/profile",
            "POST /api/dataset/{dataset_id}/configure",
            "POST /api/dataset/{dataset_id}/train",
            "POST /api/dataset/{dataset_id}/predict",
            "POST /api/dataset/{dataset_id}/batch-predict",
            "GET  /api/dataset/{dataset_id}/quantum-circuit",
            "GET  /api/dataset/{dataset_id}/report",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting FastAPI Quantum Research Server on http://0.0.0.0:{port}")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
