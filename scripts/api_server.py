"""REST API for Satellite Analysis.

FastAPI-based REST API for satellite imagery analysis.
Suitable for integrations, microservices, and web applications.

Run with:
    uvicorn scripts.api_server:app --reload --port 8000
    
Or:
    python scripts/api_server.py

API Endpoints:
    GET  /                      - API info
    GET  /health                - Health check
    POST /analyze               - Analyze a city
    POST /analyze/batch         - Analyze multiple cities
    POST /compare               - Change detection
    GET  /results/{job_id}      - Get job results
    GET  /cities                - List available cities
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import satellite analysis
from satellite_analysis import (
    analyze, analyze_batch, compare,
    export_geotiff, export_report, export_json, export_rgb,
    LAND_COVER_CLASSES,
)

# =============================================================================
# API Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for single city analysis."""
    city: str = Field(..., description="City name (e.g., Florence, Milan)")
    max_size: Optional[int] = Field(2000, description="Max image dimension")
    classifier: Optional[str] = Field("consensus", description="Classification method")
    n_clusters: Optional[int] = Field(6, description="Number of clusters for kmeans")
    raw_clusters: Optional[bool] = Field(False, description="Keep raw cluster IDs (no semantic mapping)")
    exports: Optional[List[str]] = Field(None, description="Export formats: geotiff, report, json, rgb")
    language: Optional[str] = Field("en", description="Report language: en, it")

class BatchRequest(BaseModel):
    """Request model for batch analysis."""
    cities: List[str] = Field(..., description="List of city names")
    max_size: Optional[int] = Field(2000, description="Max image dimension")
    classifier: Optional[str] = Field("consensus", description="Classification method")
    n_clusters: Optional[int] = Field(6, description="Number of clusters for kmeans")
    raw_clusters: Optional[bool] = Field(False, description="Keep raw cluster IDs (no semantic mapping)")

class CompareRequest(BaseModel):
    """Request model for change detection."""
    city: str = Field(..., description="City name")
    date_before: str = Field(..., description="Earlier date (YYYY-MM)")
    date_after: str = Field(..., description="Later date (YYYY-MM)")
    max_size: Optional[int] = Field(2000, description="Max image dimension")

class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    city: str
    shape: List[int]
    total_pixels: int
    avg_confidence: float
    execution_time: float
    class_distribution: Dict[str, Any]
    output_dir: str
    exports: Optional[Dict[str, str]] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="Satellite City Analyzer API",
    description="""
REST API for Sentinel-2 satellite imagery analysis.

## Features
- **Single Analysis**: Analyze land cover for one city
- **Batch Analysis**: Process multiple cities at once
- **Change Detection**: Compare land cover across time
- **Exports**: GeoTIFF, HTML reports, JSON

## Quick Start
```python
import requests

# Analyze a city
response = requests.post("http://localhost:8000/analyze", json={
    "city": "Florence",
    "max_size": 2000,
    "exports": ["report"]
})
print(response.json())
```
    """,
    version="2.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage (in production, use Redis or database)
jobs: Dict[str, JobStatus] = {}

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - returns basic info."""
    return {
        "name": "Satellite City Analyzer API",
        "version": "2.1.0",
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /analyze",
            "batch": "POST /analyze/batch",
            "compare": "POST /compare",
            "cities": "GET /cities",
            "health": "GET /health",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.1.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/cities", tags=["Info"])
async def list_cities():
    """List available cities with data."""
    data_dir = PROJECT_ROOT / "data" / "cities"
    
    cities = []
    
    # Check cities directory
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir() and (d / "bands").exists():
                cities.append({
                    "name": d.name.title(),
                    "key": d.name,
                    "has_data": True,
                    "source": "local"
                })
    
    return {
        "cities": cities,
        "count": len(cities),
        "land_cover_classes": LAND_COVER_CLASSES
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_city(request: AnalyzeRequest):
    """Analyze a single city.
    
    Returns land cover classification results including:
    - Class distribution
    - Confidence scores
    - Optional exports (GeoTIFF, HTML report, JSON)
    """
    try:
        # Run analysis (blocking - consider using background tasks for large images)
        result = analyze(
            request.city,
            max_size=request.max_size,
            classifier=request.classifier,
            n_clusters=request.n_clusters,
            raw_clusters=request.raw_clusters,
            project_root=PROJECT_ROOT,
        )
        
        # Handle exports
        exports_result = {}
        if request.exports:
            if "geotiff" in request.exports:
                path = export_geotiff(result)
                exports_result["geotiff"] = str(path)
            if "report" in request.exports:
                path = export_report(result, language=request.language)
                exports_result["report"] = str(path)
            if "json" in request.exports:
                path = export_json(result)
                exports_result["json"] = str(path)
            if "rgb" in request.exports:
                path = export_rgb(result, dpi=100, include_ndvi=False)
                exports_result["rgb"] = str(path)
        
        # Build response
        return AnalysisResponse(
            city=result.city,
            shape=list(result.processed_shape),
            total_pixels=result.total_pixels,
            avg_confidence=result.avg_confidence,
            execution_time=result.execution_time,
            class_distribution={
                str(k): {
                    "name": LAND_COVER_CLASSES.get(k, {}).get("name", f"Class {k}"),
                    **v
                }
                for k, v in result.class_distribution().items()
            },
            output_dir=str(result.output_dir),
            exports=exports_result if exports_result else None
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_cities_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """Analyze multiple cities (async).
    
    Creates a background job and returns job ID immediately.
    Poll /jobs/{job_id} for status and results.
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Create job
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now().isoformat()
    )
    
    # Run in background
    background_tasks.add_task(
        run_batch_job,
        job_id,
        request.cities,
        request.max_size,
        request.classifier,
        request.n_clusters,
        request.raw_clusters
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "cities": request.cities,
        "check_status": f"/jobs/{job_id}"
    }


@app.post("/compare", tags=["Analysis"])
async def compare_dates(request: CompareRequest):
    """Compare land cover between two dates.
    
    Detects changes in land cover classification between
    the two specified time periods.
    """
    try:
        changes = compare(
            request.city,
            request.date_before,
            request.date_after,
            max_size=request.max_size,
            project_root=PROJECT_ROOT,
        )
        
        return {
            "city": changes.city,
            "date_before": changes.date_before,
            "date_after": changes.date_after,
            "shape": list(changes.shape),
            "total_pixels": changes.total_pixels,
            "changed_pixels": changes.changed_pixels,
            "changed_percentage": changes.changed_percentage,
            "unchanged_percentage": changes.unchanged_percentage,
            "major_changes": changes.get_major_changes(5),
            "output_dir": str(changes.output_dir),
            "execution_time": changes.execution_time,
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get status of a background job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return jobs[job_id]


@app.get("/download/{file_type}/{city}", tags=["Downloads"])
async def download_file(file_type: str, city: str):
    """Download analysis results.
    
    Args:
        file_type: geotiff, report, or json
        city: City name
    """
    city_dir = PROJECT_ROOT / "data" / "cities" / city.lower() / "latest"
    
    if not city_dir.exists():
        raise HTTPException(status_code=404, detail=f"No results for {city}")
    
    file_map = {
        "geotiff": (f"{city.lower()}_classification.tif", "image/tiff"),
        "report": (f"{city.lower()}_report.html", "text/html"),
        "json": (f"{city.lower()}_results.json", "application/json"),
        "rgb": (f"{city.lower()}_composites_overview.png", "image/png"),
    }
    
    if file_type not in file_map:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")
    
    filename, media_type = file_map[file_type]
    file_path = city_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


# =============================================================================
# Background Tasks
# =============================================================================

async def run_batch_job(
    job_id: str,
    cities: List[str],
    max_size: int,
    classifier: str,
    n_clusters: int = 6,
    raw_clusters: bool = False
):
    """Run batch analysis in background."""
    jobs[job_id].status = "running"
    
    try:
        results = {}
        for i, city in enumerate(cities):
            jobs[job_id].progress = f"Processing {i+1}/{len(cities)}: {city}"
            
            try:
                result = analyze(
                    city,
                    max_size=max_size,
                    classifier=classifier,
                    n_clusters=n_clusters,
                    raw_clusters=raw_clusters,
                    project_root=PROJECT_ROOT,
                )
                results[city] = {
                    "status": "success",
                    "confidence": result.avg_confidence,
                    "output_dir": str(result.output_dir)
                }
            except Exception as e:
                results[city] = {
                    "status": "error",
                    "error": str(e)
                }
        
        jobs[job_id].status = "completed"
        jobs[job_id].result = results
        jobs[job_id].completed_at = datetime.now().isoformat()
        
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        jobs[job_id].completed_at = datetime.now().isoformat()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Starting Satellite Analysis API...")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
