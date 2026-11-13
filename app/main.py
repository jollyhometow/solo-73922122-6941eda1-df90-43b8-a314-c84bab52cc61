"""FastAPI application: upload, status, cancel and download endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .config import Settings
from .manager import JobManager, JobNotFoundError, LimitExceededError
from .netcdf_io import DatasetError

logger = logging.getLogger("web")

_TEMPLATES = Path(__file__).with_name("templates")


def _job_payload(job) -> dict[str, Any]:
    return {
        "id": job.id,
        "state": job.state.value,
        "stage": job.stage.value,
        "message": job.message,
        "upload_name": job.upload_name,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "baseline_active": job.baseline_active,
        "download": f"/api/jobs/{job.id}/download",
        "cancel": f"/api/jobs/{job.id}/cancel",
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = JobManager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        await manager.startup()
        app.state.manager = manager
        app.state.settings = settings
        try:
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(
        title="tas anomaly service",
        version="1.0.0",
        description=(
            "Upload monthly tas NetCDF files and receive anomaly NetCDF files "
            "relative to a fixed historical baseline."
        ),
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((_TEMPLATES / "index.html").read_text("utf-8"))

    @app.get("/api/baseline")
    async def baseline_info() -> dict[str, Any]:
        return manager.baseline_snapshot()

    @app.get("/api/samples")
    async def list_samples() -> dict[str, Any]:
        samples_dir = settings.samples_dir
        files = []
        if samples_dir.is_dir():
            for path in sorted(samples_dir.glob("*.nc")):
                files.append(
                    {
                        "name": path.name,
                        "size": path.stat().st_size,
                        "href": f"/samples/{path.name}",
                    }
                )
        return {"samples": files}

    @app.get("/samples/{name}")
    async def download_sample(name: str) -> FileResponse:
        # Prevent path traversal: only bare filenames inside samples_dir.
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="invalid sample name")
        path = settings.samples_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="sample not found")
        return FileResponse(
            path,
            media_type="application/x-netcdf",
            filename=name,
        )

    @app.post("/api/jobs", status_code=201)
    async def submit_job(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = file.filename or "upload.nc"
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="uploaded file is empty")
        try:
            job = await manager.submit(filename, payload)
        except LimitExceededError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DatasetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _job_payload(job)

    @app.get("/api/jobs")
    async def list_jobs() -> dict[str, Any]:
        jobs = [_job_payload(job) for job in manager.list_jobs()]
        active = sum(1 for item in jobs if item["state"] not in (
            "done",
            "error",
            "cancelled",
        ))
        return {
            "jobs": jobs,
            "max_active_jobs": settings.max_active_jobs,
            "active_count": active,
        }

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        try:
            job = manager.get_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return _job_payload(job)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            job = await manager.cancel(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        return _job_payload(job)

    @app.get("/api/jobs/{job_id}/download")
    async def download_job(job_id: str) -> FileResponse:
        try:
            job = manager.get_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if not job.output_path.is_file():
            raise HTTPException(
                status_code=409,
                detail=f"job is '{job.state.value}': no result file available",
            )
        stem = Path(job.upload_name).stem or "target"
        return FileResponse(
            job.output_path,
            media_type="application/x-netcdf",
            filename=f"{stem}_anomaly.nc",
        )

    @app.exception_handler(DatasetError)
    async def dataset_error_handler(_request, exc: DatasetError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
