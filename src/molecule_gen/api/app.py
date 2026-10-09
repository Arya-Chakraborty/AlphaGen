from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from molecule_gen.constraints.lookup import LookupError, papyrus_corpus_stats
from molecule_gen.constraints.spec import SpecError, parse_spec
from molecule_gen.descriptors.names import (
    FEATURED_DESCRIPTORS,
    allowed_descriptor_names,
    resolve_descriptor_name,
)
from molecule_gen.generate.run import DEFAULT_GENERATIONS, DEFAULT_POPULATION, generate_molecules
from molecule_gen.generate.transformer import default_checkpoint_path

app = FastAPI(title="molecule-gen", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, "GenerateJob"] = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()
_MAX_JOBS = 20


class DescriptorRangeIn(BaseModel):
    min: float
    max: float


class GenerateRequest(BaseModel):
    n: int = Field(default=5, ge=1, le=50)
    descriptors: dict[str, DescriptorRangeIn]
    population: int = Field(default=DEFAULT_POPULATION, ge=4, le=256)
    generations: int = Field(default=DEFAULT_GENERATIONS, ge=0, le=1000)
    seed: int = 0


class GenerateJob:
    def __init__(self, job_id: str, payload: dict[str, Any]):
        self.id = job_id
        self.payload = payload
        self.status = "queued"
        self.history: list[dict[str, Any]] = []
        self.hits: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] | None = None
        self.error: str | None = None
        self.lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        clean = _jsonable(event)
        clean["ts"] = time.time()
        with self.lock:
            self.history.append(clean)
            if clean.get("type") == "done":
                self.hits = list(clean.get("hits") or [])
                self.manifest = clean.get("manifest")
                self.status = "done"
            elif clean.get("type") == "error":
                self.error = str(clean.get("message") or "generation failed")
                self.status = "error"

    def snapshot_history(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.history)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "error": self.error,
                "n_events": len(self.history),
                "hits": list(self.hits),
                "manifest": self.manifest,
            }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and value != value:
        return None
    return value


def _store_job(job: GenerateJob) -> None:
    with _jobs_lock:
        _jobs[job.id] = job
        extra = list(_jobs.keys())[:-_MAX_JOBS]
        for key in extra:
            _jobs.pop(key, None)


def _get_job(job_id: str) -> GenerateJob:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job


@app.get("/health")
def health() -> dict[str, Any]:
    ckpt = default_checkpoint_path()
    return {
        "ok": True,
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
    }


@app.get("/descriptors")
def list_descriptors() -> dict[str, Any]:
    names = allowed_descriptor_names()
    featured = [name for name in FEATURED_DESCRIPTORS if name in names]
    return {"featured": featured, "names": names, "n": len(names)}


@app.get("/descriptors/stats")
def descriptor_stats(
    names: str = Query(..., description="Comma-separated Mordred names"),
) -> dict[str, Any]:
    resolved: list[str] = []
    seen: set[str] = set()
    for part in names.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            local = resolve_descriptor_name(text)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if local in seen:
            continue
        seen.add(local)
        resolved.append(local)
    if not resolved:
        raise HTTPException(status_code=400, detail="Provide at least one descriptor name")
    try:
        stats = papyrus_corpus_stats(resolved)
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "stats": stats,
        "note": (
            "min/max are the range of that descriptor in Papyrus. "
            "They are a reference only; generated molecules may fall outside them."
        ),
    }


@app.get("/depict")
def depict(smiles: str) -> Response:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    drawer = Draw.MolDraw2DSVG(240, 180)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Response(content=drawer.GetDrawingText(), media_type="image/svg+xml")


@app.post("/generate")
def start_generate(body: GenerateRequest) -> dict[str, str]:
    try:
        spec = parse_spec(
            {
                "n": body.n,
                "descriptors": {
                    name: {"min": item.min, "max": item.max}
                    for name, item in body.descriptors.items()
                },
            }
        )
    except SpecError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not default_checkpoint_path().exists():
        raise HTTPException(
            status_code=503,
            detail="SELFIES GPT checkpoint missing. Run train-transformer first.",
        )
    job = GenerateJob(
        uuid.uuid4().hex[:12],
        {
            "n": spec.n,
            "descriptors": spec.as_dict()["descriptors"],
            "population": body.population,
            "generations": body.generations,
            "seed": body.seed,
        },
    )
    _store_job(job)
    thread = threading.Thread(
        target=_run_generate,
        args=(job, spec, body.population, body.generations, body.seed),
        daemon=True,
    )
    thread.start()
    return {"id": job.id}


@app.get("/generate/{job_id}")
def get_generate(job_id: str) -> dict[str, Any]:
    return _get_job(job_id).snapshot()


@app.get("/generate/{job_id}/events")
async def generate_events(job_id: str) -> StreamingResponse:
    job = _get_job(job_id)

    async def stream():
        idx = 0
        while True:
            with job.lock:
                chunk = job.history[idx:]
                status = job.status
                n_hist = len(job.history)
            if chunk:
                for event in chunk:
                    idx += 1
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in {"done", "error"}:
                        return
            elif status in {"done", "error"} and idx >= n_hist:
                return
            else:
                await asyncio.sleep(0.2)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _run_generate(job: GenerateJob, spec, population: int, generations: int, seed: int) -> None:
    job.status = "running"
    job.emit({"type": "status", "message": "queued behind any active job" if _run_lock.locked() else "starting"})
    with _run_lock:
        try:
            job.emit(
                {
                    "type": "status",
                    "message": (
                        f"generate n={spec.n} descriptors={', '.join(spec.names())} "
                        f"pop={population} gens={generations}"
                    ),
                }
            )
            generate_molecules(
                spec,
                population_size=population,
                generations=generations,
                seed=seed,
                prior="transformer",
                on_progress=job.emit,
            )
            if job.status != "done":
                job.emit(
                    {
                        "type": "done",
                        "message": "finished",
                        "hits": job.hits,
                        "manifest": job.manifest,
                    }
                )
        except Exception as exc:
            traceback.print_exc()
            job.emit({"type": "error", "message": str(exc)})
