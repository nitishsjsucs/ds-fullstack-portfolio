"""FastAPI application: one server, eight projects.

Routing model
-------------
Most of what a project exposes is a *pinned training artifact*, so the bulk of the
surface is generic:

    GET  /api/projects                    navigation index
    GET  /api/projects/{slug}             metadata + provenance
    GET  /api/projects/{slug}/{payload}   eda | crispdm | leaderboard | evaluation |
                                          explain | model_card | audit | extras
    POST /api/projects/{slug}/predict     live inference

Anything genuinely project-specific (streaming text generation, a rule query, an
alert-budget simulator) lives in that project's own ``router``, mounted under the
same prefix. This keeps eight dashboards consistent without forcing every project
into an identical mould.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import registry
from .core import paths
from .core.artifacts import PAYLOADS, ArtifactStore
from .core.serialization import jsonable

_stores: dict[str, ArtifactStore] = {}
_boot: dict[str, Any] = {}


def store_for(slug: str) -> ArtifactStore:
    if slug not in _stores:
        _stores[slug] = ArtifactStore(slug)
    return _stores[slug]


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    trained = [s for s in registry.SLUGS if store_for(s).is_trained]
    _boot.update({
        "started_at": time.time(),
        "trained_projects": trained,
        "untrained_projects": [s for s in registry.SLUGS if s not in trained],
        "boot_ms": round((time.perf_counter() - t0) * 1000, 2),
    })
    banner = f"{len(trained)}/{len(registry.SLUGS)} projects have artifacts"
    print(f"[startup] {banner} ({_boot['boot_ms']} ms)")
    if _boot["untrained_projects"]:
        print(f"[startup] not yet trained: {', '.join(_boot['untrained_projects'])} "
              f"-- run `python scripts/train_all.py`")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Data Science Full Stack Portfolio",
        version="1.0.0",
        description=(
            "Eight end-to-end data science systems over eight real public datasets, "
            "each following CRISP-DM with a leakage audit."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        # The Vite dev server and any localhost preview build.
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = APIRouter(prefix="/api")

    # ---------------------------------------------------------------- meta --
    @api.get("/health", tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok",
            "projects": len(registry.PROJECTS),
            "trained": _boot.get("trained_projects", []),
            "untrained": _boot.get("untrained_projects", []),
            "uptime_s": round(time.time() - _boot.get("started_at", time.time()), 1),
        }

    @api.get("/projects", tags=["meta"])
    def list_projects() -> dict:
        return jsonable(registry.index_payload())

    @api.get("/datasets", tags=["meta"])
    def datasets() -> dict:
        """Provenance for every curated dataset: source, licence, sha256, curation rule."""
        if not paths.MANIFEST.exists():
            raise HTTPException(404, "data/MANIFEST.json missing -- run scripts/fetch_data.py")
        return json.loads(paths.MANIFEST.read_text())

    @api.get("/audit", tags=["meta"])
    def portfolio_audit() -> dict:
        """The cross-project leakage & methodology audit, if it has been generated."""
        path = paths.ARTIFACTS / "_portfolio_audit.json"
        if not path.exists():
            raise HTTPException(404, "Audit not generated -- run scripts/audit.py")
        return json.loads(path.read_text())

    # ------------------------------------------------------------ projects --
    @api.get("/projects/{slug}", tags=["projects"])
    def project_detail(slug: str) -> dict:
        try:
            meta = registry.get(slug)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        store = store_for(slug)
        return jsonable({
            **meta.to_dict(),
            "trained": store.is_trained,
            "provenance": store.provenance(),
            "available_payloads": [
                name for name in PAYLOADS if (store.dir / f"{name}.json").exists()
            ],
            "has_live_inference": store.has_model(),
        })

    @api.get("/projects/{slug}/{payload}", tags=["projects"])
    def project_payload(slug: str, payload: str) -> Any:
        try:
            registry.get(slug)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if payload not in PAYLOADS:
            raise HTTPException(
                404, f"Unknown payload {payload!r}. Available: {', '.join(PAYLOADS)}"
            )
        store = store_for(slug)
        if not store.is_trained:
            raise HTTPException(
                503,
                f"Project '{slug}' has no artifacts yet. "
                f"Run `python scripts/train_all.py --only {slug}`.",
            )
        data = store.payload(payload)
        if data is None:
            raise HTTPException(404, f"Project '{slug}' does not publish a '{payload}' payload.")
        return data

    @api.post("/projects/{slug}/predict", tags=["inference"])
    def predict(slug: str, body: dict) -> Any:
        try:
            registry.get(slug)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        try:
            mod = registry.module(slug)
        except Exception as exc:
            raise HTTPException(500, f"Could not import project '{slug}': {exc}") from exc
        fn = getattr(mod, "predict", None)
        if fn is None:
            raise HTTPException(405, f"Project '{slug}' does not expose live inference.")
        t0 = time.perf_counter()
        try:
            result = fn(body or {})
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, f"Invalid input: {exc}") from exc
        latency = round((time.perf_counter() - t0) * 1000, 2)
        if isinstance(result, dict):
            result = {**result, "latency_ms": latency}
        return jsonable(result)

    # Project-specific routers are mounted BEFORE the generic router, because
    # FastAPI matches routes in registration order and `/projects/{slug}/{payload}`
    # would otherwise swallow every one of them -- `/projects/basket/items` would
    # resolve to the catch-all and 404 with "Unknown payload 'items'". Specific
    # before general is the rule whenever a path parameter can shadow a literal.
    for meta in registry.PROJECTS:
        try:
            mod = registry.module(meta.slug)
        except Exception as exc:  # a broken project must not take down the server
            print(f"[startup] project '{meta.slug}' failed to import: {exc}")
            continue
        extra = getattr(mod, "router", None)
        if extra is not None:
            app.include_router(extra, prefix=f"/api/projects/{meta.slug}",
                               tags=[meta.slug])

    app.include_router(api)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # pragma: no cover
        return JSONResponse(
            status_code=500,
            content={"detail": f"{type(exc).__name__}: {exc}", "path": str(request.url.path)},
        )

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": "Data Science Full Stack Portfolio API",
            "docs": "/docs",
            "projects": "/api/projects",
        }

    return app


app = create_app()
