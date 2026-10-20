"""AlignForge FastAPI main application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.annotations import router as annotations_router
from app.api.v1.judge import router as judge_router
from app.api.v1.prompts import router as prompts_router
from app.api.v1.training import router as training_router
from app.core.config import get_settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.auto_create_tables:
        await init_db()
    yield


settings = get_settings()

app = FastAPI(
    title=f"{settings.app_name} API",
    version="1.0.0",
    description="Active-learning RLHF/DPO human annotation workbench, LLM-as-a-Judge audit, and one-click DPO training.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prompts_router, prefix=settings.api_prefix)
app.include_router(annotations_router, prefix=settings.api_prefix)
app.include_router(judge_router, prefix=settings.api_prefix)
app.include_router(training_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "tagline": "Rank, Fix, and Train AI",
        "docs_url": "/docs",
        "api_v1_prefix": settings.api_prefix,
        "status": "online",
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
