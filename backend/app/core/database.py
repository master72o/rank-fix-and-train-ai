"""Async SQLAlchemy engine, session factory and schema bootstrap."""

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None, *, null_pool: bool = False) -> AsyncEngine:
    kwargs: dict = {}
    if null_pool:
        kwargs["poolclass"] = NullPool
    return create_async_engine(url or get_settings().database_url, **kwargs)


def make_session_factory(bind: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind, expire_on_commit=False)


engine = make_engine()
SessionLocal = make_session_factory(engine)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def init_db(target: AsyncEngine | None = None) -> None:
    """Create tables (used for SQLite/dev). Production Postgres uses Alembic migrations."""
    from app.models import domain  # noqa: F401  (register models)

    async with (target or engine).begin() as conn:
        if conn.dialect.name == "postgresql":
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
