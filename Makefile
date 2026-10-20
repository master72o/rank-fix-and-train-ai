.PHONY: help install dev test seed up down lint clean

help:
	@echo "AlignForge — Rank, Fix, and Train AI"
	@echo ""
	@echo "Usage:"
	@echo "  make install    Install backend and frontend dependencies"
	@echo "  make dev        Run local backend API and frontend dev servers"
	@echo "  make test       Run backend test suite (IAA metrics, Active Learning)"
	@echo "  make seed       Seed the 40 golden benchmark prompts"
	@echo "  make up         Start all services with Docker Compose"
	@echo "  make down       Stop Docker Compose services"
	@echo "  make lint       Run linters (Ruff)"
	@echo "  make clean      Remove build artifacts and cache"

install:
	cd backend && pip install -e ".[dev]"
	cd frontend && npm install

dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

seed:
	cd backend && python scripts/seed_golden_dataset.py

test:
	cd backend && pytest -v

lint:
	cd backend && ruff check .

up:
	docker compose up --build -d

down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf backend/artifacts
