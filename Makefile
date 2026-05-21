.PHONY: help pipeline api dashboard

API_HOST ?= 0.0.0.0
API_PORT ?= 8000

help:
	@echo "Available targets:"
	@echo "  make pipeline    Run ingest -> preprocess -> features"
	@echo "  make api         Start FastAPI on $(API_HOST):$(API_PORT) with --reload"
	@echo "  make dashboard   Launch the Streamlit dashboard"

pipeline:
	uv run python -m scf_agent.pipeline.ingest
	uv run python -m scf_agent.pipeline.preprocess
	uv run python -m scf_agent.pipeline.features

api:
	uv run uvicorn scf_agent.api.main:app --reload --host $(API_HOST) --port $(API_PORT)

dashboard:
	uv run streamlit run dashboard/app.py
