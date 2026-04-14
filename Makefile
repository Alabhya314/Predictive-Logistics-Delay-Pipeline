.PHONY: up down ingest validate transform train evaluate test clean logs

# ── Docker operations ──────────────────────────────────────────────────────
up:
	docker-compose up --build -d
	@echo ""
	@echo "✓ Services starting..."
	@echo "  Airflow  → http://localhost:8080  (admin / admin)"
	@echo "  MLflow   → http://localhost:5000"

down:
	docker-compose down -v

logs:
	docker-compose logs -f --tail=50

# ── Run individual pipeline stages locally ─────────────────────────────────
ingest:
	PYTHONPATH=src python src/ingestion/ingest.py --n-rows 5000

validate: ingest
	PYTHONPATH=src python src/validation/validate.py \
		$$(ls -t data/raw/*.parquet | head -1)

transform: validate
	PYTHONPATH=src python src/features/engineer.py \
		$$(ls -t data/validated/*.parquet | head -1)

train: transform
	PYTHONPATH=src python src/training/train.py \
		$$(ls -t data/processed/*.parquet | head -1)

evaluate: train
	PYTHONPATH=src python src/training/evaluate.py \
		$$(ls -t data/processed/*.parquet | head -1)

# Run the full pipeline end-to-end locally (no Docker)
pipeline: ingest validate transform train evaluate
	@echo "✓ Pipeline complete"

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=src pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Utilities ──────────────────────────────────────────────────────────────
clean:
	rm -rf data/raw data/validated data/processed data/models data/alerts
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
