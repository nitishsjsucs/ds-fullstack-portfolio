.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup data train train-quick api web dev test audit docs shots verify clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install pinned dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt
	cd frontend && npm install

data:  ## Download the eight datasets and write data/MANIFEST.json
	$(PY) scripts/fetch_data.py

train:  ## Train every project (~1 hour)
	$(PY) scripts/train_all.py

train-quick:  ## Train every project at reduced fidelity (~4 minutes)
	$(PY) scripts/train_all.py --quick

api:  ## Serve the API on :8000
	$(PY) -m uvicorn app.main:app --app-dir backend --reload --port 8000

web:  ## Serve the console on :5173
	cd frontend && npm run dev

dev:  ## Run the API and the console together
	@$(MAKE) -j2 api web

test:  ## Run the test suite
	$(PY) -m pytest backend/tests -q

audit:  ## Static leakage scan + artifact aggregation -> AUDIT_REPORT.md
	$(PY) scripts/audit.py

docs:  ## Regenerate the papers and the README table from the artifacts
	$(PY) scripts/generate_docs.py

shots:  ## Capture every screen (needs the API and console running)
	$(PY) scripts/capture_screenshots.py

verify: test audit  ## Everything a reviewer should run
	cd frontend && npm run build

clean:  ## Remove artifacts and build output (data and venv are kept)
	rm -rf backend/artifacts frontend/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
