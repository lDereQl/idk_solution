COMPOSE := docker compose
PY := .venv/bin/python
PORT ?= 8000
export PORT

.DEFAULT_GOAL := help

.PHONY: help install data web-build run up down logs api-dev test

help:
	@echo "Setup:     install data"
	@echo "Docker:    run up down logs"
	@echo "Local dev: web-build api-dev"
	@echo "Quality:   test"
	@echo "Page:      http://localhost:$(PORT)/demo   (PORT=8001 make up to use another port)"

install:
	uv venv --allow-existing .venv
	uv pip install --python $(PY) -r requirements.txt

data:
	$(PY) scripts/make_stub_data.py

web-build:
	$(PY) web/build.py

run:
	$(COMPOSE) up -d --build --wait
	@echo "Presentation: http://localhost:$(PORT)/demo"
	@(xdg-open http://localhost:$(PORT)/demo || open http://localhost:$(PORT)/demo) >/dev/null 2>&1 &

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

api-dev: web-build
	DEV_CORS=1 $(PY) -m uvicorn app.main:app --reload --port $(PORT)

test:
	$(PY) -m pytest
