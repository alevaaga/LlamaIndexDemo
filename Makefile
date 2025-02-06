include .env
export

install:
	pdm install

build: install
	docker compose build

build-debug: install
	docker compose build --progress plain

build-clean: install
	docker compose  build --no-cache --progress plain

run:
	docker compose up

enter: run
	docker compose exec schmopilot-api bash
