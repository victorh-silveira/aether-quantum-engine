.PHONY: test lint train run infra-up infra-down

test:
	cd app && PYTHONPATH=src python -m pytest tests --cov=src --cov-report=term-missing --cov-fail-under=100

lint:
	cd app && python -m ruff check src tests run.py train.py ../run.py ../train.py
	cd app && python -m ruff format --check src tests run.py train.py ../run.py ../train.py

train:
	python app/train.py

run:
	python run.py

infra-up:
	docker compose -f infra/docker/docker-compose.yml up -d

infra-down:
	docker compose -f infra/docker/docker-compose.yml down
