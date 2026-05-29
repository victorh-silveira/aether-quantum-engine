APP_DIR=app
PYTHON=python

.PHONY: install lint test security run backtest pre-commit

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(APP_DIR)/requirements.txt -r $(APP_DIR)/requirements-dev.txt

lint:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage lint

test:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage test

security:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage security

run:
	$(PYTHON) run.py

backtest:
	$(PYTHON) $(APP_DIR)/scripts/backtest/medallion_backtest.py $(ARGS)

pre-commit:
	bash linters/git-hooks/install.sh
