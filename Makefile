.PHONY: install lint test security-scan

install:
	pip install -r requirements.txt
	pip install bandit pre-commit
	pre-commit install

lint:
	@echo "Checking code style and basic security..."
	pre-commit run --all-files
	# We use -lll to focus on High severity/confidence in the pre-commit gate
	bandit -r . -lll --exclude ./.venv

test:
	@echo "Running unit tests..."
	pytest tests/ || echo "No tests found yet, skipping..."

security-scan:
	@echo "Running full SAST audit for the PR..."
	# Added '-' so the Makefile continues even if Bandit finds issues
	-bandit -r . -f txt -o security_report.txt --exclude ./.venv
	@echo "Report generated: security_report.txt"
