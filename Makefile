# ResQCity Makefile. Every target delegates to tools/tasks.py, so on machines without GNU make
# you can run the same thing with:  uv run python -m tools.tasks <target>

TASKS    := uv run python -m tools.tasks
SCENARIO ?= normal
POLICY   ?= B0

.PHONY: setup lint format test types data streams api ui demo sim experiments figures reproduce

setup:
	uv sync
	$(TASKS) setup

lint format test types data streams api ui demo experiments figures reproduce:
	$(TASKS) $@

sim:
	$(TASKS) sim --scenario $(SCENARIO) --policy $(POLICY)
