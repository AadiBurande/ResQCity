# ResQCity Makefile. Every target delegates to src/resqcity/tools/tasks.py, so on machines without GNU make
# you can run the same thing with:  uv run python -m resqcity.tools.tasks <target>

TASKS    := uv run python -m resqcity.tools.tasks
SCENARIO ?= normal
POLICY   ?= B0
REPS     ?= 30

.PHONY: setup lint format test types data vision fusion forecast streams api ui demo sim experiments figures reproduce

setup:
	uv sync
	$(TASKS) setup

lint format test types data vision forecast streams api ui demo experiments figures reproduce:
	$(TASKS) $@

fusion:
	$(TASKS) fusion --scenario $(SCENARIO) --reps $(REPS)

sim:
	$(TASKS) sim --scenario $(SCENARIO) --policy $(POLICY)
