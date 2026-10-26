"""Ward-hour demand forecasting (phase 5, D-030).

The panel itself is built in ``sim.panel`` because it reads the generator (ground truth) and
``tests/test_architecture.py`` forbids ``forecasting -> sim``. This package holds the typed
config, the models and the evaluation, all of which read a saved panel.
"""
