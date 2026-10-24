"""Fusion of CCTV events, 112 calls and operator reviews into incidents (phase 4, D-028).

Modules: ``config`` (typed YAML), ``triage`` (rule and optional offline-LLM call triage),
``model`` (evidence rates and posterior), ``engine`` (association and incident updates),
``replay`` (scripted acceptance scenarios). The synthetic call and camera-event processes live in
``sim`` (``sim.calls``, ``sim.detection``) because only ``sim`` may read ground truth.
"""
