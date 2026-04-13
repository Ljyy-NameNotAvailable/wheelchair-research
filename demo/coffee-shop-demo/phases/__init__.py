"""
phases/ — Phase modules for the coffee-shop demo run plan.

Modules:
  common        — shared paths, state I/O, logging helpers
  phase0_baseline   — baseline metrics + Roboflow crop upload
  phase1_dataset    — Oxford Town Centre download + dataset merge
  phase2_finetune   — YOLOv8n fine-tune (Apple MPS or CPU)
  phase3_evaluate   — re-evaluate + comparison report
"""
