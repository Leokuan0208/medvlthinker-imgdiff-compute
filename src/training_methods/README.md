# Training-based methods (SEPARATE from training-free cascade work)

This folder holds methods that REQUIRE TRAINING (trained routers/deferral nets, distillation,
LoRA fine-tuning, draft heads for speculative decoding, learned verifiers/pruners, etc.).
Kept separate from the training-free cascade code in src/cascade_methods/ (do not mix).
Results here are used as ABLATIONS toward / against the training-free ACC method.

Constraints (2026-06-19): network throttled ~1 day -> NO new model downloads; use only LOCALLY
cached models (MedVLThinker-7B/32B local; InternVL2.5-8B, Phi-3.5-V, SigLIP cached) + local train
splits (/data/dan/dataset/*). Keep main drive <80%; checkpoints -> /data, prefer small adapters.
