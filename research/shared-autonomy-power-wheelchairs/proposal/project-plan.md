# Project Plan
## Edge Deployment of LLM-Based Semantic Intent Inference for Shared-Control Wheelchairs

**Course:** Edge Computing
**Team:** Jerry (PhD), Kris (Master's), Jinyang (Master's)
**Date:** April 3, 2026

---

## Team Roles

| Member | Primary Responsibility |
|--------|----------------------|
| **Jerry** | LLM pipeline; model quantization and distillation; methodology writing |
| **Kris** | Benchmarking harness; on-device profiling; results and plots |
| **Jinyang** | Adaptive offloading policy; network simulation; evaluation pipeline; evaluation writing |

---

## Milestones Overview

| Phase | Weeks | Milestone |
|-------|-------|-----------|
| 1 — Setup & Baseline | 1–3 | Pipeline running end-to-end on Nano; baseline latency number in hand |
| 2 — Compression | 4–6 | Latency-accuracy curve for on-device model variants |
| 3 — Offloading Policy | 7–9 | Adaptive policy running under simulated network conditions |
| 4 — Evaluation & Writing | 10–13 | Full benchmark results; paper draft to Professor Brocanelli |
| 5 — Polish & Submission | 14 | Final paper + reproducible code repo |

---

## Phase 1 — Setup & Baseline (Weeks 1–3)

**Goal:** Everything running end-to-end, even if slowly.

| Task | Owner | Done When |
|------|-------|-----------|
| Containerize LLM pipeline (Docker/Conda) for reproducible runs on Nano and cloud | Jerry | Pipeline runs identically on both environments |
| Set up benchmarking harness on Jetson Orin Nano; measure raw latency, power (`tegrastats`), memory | Kris | Baseline numbers logged for full-precision model |
| Provision cloud endpoint (TBD); measure round-trip latency Jetson → cloud → Jetson | Jinyang | RTT measured under stable network conditions |
| Agree on "acceptable accuracy" threshold (e.g., IoU ≥ X vs. full-precision baseline) | All | Written down and approved by Professor Brocanelli |

**End-of-phase check-in:** Present baseline latency number to Professor Brocanelli — even if it's 800ms, this is the starting point.

---

## Phase 2 — Compression Experiments (Weeks 4–6)

**Goal:** Find how small the model can go on-device without falling below the accuracy threshold.

| Task | Owner | Done When |
|------|-------|-----------|
| Produce 2–3 quantized/distilled model variants (FP16 → INT8 → smaller distilled model) | Jerry | Variants saved and versioned |
| Build ground-truth evaluation set — annotated scenes for scoring segmentation accuracy | Jinyang | ≥50 scenes annotated with ground-truth relative position labels |
| Benchmark each model variant on Nano: latency (p50/p95/p99), accuracy (IoU, F1), power draw | Kris | Results table complete for all variants |
| Plot latency-accuracy-energy frontier | Kris | Chart ready for paper |

**End-of-phase check-in:** You now know the on-device ceiling. If no variant meets 100ms, offloading is confirmed as necessary — H1 validated.

---

## Phase 3 — Offloading Policy (Weeks 7–9)

**Goal:** Build and test the adaptive partitioning logic.

| Task | Owner | Done When |
|------|-------|-----------|
| Design rule-based offloading policy: if RTT > threshold → run local model; else → offload | Jinyang | Policy logic documented and implemented |
| Simulate network conditions on Jetson using `tc netem` (RTT: 10/50/100/200ms; packet loss: 0/2/5%) | Jinyang | 5 network scenarios reproducible on demand |
| Integrate offloading policy between sensor input and LLM call | Jerry + Jinyang | System switches strategies dynamically during a run |
| Benchmark adaptive policy under all network scenarios; compare against fixed on-device and fixed cloud | Kris | Results table: strategy × network condition × latency |

**End-of-phase check-in:** Core experiment complete. All three strategies benchmarked. If H3 holds, adaptive partitioning is your contribution.

---

## Phase 4 — Evaluation & Writing (Weeks 10–13)

**Goal:** Produce the final results and paper draft.

| Task | Owner | Done When |
|------|-------|-----------|
| Run full benchmark suite (all strategies × all network conditions × all model configs) | All | Numbers finalized; no re-runs needed |
| Produce final figures: latency CDF, accuracy bar chart, energy comparison, policy decision trace | Kris | All figures publication-ready |
| Write: Abstract, Background, Related Work | Jerry | Draft reviewed by Kris and Jinyang |
| Write: Methodology, System Design | Jerry + Jinyang | Draft reviewed by all |
| Write: Evaluation, Results, Discussion | Kris + Jinyang | Draft reviewed by all |
| Submit full draft to Professor Brocanelli | All | Submitted by end of Week 13 |

---

## Phase 5 — Polish & Submission (Week 14)

| Task | Owner |
|------|-------|
| Incorporate Professor Brocanelli's feedback | All |
| Clean up code repo; write README with reproduction steps | Kris |
| Final proofread | Jinyang |
| Submit | Jerry |

---

## Key Decisions (Resolve in Week 1)

| Decision | Options | Impact |
|----------|---------|--------|
| Cloud platform | Lab server / AWS / GCP / Azure | Affects latency baseline, cost, and network variability |
| Base LLM for Jerry's pipeline | Phi-3 mini, Qwen2.5-0.5B, LLaMA 3.2 1B, etc. | Determines quantization toolchain (TensorRT / llama.cpp / GGUF) |
| Offloading policy type | Rule-based (safe) vs. learned (more publishable) | Scope and complexity of Phase 3 |
| 100ms latency threshold | Validate with Professor Brocanelli | Defines pass/fail for H1 and H2 |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Model too large to quantize meaningfully on Nano | Medium | Scope to a smaller base model from the start (≤1B parameters) |
| Cloud RTT too variable to be a reliable offload target | Medium | Use a lab server with controlled network rather than public cloud |
| Network simulation doesn't reflect real wheelchair environment | Low-Medium | Document as a limitation; real-world measurement is future work |
| Writing falls behind experiment schedule | Medium | Assign writing owners in Phase 2; don't leave it all to Phase 4 |

---

## Communication

- **Weekly sync:** All three meet once a week to share blockers and update progress
- **Shared repo:** All code, benchmarking scripts, and results go into the GitHub repo
- **Decision log:** Any change to scope or methodology gets a one-line note in this document

---

## Open Questions for Professor Brocanelli

1. Which cloud platform does the lab have access to?
2. Is 100ms the right latency threshold, or does he know a more grounded figure from the wheelchair control literature?
3. Rule-based vs. learned offloading policy — does he have a preference given the semester scope?
4. Does the course require a live demo, or is a paper + reproducible benchmark sufficient?
