# Research Proposal
## Edge Deployment of LLM-Based Semantic Intent Inference for Shared-Control Wheelchairs

**Submitted to:** Professor Marco Brocanelli
**Course:** Edge Computing
**Date:** April 3, 2026

---

### Background and Motivation

Shared autonomy in powered wheelchairs requires real-time inference of user intent from sensor input — typically geometric data such as occupancy grids or point clouds. A limitation of purely geometric approaches is that they lack semantic context: a doorway and a narrow gap between obstacles may look identical to a distance sensor but carry entirely different navigational meaning to the user.

Large language models (LLMs) and vision-language models (VLMs) have recently demonstrated strong performance on spatial reasoning and semantic scene description tasks. Our team has developed an LLM-based pipeline that generates structured relative position descriptions from sensor input (e.g., *"obstacle 1.2m ahead-left, doorway 2m ahead-right"*), providing a richer semantic representation than geometric-only methods. However, deploying this pipeline on a wheelchair — an edge device with strict latency, power, and memory constraints — introduces substantial systems challenges that have not been addressed in the literature.

Shared autonomy control loops require sensor-to-decision latencies on the order of 100ms or less to feel responsive to the user (Javdani et al., 2018; Carlson & Demiris, 2012). Current LLM inference, even for small models, typically exceeds this budget on commodity edge hardware. This creates a fundamental tension: semantic representations improve intent inference quality, but the computation required to produce them may make the system unusable in practice.

---

### Research Question

> **What edge deployment strategy — on-device inference, cloud offloading, or adaptive partitioning — best satisfies the latency and accuracy requirements of LLM-based semantic intent inference for real-time shared-control wheelchair navigation?**

---

### Proposed Study

**Approach:** Benchmark and compare three deployment strategies for the LLM semantic segmentation pipeline across two edge hardware platforms, measuring latency, accuracy, and energy consumption under realistic network conditions.

---

#### Deployment Strategies Under Comparison

| Strategy | Description |
|----------|-------------|
| **On-device** | Full LLM inference runs locally on the wheelchair's edge hardware |
| **Cloud offload** | Inference offloaded to a remote server; edge device handles only sensing and control |
| **Adaptive partitioning** | System decides per-frame whether to run locally or offload based on current network RTT and queue depth |

---

#### Hardware Platforms

- **NVIDIA Jetson Orin Nano** — representative high-end wheelchair edge platform
- **Raspberry Pi 5** — representative low-cost edge platform

---

#### LLM Configurations Tested (On-Device)

To characterize the accuracy-latency frontier for on-device inference:

| Config | Model | Quantization |
|--------|-------|-------------|
| A | Full model (baseline) | None (FP16) |
| B | Full model | INT8 quantization |
| C | Distilled / smaller model | INT8 quantization |

---

#### Metrics

**Latency:**
- End-to-end inference time (sensor input → relative position output)
- 50th, 95th, 99th percentile latency over 500 frames
- Target threshold: ≤100ms for shared control responsiveness

**Accuracy:**
- Semantic segmentation quality vs. full-precision cloud baseline (IoU, F1)
- Relative position description correctness (evaluated against ground-truth annotated scenes)

**Energy:**
- Power draw (watts) during inference on each platform
- Estimated battery impact per hour of operation

**Network sensitivity (offloading strategies):**
- Latency under varying simulated network RTT: 10ms, 50ms, 100ms, 200ms, packet loss 0–5%
- Offloading decision accuracy for adaptive partitioning policy

---

### Hypotheses

- **H1:** On-device inference on Jetson Orin Nano with INT8 quantization meets the 100ms latency threshold; Raspberry Pi 5 does not.
- **H2:** Cloud offloading meets the latency threshold only under low-RTT conditions (≤50ms); it degrades under realistic mobile network variability.
- **H3:** Adaptive partitioning achieves lower 95th-percentile latency than either fixed strategy across varied network conditions.

---

### Team Contributions

| Member | Contribution |
|--------|-------------|
| Jerry (PhD) | LLM semantic segmentation pipeline (existing); model distillation and quantization experiments |
| Master's student 1 | Edge deployment, benchmarking infrastructure, on-device profiling |
| Master's student 2 | Adaptive partitioning policy design; network simulation; evaluation pipeline |

---

### Expected Contribution

This work produces three concrete outputs:

1. **Benchmark results** characterizing the latency-accuracy-energy tradeoff for LLM-based semantic inference on wheelchair-class edge hardware — a dataset and baseline the community currently lacks.
2. **An adaptive offloading policy** for LLM inference in latency-sensitive assistive robotics, with decision criteria derived empirically rather than assumed.
3. **A deployment recommendation** for practitioners: which hardware and strategy combination is viable for real-time shared autonomy today, and what remains out of reach.

---

### Feasibility

All components are in hand: Jerry's LLM pipeline is operational, edge hardware is commercially available, and network simulation requires no specialized infrastructure. The benchmark is reproducible and does not require human subjects, IRB approval, or clinical access. Results are achievable within a single semester.

---

### Limitations

- Evaluation uses pre-recorded or simulated scenes rather than a live wheelchair deployment; end-to-end closed-loop testing remains future work.
- Network conditions are simulated, not measured in a real hospital or home environment.
- User-facing evaluation (does semantic intent inference actually improve navigation outcomes?) is out of scope for this study and constitutes a natural follow-on.

---

### References

Carlson, T., & Demiris, Y. (2012). Collaborative control for a robotic wheelchair. *IEEE Transactions on Systems, Man, and Cybernetics*, 42(3), 876–888.

Javdani, S., Admoni, H., Pellegrinelli, S., Srinivasa, S. S., & Bagnell, J. A. (2018). Shared autonomy via hindsight optimization. *International Journal of Robotics Research*, 37(13–14), 1515–1532.

Reddy, S., Dragan, A., & Levine, S. (2018). Shared autonomy via deep reinforcement learning. *Robotics: Science and Systems*.

Xu, J., et al. (2025). CoNav Chair: Development and evaluation of a shared control based wheelchair for the built environment. *arXiv preprint*.
