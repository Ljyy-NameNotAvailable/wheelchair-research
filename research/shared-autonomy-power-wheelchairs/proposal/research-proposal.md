# Research Proposal
## User Adaptation to Shared Control Assistance in Powered Wheelchairs

**Submitted to:** Professor Marco Brocanelli
**Date:** April 3, 2026

---

### Background and Motivation

Shared autonomy — where control authority is divided between a human operator and an autonomous system — is the dominant paradigm in assistive wheelchair robotics. A substantial body of literature demonstrates that shared control reduces collisions and cognitive load compared to unassisted manual operation (Carlson & Demiris, 2012; Javdani et al., 2018). However, virtually all existing intent prediction algorithms share a critical assumption: that user input behavior remains stable across conditions.

A recent study by Aronson & Short (2024) directly challenges this assumption, providing preliminary evidence that users *adapt* their joystick inputs when assistance is active — effectively co-evolving with the system. This has a significant practical consequence: intent prediction models trained on unassisted behavior may degrade upon deployment, because the input distribution shifts as users adapt. This phenomenon has not been systematically characterized, and no study has examined whether adaptation compounds over repeated exposure.

---

### Research Question

> **Do users systematically alter their joystick input behavior when shared control assistance is active, and does this adaptation grow over repeated sessions?**

---

### Proposed Study

**Design:** Within-subjects, repeated measures (2 sessions, one week apart)

**Participants:** 14–16 able-bodied adults (power analysis: 80% power, medium effect, α = .05); 3–4 manual wheelchair users recruited separately for qualitative contrast

**Platform:** Instrumented powered wheelchair or wheelchair simulator with loggable joystick input and configurable shared control (e.g., CoNav Chair, Xu et al., 2025)

**Conditions:** Each participant completes three conditions in both sessions (order counterbalanced):

| Condition | Autonomy weight |
|-----------|----------------|
| Manual control | 0% |
| Low shared control | ~30% |
| High shared control | ~70% |

**Task:** A standardized indoor navigation circuit — straight corridor, doorway passage, obstacle field — repeated three times per condition.

**Measures:**

*Primary (intent signal characteristics):*
- Joystick input magnitude
- Direction change frequency
- Input onset latency
- Yielding events (user releases joystick mid-maneuver)

*Secondary:*
- Path efficiency, collision count, task completion time
- NASA-TLX (cognitive load)
- Control satisfaction and trust (Jian et al., 2000)

**Analysis:** Repeated-measures ANOVA (condition × session × task difficulty); η² reported for all effects; 5-minute post-session interview for qualitative triangulation.

---

### Hypotheses

- **H1:** Users show reduced input magnitude and increased onset latency under shared control vs. manual control.
- **H2:** This behavioral shift is larger in Session 2 than Session 1, indicating adaptation compounds over time.
- **H3:** High-autonomy assistance produces greater behavioral shift than low-autonomy assistance.

---

### Expected Contribution

If confirmed, these findings will establish that user intent signals are not a stable input to be predicted, but a *dynamic variable that co-evolves with the assistance system*. This directly motivates a new class of adaptive intent prediction algorithms that account for user behavioral drift — a concrete direction for subsequent engineering work. If not confirmed, the null result is equally informative: it would validate the static-user assumption and support the robustness of existing algorithms.

---

### Scope and Feasibility

This study requires no novel hardware and no new algorithm development. It requires access to a powered wheelchair or simulator, standard data logging, and participants. All measures use validated instruments. The two-session protocol is completable within a single semester.

---

### Limitations

- Able-bodied proxy participants limit direct generalizability to clinical populations; this is appropriate for a first characterization study and explicitly scoped as future work.
- Two sessions capture only short-term adaptation; longitudinal effects over weeks or months remain an open question.
- Findings are specific to joystick input; transfer to other modalities (BCI, gaze) is not assumed.

---

### References

Aronson, R., & Short, E. S. (2024). Intentional user adaptation to shared control assistance. *ACM/IEEE HRI*.

Carlson, T., & Demiris, Y. (2012). Collaborative control for a robotic wheelchair. *IEEE Transactions on Systems, Man, and Cybernetics*, 42(3), 876–888.

Javdani, S., Admoni, H., Pellegrinelli, S., Srinivasa, S. S., & Bagnell, J. A. (2018). Shared autonomy via hindsight optimization. *International Journal of Robotics Research*, 37(13–14), 1515–1532.

Xu, J., et al. (2025). CoNav Chair: Development and evaluation of a shared control based wheelchair for the built environment. *arXiv preprint*.
