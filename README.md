# cs290-sp26 — Autonomous Driving with Multi-Agent Lane Merging

**Shanaya Malik, Nathan McNaughton**
ELENG 290: Learning-Enabled Multi-Agent Systems — Spring 2026

Extension of Sadigh et al. (2016) to multi-vehicle lane merging with 3+ agents. Each human driver is assigned a predefined reward archetype (cautious / normal / aggressive). The pipeline has three stages: iterative best-response MPC expert → behavioral cloning → PPO fine-tuning.

See [implementation_guide.md](implementation_guide.md) for the full step-by-step implementation plan.
See [report_notes.md](report_notes.md) for things to address in the final paper writeup.
