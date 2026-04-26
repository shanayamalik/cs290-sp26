# Autonomous Driving with Multi-Agent Lane Merging

**Shanaya Malik, Nathan McNaughton**
ELENG 290: Learning-Enabled Multi-Agent Systems — Spring 2026

Extension of Sadigh et al. (2016) to multi-vehicle lane merging with 3+ agents. Each human driver is assigned a predefined reward archetype (cautious / normal / aggressive). The pipeline has three stages: iterative best-response MPC expert → behavioral cloning → PPO fine-tuning.

See [implementation_guide.md](implementation_guide.md) for the full step-by-step implementation plan.
See [report_notes.md](report_notes.md) for things to address in the final paper writeup.

---

## Setup

**Requirements:** Python 3.10+, no GPU needed.

```bash
# 1. Clone the repo
git clone https://github.com/shanayamalik/cs290-sp26.git
cd cs290-sp26

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

To verify the environment is working:

```bash
python3 -c "import highway_env, gymnasium, torch, stable_baselines3; print('All dependencies OK')"
```

Then run the environment sanity check (a render window with cars should appear, and the terminal will print the observation shape and vehicle states):

```bash
python3 merge_env.py
```
