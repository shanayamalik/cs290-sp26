"""
Generate the Highway-Env merge-v0 scenario diagram for the paper.
Saves to figures/merge_diagram.pdf and figures/merge_diagram.png.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch as FBP

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

os.makedirs('figures', exist_ok=True)

# ── canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 22)
ax.set_ylim(-1.2, 7.5)
ax.set_aspect('equal')
ax.axis('off')

# ── road geometry ────────────────────────────────────────────────────────────
ROAD_LEFT   = 0.5
ROAD_RIGHT  = 21.5
LANE_W      = 1.6     # lane half-width
LANE0_Y     = 0.0     # bottom lane (NPC Cautious)
LANE1_Y     = LANE_W * 2   # upper highway lane (EGO)
ROAD_TOP    = LANE1_Y + LANE_W
ROAD_BOT    = LANE0_Y - LANE_W

# road fill
road = plt.Polygon(
    [[ROAD_LEFT, ROAD_BOT], [ROAD_RIGHT, ROAD_BOT],
     [ROAD_RIGHT, ROAD_TOP], [ROAD_LEFT, ROAD_TOP]],
    closed=True, color='#b0b0b0', zorder=0)
ax.add_patch(road)

# road edges
for y in [ROAD_BOT, ROAD_TOP]:
    ax.plot([ROAD_LEFT, ROAD_RIGHT], [y, y], color='#333333', lw=2.5, zorder=1)

# lane divider (dashed)
ax.plot([ROAD_LEFT, ROAD_RIGHT], [LANE0_Y + LANE_W, LANE0_Y + LANE_W],
        color='white', lw=1.5, ls='--', dashes=(8, 6), zorder=1)

# lane 0 center dashes (broken line)
for x in np.arange(ROAD_LEFT + 0.3, ROAD_RIGHT, 1.6):
    ax.plot([x, x + 0.9], [LANE0_Y, LANE0_Y],
            color='#cccccc', lw=1.2, ls='--', dashes=(4, 3), zorder=1)

# lane labels
ax.text(ROAD_LEFT - 0.05, LANE1_Y, 'Lane 1',
        va='center', ha='right', fontsize=9, color='#444444')
ax.text(ROAD_LEFT - 0.05, LANE0_Y, 'Lane 0',
        va='center', ha='right', fontsize=9, color='#444444')

# ── ramp ─────────────────────────────────────────────────────────────────────
# Ramp rises from lane-1 level on the right and angles up-right
RAMP_BASE_X  = 11.0   # where ramp diverges from road top
RAMP_TIP_X   = 21.0
RAMP_TIP_Y   = 6.5
RAMP_W       = 1.6    # ramp width (perpendicular)

# direction vector of ramp centreline
dx = RAMP_TIP_X - RAMP_BASE_X
dy = RAMP_TIP_Y - ROAD_TOP
length = np.hypot(dx, dy)
nx, ny = -dy / length, dx / length  # normal (left side of ramp)

half = RAMP_W / 2
pts_outer = [
    [RAMP_BASE_X + nx * half, ROAD_TOP + ny * half],
    [RAMP_TIP_X  + nx * half, RAMP_TIP_Y + ny * half],
    [RAMP_TIP_X  - nx * half, RAMP_TIP_Y - ny * half],
    [RAMP_BASE_X - nx * half, ROAD_TOP  - ny * half],
]
ramp_poly = plt.Polygon(pts_outer, closed=True, color='#b0b0b0', zorder=0)
ax.add_patch(ramp_poly)
# ramp edges
for sign in (+1, -1):
    xs = [RAMP_BASE_X + nx * half * sign, RAMP_TIP_X + nx * half * sign]
    ys = [ROAD_TOP    + ny * half * sign, RAMP_TIP_Y + ny * half * sign]
    ax.plot(xs, ys, color='#333333', lw=2.2, zorder=1)

# "Ramp" label
ax.text(RAMP_TIP_X - 0.5, RAMP_TIP_Y + 0.35, 'Ramp',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#222222')

# ── helper: rounded vehicle box ───────────────────────────────────────────────
def vehicle_box(cx, cy, label, sublabel, color, textcolor='white',
                w=2.1, h=1.15, zorder=3):
    box = FBP((cx - w/2, cy - h/2), w, h,
              boxstyle='round,pad=0.05', linewidth=1.4,
              edgecolor='white', facecolor=color, zorder=zorder)
    ax.add_patch(box)
    ax.text(cx, cy + 0.16, label, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=textcolor, zorder=zorder+1)
    ax.text(cx, cy - 0.26, sublabel, ha='center', va='center',
            fontsize=7.5, color=textcolor, zorder=zorder+1,
            style='italic' if sublabel.startswith('(') else 'normal')

# ── vehicles ─────────────────────────────────────────────────────────────────
# EGO
EGO_X = 3.8
vehicle_box(EGO_X, LANE1_Y, 'EGO', '(AV ★)', '#2a9d2a')

# NPC Aggressive (Lane 1, ahead of EGO)
AGG_X = 7.5
vehicle_box(AGG_X, LANE1_Y, 'NPC', 'Aggressive', '#cc3333')

# NPC Normal (Lane 1, far ahead)
NOR_X = 17.5
vehicle_box(NOR_X, LANE1_Y, 'NPC', 'Normal', '#555566')

# NPC Cautious (Lane 0)
CAU_X = 5.8
vehicle_box(CAU_X, LANE0_Y, 'NPC', 'Cautious', '#2266bb')

# NPC Merging (on ramp, above merge point)
MRG_CX = RAMP_BASE_X + (dx / length) * 2.8
MRG_CY = ROAD_TOP    + (dy / length) * 2.8
vehicle_box(MRG_CX, MRG_CY, 'NPC', 'Merging', '#e07800', zorder=5)

# ── gap arrow ─────────────────────────────────────────────────────────────────
GAP_Y   = LANE1_Y + 0.65
GAP_X0  = EGO_X + 1.05     # right edge of EGO box
GAP_X1  = AGG_X - 1.05     # left edge of AGG box

ax.annotate('', xy=(GAP_X0, GAP_Y), xytext=(GAP_X1, GAP_Y),
            arrowprops=dict(arrowstyle='<->', color='#22aa22', lw=2.0), zorder=6)
ax.text((GAP_X0 + GAP_X1) / 2, GAP_Y + 0.28, 'gap',
        ha='center', va='bottom', fontsize=9, color='#22aa22', fontweight='bold')

# ── merge arrow (NPC → gap) ───────────────────────────────────────────────────
arrow_start_x = MRG_CX - (dx / length) * 1.1
arrow_start_y = MRG_CY - (dy / length) * 1.1
arrow_end_x   = (GAP_X0 + GAP_X1) / 2 + 0.2
arrow_end_y   = LANE1_Y + 0.0

ax.annotate('', xy=(arrow_end_x, arrow_end_y),
            xytext=(arrow_start_x, arrow_start_y),
            arrowprops=dict(arrowstyle='->', color='#e07800',
                            lw=2.2, connectionstyle='arc3,rad=0.15'), zorder=6)
ax.text(arrow_start_x + 0.5, arrow_start_y - 0.55, 'merges\ninto gap',
        ha='left', va='top', fontsize=8, color='#e07800', style='italic')

# ── highway label ─────────────────────────────────────────────────────────────
ax.text(0.02, -1.05, '← Highway →',
        ha='left', va='bottom', fontsize=9, color='#333333',
        transform=ax.transData)

# ── caption ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         r'$\bf{Highway\!-\!Env}$ $\bf{merge\!-\!v0}$ — ego manages gap while NPC merges from ramp' + '\n'
         r'Driver types randomly assigned each episode from {cautious, normal, aggressive}',
         ha='center', va='bottom', fontsize=8.5, color='#333333',
         style='italic')

plt.tight_layout(rect=[0, 0.07, 1, 1])

fig.savefig('figures/merge_diagram.pdf', bbox_inches='tight', dpi=150)
fig.savefig('figures/merge_diagram.png', bbox_inches='tight', dpi=180)
print("Saved figures/merge_diagram.pdf and figures/merge_diagram.png")
plt.close()
