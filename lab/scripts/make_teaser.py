"""Compose the CoRL workshop teaser from rendered simulation frames."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

ROOT = Path("/home/yizhou/Projects/RoboChess-Challenge")
OUT = ROOT / "lab/docs/images/robochess_teaser.png"
NAVY, TEAL, INK, MUTE = "#12233f", "#2a9d8f", "#16181d", "#5a6472"

def load(rel, box=None):
    im = Image.open(ROOT / rel).convert("RGB")
    return np.asarray(im.crop(box) if box else im)

# The "showcase" camera frames the whole arm; the default replay view is aimed at the
# board and cuts every robot off above the elbow. Frame 03 is the carry moment, and the
# top 44 px are trimmed to drop the burnt-in caption.
CARRY_FRAME = (0, 44, 1280, 720)
arms = [
    ("Franka Panda", load("lab/docs/images/showcase_franka/demo_0_frames/03.png", CARRY_FRAME)),
    ("reBot",        load("lab/docs/images/showcase_rebot/demo_0_frames/03.png", CARRY_FRAME)),
    ("Piper",        load("lab/docs/images/showcase_piper/demo_0_frames/03.png", CARRY_FRAME)),
    ("YAM",          load("lab/docs/images/showcase_yam/demo_0_frames/03.png", CARRY_FRAME)),
]
variants = [
    # The 1D board runs away from the head-on camera, so it reads as a stub there;
              # the angled render shows all six pieces and the gap between the sides.
              ("1D chess  ·  K N R", load("lab/docs/images/hero_1d_angle.png", (480, 480, 1380, 840))),
    ("3x3 Hexapawn", load("lab/docs/images/hero_3x3_hero.png", (450, 640, 1350, 1000))),
    ("4x4 minichess", load("lab/docs/images/hero_minichess_hero.png", (450, 620, 1350, 980))),
]
hero = load("lab/docs/images/hero_minichess_hero.png", (60, 0, 1740, 1000))

fig = plt.figure(figsize=(16, 9.6), dpi=150, facecolor="white")

# ---- title -------------------------------------------------------------------
fig.text(0.032, 0.982, "RoboChess", fontsize=38, fontweight="bold", color=NAVY, va="top")
fig.text(0.032, 0.933, "Two robot arms playing chess in Isaac Lab", fontsize=18, color=INK, va="top")
fig.text(0.032, 0.902, "A GPU-accelerated manipulation benchmark for generalizable, sim-to-real robot policies"
                       "   ·   CoRL 2026 Workshop", fontsize=12, color=MUTE, va="top")
fig.add_artist(plt.Line2D([0.032, 0.968], [0.884, 0.884], color=NAVY, lw=2.2))

# ---- hero --------------------------------------------------------------------
ax = fig.add_axes([0.032, 0.325, 0.545, 0.545]); ax.imshow(hero); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.012, 0.885), 0.40, 0.093, transform=ax.transAxes,
                            boxstyle="round,pad=0.008", facecolor=NAVY, alpha=0.92, edgecolor="none"))
ax.text(0.030, 0.932, "White and Black, one arm each", transform=ax.transAxes,
        color="white", fontsize=13, fontweight="bold", va="center")

# ---- embodiments -------------------------------------------------------------
fig.text(0.607, 0.862, "FOUR EMBODIMENTS, ONE TASK DEFINITION", fontsize=11.5,
         fontweight="bold", color=NAVY, va="top")
for i, (name, img) in enumerate(arms):
    x = 0.607 + (i % 2) * 0.187
    y = 0.640 - (i // 2) * 0.238
    a = fig.add_axes([x, y, 0.175, 0.198]); a.imshow(img); a.axis("off")
    a.add_patch(FancyBboxPatch((0.02, 0.02), 0.62, 0.145, transform=a.transAxes,
                               boxstyle="round,pad=0.01", facecolor="white", alpha=0.88, edgecolor="none"))
    a.text(0.055, 0.093, name, transform=a.transAxes, fontsize=11, fontweight="bold", color=NAVY, va="center")

# ---- variants ----------------------------------------------------------------
fig.text(0.032, 0.283, "THREE VARIANTS, PLAYED BY THE RULES", fontsize=11.5,
         fontweight="bold", color=NAVY, va="top")
for i, (name, img) in enumerate(variants):
    a = fig.add_axes([0.032 + i * 0.196, 0.055, 0.185, 0.200]); a.imshow(img); a.axis("off")
    a.text(0.5, -0.10, name, transform=a.transAxes, fontsize=11, color=INK, ha="center", va="top")

# ---- numbers -----------------------------------------------------------------
box = fig.add_axes([0.632, 0.045, 0.336, 0.238]); box.axis("off")
box.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=box.transAxes,
                             boxstyle="round,pad=0.012", facecolor="#eef2f8", edgecolor=NAVY, lw=1.4))
rows = [("226", "recorded pick-and-place demonstrations"),
        ("66.8k", "transitions, all success-verified"),
        ("6 / 6", "piece types covered on the full board"),
        ("2.6 mm", "median placement error"),
        ("900", "random games verifying the rule engine")]
for i, (big, small) in enumerate(rows):
    y = 0.855 - i * 0.192
    box.text(0.055, y, big, fontsize=17, fontweight="bold", color=TEAL, va="center", ha="left")
    box.text(0.30, y, small, fontsize=10.5, color=INK, va="center", ha="left")

fig.text(0.032, 0.016, "Isaac Lab 3.0  ·  grasps from GraspGen  ·  pieces, board and robot models "
                       "released as .usd / .stl", fontsize=10, color=MUTE)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.18)
print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")
