"""Centralized project configuration.

Edit ACTIVE_TASK and ACTIVE_EPISODES to control which data subset
scripts process during small-scale testing.
"""

import os

# Project root (one level up from scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Mesh / URDF paths ──
_MESH_ROOT = os.path.join(BASE_DIR, "data", "unitree_G1_WBT", "mesh")
MESH_DIR = os.path.join(_MESH_ROOT, "meshes")
G1_URDF = os.path.join(_MESH_ROOT,
                       "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")
HUMAN_URDF = os.path.join(_MESH_ROOT, "human_arm_overlay.urdf")

# ── Dataset root (LeRobot format, downloaded via download_g1_wbt.sh) ──
DATASET_ROOT = os.path.join(BASE_DIR, "data", "unitree_G1_WBT")

# ── Active task + episodes for small-scale testing ──
# Change these when switching tasks. Scripts read ACTIVE_TASK / ACTIVE_EPISODES.
ACTIVE_TASK = "G1_WBT_Brainco_Make_The_Bed"
ACTIVE_EPISODES = [0, 4, 50]

# Derived path
ACTIVE_DATA_DIR = os.path.join(DATASET_ROOT, ACTIVE_TASK)

# ── All available tasks ──
ALL_TASKS = [
    "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
    "G1_WBT_Brainco_Make_The_Bed",
    "G1_WBT_Brainco_Pickup_Pillow",
    "G1_WBT_Inspire_Collect_Clothes_MainCamOnly",
    "G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
    "G1_WBT_Inspire_Put_Clothes_Into_Basket",
    "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine",
    "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
]

# ── Camera parameters (PSO-calibrated, IoU=0.8970) ──
BEST_PARAMS = {
    "dx": 0.039, "dy": 0.052, "dz": 0.536,
    "pitch": -53.6, "yaw": 4.7, "roll": 3.0,
    "fx": 315, "fy": 302, "cx": 334, "cy": 230,
    "k1": 0.63, "k2": 0.17, "k3": 1.19, "k4": 0.25,
}

# ── Mesh skip list (links not to render) ──
SKIP_MESHES = {"head_link", "logo_link", "d435_link"}

# ── Hand type detection ──
def get_hand_type(task_name=None):
    """Detect hand type from task name. Returns 'brainco' or 'inspire'."""
    if task_name is None:
        task_name = ACTIVE_TASK
    return "brainco" if "Brainco" in task_name else "inspire"


# ── Output ──
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results")

# NOTE: auto_calibrate.py / interactive_calibrate.py use an older URDF
# (g1_29dof_rev_1_0.urdf) and different mesh paths. Calibration is already
# completed (IoU=0.8970), so those scripts are not refactored here.
