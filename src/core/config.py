"""Centralized project configuration.

Edit ACTIVE_TASK and ACTIVE_EPISODES to control which data subset
scripts process during small-scale testing.
"""

import os

# Project root (two levels up from src/core/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Mesh / URDF paths ──
_MESH_ROOT = os.path.join(BASE_DIR, "data", "unitree_G1_WBT", "mesh")
MESH_DIR = os.path.join(_MESH_ROOT, "meshes")
G1_URDF = os.path.join(_MESH_ROOT,
                       "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")
HUMAN_URDF = os.path.join(_MESH_ROOT, "human_arm_overlay.urdf")

# ── SMPLH model path ──
SMPLH_PATH = (
    "/disk_n/zzf/video-gen/MIMO/video_decomp/"
    "models--menyifang--MIMO_VidDecomp/snapshots/"
    "41a6023cc405f73d888cabe5cb7506da99bbbec6/assets/smplh/SMPLH_NEUTRAL.npz"
)

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

# ── Camera model selection (change this line to switch) ──
CAMERA_MODEL = "pinhole_fixed"   # "pinhole_fixed" | "pinhole_f" | "pinhole" | "fisheye"

# ── Per-model best parameters ──
_EXTRINSICS = {"dx": 0.0758, "dy": 0.0226, "dz": 0.4484,
               "pitch": -61.5855, "yaw": 2.1690, "roll": 0.2331}

BEST_PARAMS_BY_MODEL = {
    "pinhole_fixed": {**_EXTRINSICS, "cy": 313.6840},
    "pinhole_f": {**_EXTRINSICS, "f": 290.78, "cx": 329.37, "cy": 313.68},
    "pinhole":   {**_EXTRINSICS, "fx": 290.7808, "fy": 287.3524, "cx": 329.3684, "cy": 313.6840},
    "fisheye":   {**_EXTRINSICS, "fx": 290.7808, "fy": 287.3524, "cx": 329.3684, "cy": 313.6840,
                  "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
}

BEST_PARAMS = BEST_PARAMS_BY_MODEL[CAMERA_MODEL]

# ── Mesh skip list (links not to render) ──
SKIP_MESHES = {"head_link", "logo_link", "d435_link"}

# Inspire hand link names — used to skip hand mesh rendering for non-Inspire tasks
_INSPIRE_HAND_LINKS = {
    f"{side}_{finger}_{seg}"
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "little")
    for seg in ("1", "2", "3", "4",
                "force_sensor_1", "force_sensor_2",
                "force_sensor_3", "force_sensor_4")
} | {
    "left_base_link", "right_base_link",
    "left_palm_force_sensor", "right_palm_force_sensor",
}

# ── Hand type detection ──
def get_hand_type(task_name=None):
    """Detect hand type from task name. Returns 'brainco' or 'inspire'."""
    if task_name is None:
        task_name = ACTIVE_TASK
    return "brainco" if "Brainco" in task_name else "inspire"


def get_skip_meshes(hand_type=None):
    """Return skip set for mesh rendering. Skips Inspire hand links for BrainCo tasks."""
    if hand_type is None:
        hand_type = get_hand_type()
    if hand_type == "brainco":
        return SKIP_MESHES | _INSPIRE_HAND_LINKS
    return SKIP_MESHES


# ── Output ──
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
