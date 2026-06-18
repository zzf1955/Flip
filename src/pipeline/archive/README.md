# Pipeline Archive

This directory keeps pipeline modules that are no longer part of the maintained
FLIP runtime surface. They are preserved for reference, reproduction, or manual
one-off reruns, but `scripts/flip_run.sh` does not dispatch to them.

Current maintained entry points stay directly under `src/pipeline/`.

| Subdirectory | Contents |
| --- | --- |
| `inpaint_retarget/` | Old FK/SAM2 inpaint, per-frame inpaint, segment pipeline, human overlay, and retarget video flows. |
| `patch/` | Old robot full-body patch and hand patch data utilities. |
| `comfyui_wan/` | Old local ComfyUI Wan/Cosmos regeneration experiments. |
| `idm/` | Old Wan/Humanoid Everyday/AdaWorld inverse-dynamics and action-mask experiments. |
| `training_legacy/` | Old mixed h2r training entry and runtime split builder. |
| `baseline/` | Old Masquerade-style direct-render baseline. |

