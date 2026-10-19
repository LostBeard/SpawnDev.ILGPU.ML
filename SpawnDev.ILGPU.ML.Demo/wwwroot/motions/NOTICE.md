# Motion clips

The clips in this folder (`.bin` files that are `.npz` archives, renamed because static-file servers do not know .npz) are the preset moves of **image-to-3dlab** (https://github.com/Bingeljell/image-to-3dlab, Apache-2.0, `image_to_3dlab/motions/`),
first take of each move. They were made with **NVIDIA Kimodo** (Kimodo-SOMA-RP-v1.1; code Apache-2.0, model under the NVIDIA Open Model License) from the text prompts
in `moves.json`; NVIDIA claims no ownership of Kimodo's outputs. Each clip holds, for NVIDIA's 77-joint SOMA skeleton, the world rotation of every joint away from SOMA's T-pose per
frame, the frame-0 joint positions and the hips track (see `SpawnDev.ILGPU.ML.Animation.MotionClip`).

They are bundled for the demo's "Animate" button. Check the licences above before using them commercially.
