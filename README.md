# hnode
Code, examples, and pre-trained checkpoints for "A hybridizable neural time integrator for stable autoregressive forecasting"

All notebooks are set up currently to load from the pre-trained checkpoints and run in inference. To train, simply set NUM_EPOCHS to a nonzero quantity.
## Large checkpoint reconstruction

GitHub LFS rejects individual objects larger than 2 GiB, so the two largest checkpoints are stored as split LFS parts. After cloning and running `git lfs pull`, reconstruct them with:

```bash
cat shear_flow_redo_part_2.pt.part-* > shear_flow_redo_part_2.pt
cat swin_plasma_e2e_redo_part_2.pth.part-* > swin_plasma_e2e_redo_part_2.pth
```

The reconstructed files are git-ignored locally because they are generated from the tracked parts.
