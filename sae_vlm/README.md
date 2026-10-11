# `sae_vlm/`

SAE training and evaluation code for this submission. See the top-level
`README.md` for which script reproduces which experiment.

This directory builds on the public PatchSAE code base
(Lim et al., "Sparse autoencoders reveal selective remapping of visual concepts
during adaptation", ICLR 2025; https://github.com/hyesulim/patchsae), released
under the MIT license. The original license and third-party notices are kept in
`LICENSE` and `NOTICE` and apply to the code derived from it.

Run scripts from this directory with `PYTHONPATH=./`, for example

```bash
PYTHONPATH=./ python train_sae.py --help
```
