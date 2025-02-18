# Python-motion

This repository contains code (that I mostly didn't write myself) to load and process BioVision Hierarchy (.bvh) files in an easy, and convert the animation data to various representations, including several angular representations (Euler, quaternions, 6d representation of [Zhou et al.](https://arxiv.org/abs/1812.07035))

motion_tutorials contains example snippets of code to do various things.

:warning: To work with this code, BVH files are expected to have Z, X, Y rotation order and Y-up convention. I'm not completely sure other conventions work.

:warning: **Must be used with matplotlib 3.3**, newer versions of matplotlib might break the plot_script functions


smpl2bvh repository here: https://github.com/KosukeFukazawa/smpl2bvh
SMPL models can't be distributed with this software because of SMPL license.

:warning: This repository is still very much a work in progress (for example there are two bvh files/processing, how confusing right?). I'm just making it public if it can help someone.
