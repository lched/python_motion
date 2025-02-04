# Python-motion

This repository contains code (that I mostly didn't write myself) to load and process BioVision Hierarchy (.bvh) files in an easy, and convert the animation data to various representations, including several angular representations (Euler, quaternions, 6d representation of [Zhou et al.](https://arxiv.org/abs/1812.07035))

motion_tutorials contains example snippets of code to do various things.

:warning: To work with this code, BVH files are expected to have Z, X, Y rotation order and Y-up convention. I'm not completely sure other conventions work.

:warning: **Must be used with matplotlib 3.3**, newer versions of matplotlib might break the plot_script functions
