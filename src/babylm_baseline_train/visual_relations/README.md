# Example commands

```
for sd in 1 2 ; do for task in "vgr" "vga" ; do python vgr_vga.py --setting "BabyLM/exp_txt_vis.py:base_git_1vd25_s${sd}" --all_ckpts --high_level_task ${task} ; done ; done
```
