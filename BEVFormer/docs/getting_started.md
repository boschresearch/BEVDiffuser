# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train BEVDiffuser with 4 GPUs 
```
cd BEVFormer/projects/bevdiffuser
# 
./train.sh 4
```

Eval BEVDiffuser
```
cd BEVFormer/projects/bevdiffuser
./test.sh
```