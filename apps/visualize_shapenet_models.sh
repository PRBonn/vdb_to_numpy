#!/bin/bash
# ./apps/visualize_shapenet_models.sh $DATASETS/shapenet/
set -e
MODELS=$1
for model in $MODELS/**/*.obj; do
    echo "Conerting $model to vdb WITHOUT filling holes..."
    ./apps/visualize_vdb.py --voxel_size 0.01 --scale $model
    echo "Conerting $model to vdb WITH filling holes..."
    ./apps/visualize_vdb.py --voxel_size 0.01 --scale --fill_holes $model || true
done
