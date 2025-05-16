# NuScenes dataset prepare

Following [BEVFormer repo](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md) to prepare nuScenes dataset.


Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

**Folder structure**
```
BEVDiffuser
├── BEVFormer
│   ├── projects/
│   ├── tools/
│   ├── ckpts/
│   │   ├── bevformer_tiny_epoch_24.pth
│   ├── data/
│   │   ├── can_bus/
│   │   ├── nuscenes/
│   │   │   ├── maps/
│   │   │   ├── samples/
│   │   │   ├── sweeps/
│   │   │   ├── v1.0-test/
│   |   |   ├── v1.0-trainval/
│   |   |   ├── nuscenes_infos_temporal_train.pkl
│   |   |   ├── nuscenes_infos_temporal_val.pkl
```
