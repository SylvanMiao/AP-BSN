# AP-BSN: Self-Supervised Denoising for Confocal Microscopy

基于 [AP-BSN (CVPR 2022)](https://arxiv.org/abs/2203.11799) 的显微图像自监督去噪。原始方法针对 sRGB 自然图像设计，本项目将其适配到**Confocal图像**（单通道、混合 8/16-bit、灰度图）。

---

## 环境

- Python ≥ 3.9
- PyTorch ≥ 1.9
- numpy, opencv-python, scikit-image, scipy, pyyaml, tensorboard

---

## Hands-on

### 1. 准备数据

将共聚焦图像（PNG/TIF，单通道灰度）放入目录，例如：

```
dataset/confocal/
├─ img_001.png      # 16-bit
├─ img_002.png      # 8-bit
├─ img_003.tif      # 16-bit
└─ ...
```

> 支持同一数据集中混合 8-bit / 16-bit，代码会自动逐张检测位深并归一化。

修改 `conf/APBSN_CONFOCAL.yaml` 中数据集的路径：

```yaml
# src/datahandler/CONFOCAL.py 中修改 _scan() 里的 dataset_path
```

### 2. 训练

```bash
# 直接训练
python train.py -c APBSN_CONFOCAL -g 0 --thread 8

# 或使用脚本
bash train_confocal.sh
```

检查点保存在 `output/APBSN_CONFOCAL/checkpoint/`，每个 epoch 保存一次。

### 3. 推理

```bash
# 单张图
python test.py -c APBSN_CONFOCAL -g 0 --pretrained APBSN_CONFOCAL.pth --test_img ./test.png

# 整个文件夹
python test.py -c APBSN_CONFOCAL -g 0 --pretrained APBSN_CONFOCAL.pth --test_dir ./test_origin/

# 用训练中的检查点，不用预训练权重
python test.py -c APBSN_CONFOCAL -g 0 -e 50 --test_dir ./test_origin/
```

| 参数 | 说明 |
|------|------|
| `--test_img PATH` | 单张图像，输出 `<原名>_DN.png` |
| `--test_dir PATH` | 目录，输出到 `<dir>/results/` |
| `--self_en` | 几何自集成（8次旋转/翻转平均，更高质量） |

---

## 配置说明

编辑 `conf/APBSN_CONFOCAL.yaml`：

```yaml
model:
  kwargs:
    pd_a: 5           # 训练PD因子（须能整除 crop_size）
    pd_b: 2           # 推理PD因子
    bsn_base_ch: 128  # 主干网络通道数
    bsn_num_module: 9 # 主干网络深度

training:
  dataset_args:
    crop_size: [255, 255]  # 必须被 pd_a 整除
    n_repeat: 8
  batch_size: 8
  max_epoch: 100
  init_lr: 1e-4
  scheduler:
    type: step
    step:
      step_size: 8    # 每 N 个epoch衰减LR
      gamma: 0.1      # 衰减倍率
  loss: 1*self_L1     

validation:
  val: True
  save_image: True
  start_epoch: 1
  interval_epoch: 1   # 每N个epoch验证一次

test:
  save_image: True
```

### 超参数调整建议

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `init_lr` | 初始学习率 | `5e-5` ~ `1e-4` |
| `step_size` | LR 衰减频率 | `8` |
| `gamma` | LR 衰减倍率 | `0.1` |
| `pd_a` | 盲点约束强度 | `5` |
| `bsn_base_ch` | 模型宽度 | `256` |
| `bsn_num_module` | 模型深度 | `12` |
| `crop_size` | 训练patch大小 | `[128, 128]` 或 `[255, 255]` |

---

## 原始数据集

原始论文还支持 SIDD、DND、NIND 等 sRGB 数据集，配置文件在 `conf/APBSN_SIDD.yaml` 等，预训练权重见原仓库。本项目主要关注共聚焦显微图像。

---

## 改进记录

基于原版 AP-BSN 做的改动：

### 混合位深支持

原版假定所有图像同一位深。改动后 `_load_data()` 逐张检测 `norm_factor`（8-bit=255, 16-bit=65535），归一化和反归一化均按单张图像独立处理。8-bit 和 16-bit 图像可以混在同一个数据集里。

> `src/datahandler/CONFOCAL.py`, `src/trainer/base.py`

### 修复验证/推理时图像全黑

原版 `save_img_tensor_denorm()` 靠像素最大值 `≤ 255` 来判断存 uint8 还是 uint16。训练后期模型输出略微超出 `[0, 1]` 时，8-bit 图经过反归一化 + add_con 后 max 可能略超 255，被误判为 uint16，导致在 0-65535 范围下几乎全黑。

改为传入真实的 `norm_factor` 来确定位深。

> `src/util/file_manager.py`, `src/trainer/base.py`

### 修复可选参数 KeyError

`--test_img`、`--test_dir`、`--pretrained`、`-g` 未传时不在 config 字典中，原版 `self.cfg['test_img']` 直接抛 KeyError。给 `ConfigParser` 增加 `get()` 方法兜底。

> `src/util/config_parse.py`, `src/trainer/trainer.py`, `src/trainer/base.py`, `test.py`, `train.py`

### 单图/目录推理不依赖数据集路径

`_before_test()` 增加 `dataset_load` 参数，单图和目录推理时跳过数据集初始化，不需要存在数据集目录。

> `src/trainer/trainer.py`, `src/trainer/base.py`

---

## citation

```
@inproceedings{lee2022apbsn,
  title={AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network},
  author={Lee, Wooseok and Son, Sanghyun and Lee, Kyoung Mu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
