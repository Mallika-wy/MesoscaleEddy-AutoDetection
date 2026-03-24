# 中尺度涡识别
## 任务说明
实现基于人工智能的中尺度涡旋的自动化识别模块，完成边界定位、形态分割
三分类任务，背景/气旋/反气旋


## 数据说明
1. 海区：黑潮延伸体（25–45°N，140–180°E）
2. 数据集：基于META4.0 DT的方法标注涡旋轮廓
    - 空间分辨率：0.125°
    - 时间分辨率：daily

训练集	1993-01-01到2022-12-31	
- 19930101_20021231.nc
- 20030101_20121231.nc
- 20130101_20221231.nc
测试集	2023-01-01到2023-12-31	
- 20230101_20231231.nc
验证集	2024-01-01到2024-12-31	
- 20240101_20241231.nc

3. 变量：
- ADT：sea surface height above geoid
- UGOS：surface geostrophic eastward sea water velocity
- VGOS：surface geostrophic eastward sea water velocity

## py-eddy-tracker 标注

仓库中新增了批量标注脚本 [scripts/label_eddy_contours.py](scripts/label_eddy_contours.py)，默认按 META4.0 DT 常用参数运行：
- `cut_wavelength=800 km`
- `filter_order=1`
- `isoline_step=0.002 m`
- `shape_error=55`
- `pixel_limit=(5, 2000)`

建议在 `ocean-swinlstm` 环境里运行：

```bash
conda run -n ocean-swinlstm python scripts/label_eddy_contours.py ^
  --start-date 2023-01-01 ^
  --end-date 2023-01-07
```

默认输出到 `outputs/meta40_dt_contours/`：
- 每天两份轮廓文件：`Anticyclonic_YYYYMMDDT000000.nc` 和 `Cyclonic_YYYYMMDDT000000.nc`
- 一份汇总表：`summary.csv`

## JTECH 2022 复现适配主线

新增了一套独立于 `whirlpool/` 的全场时空分割实现，核心方法对应论文
`A Dual-Attention Mechanism Deep Learning Network for Mesoscale Eddy Detection by Mining Spatiotemporal Characteristics`，
并按当前任务改为黑潮延伸体的三分类分割。

默认配置文件：

```bash
configs/jtech_adapt.yaml
```

推荐流程：

1. 先生成 py-eddy-tracker 轮廓文件
2. 再把轮廓栅格化成每日标签掩膜
3. 训练模型
4. 运行预测与评估

示例命令：

```bash
conda run -n ocean-swinlstm python scripts/build_label_store.py
conda run -n ocean-swinlstm python scripts/train_spatiotemporal_segmentation.py
conda run -n ocean-swinlstm python scripts/predict_spatiotemporal_segmentation.py --split test
conda run -n ocean-swinlstm python scripts/evaluate_spatiotemporal_segmentation.py --split test
```

统一 `sh` 启动脚本：

```bash
sh scripts/run_jtech_pipeline.sh train
sh scripts/run_jtech_pipeline.sh predict test
sh scripts/run_jtech_pipeline.sh evaluate test
sh scripts/run_jtech_pipeline.sh train_eval test
sh scripts/run_jtech_pipeline.sh full 1993-01-01 2024-12-31 test
```

实现约定：

- 输入变量：`adt`、`ugos`、`vgos`
- 输入张量：`[T=3, C=3, H=160, W=320]`
- 标签定义：`0=background`、`1=cyclonic`、`2=anticyclonic`、`255=ignore(land)`
- 训练/验证/测试划分与本文档开头保持一致
