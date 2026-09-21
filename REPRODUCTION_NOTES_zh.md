# PertaBind 复现说明

## 复现结论

本代码包是依据手稿方法部分和图 1B 独立重建的端到端实现，不是作者源码的镜像。
手稿列出的 GitHub 地址在整理本代码时无法公开访问，因此无法验证作者的具体类名、张量维度、
数据清洗代码或随机种子。

## 手稿明确给出的设置

- 输入：蛋白序列、ESMFold 预测的 apo-like 结构、holo 蛋白-配体复合物、SMILES、
  RDKit 生成并能量最小化的游离配体三维构象、holo 中的结合构象。
- 口袋：任一残基原子到任一配体重原子的最小距离不超过 8 Å。
- 三层：直接接触层 0–4 Å、近接触层 4–6 Å、远端结构层 6–8 Å。
- 教师：holo 全局蛋白、结合态配体、异构全原子口袋、原子到残基汇聚、多层扰动表示。
- 学生：蛋白序列 + apo-like 几何图、SMILES 二维图 + 游离态三维构象。
- 蒸馏：预测值 L1 距离和隐表示 L2 距离，教师侧 stop-gradient。
- 总损失：`L_aff + λ1 L_rank + λ2 L_align + λ3 L_dist`。
- 优化器：AdamW，`β1=0.9`，`β2=0.999`。
- 阶段 I：学习率 `1e-4`，batch size `8`。
- 阶段 II：学习率 `5e-4`，batch size `32`。
- 五折交叉验证，以验证集最低 RMSE 选择 checkpoint。
- MdrDB：直接预测实验 `ΔΔG`；抗性阈值为 `ΔΔG > 1.36 kcal/mol`。
- 回归指标：RMSE、MAE、PCC；分类指标：AUPRC、MCC、F1。

## 手稿未公开、代码中采用可配置默认值的设置

- 隐藏维度、网络层数、注意力头数、RBF 数量和 dropout。
- `λ1/λ2/λ3`、教师监督项权重、weight decay、梯度裁剪。
- 两阶段 epoch 数、early-stopping patience、学习率调度器。
- 原子/残基特征的精确编码方式，以及 DSSP/SASA/depth 的具体软件与参数。
- SE(3) 网络的具体实现、邻居数与截断半径。
- PDBbind CleanSplit 的确切版本和 MdrDB 的逐条清洗规则。
- ESMFold checkpoint revision、五折原始划分和全部随机种子。

这些值集中在 `configs/default.yaml`，并已逐项标注 `[paper]` 或
`reconstruction default`。一旦获得作者参数，可以只修改配置或对应模块，无需重写训练流程。

## 推荐的严格复现顺序

1. 固化原始数据版本、许可文件、文件哈希和 CleanSplit/CASF membership。
2. 固化 ESMFold checkpoint revision，批量生成 apo-like 结构。
3. 用 `prepare_pdbbind.py` 生成原始 manifest，并保留转换前后的标签。
4. 用 `processdata.py` 生成图缓存，检查 `failures.csv`，不要静默丢弃失败样本。
5. 固化 `folds.csv`，确认 CASF-2016 和独立子集完全不参与拟合、调参和选模。
6. 分别训练五个 fold；每个 fold 先阶段 I，再阶段 II。
7. 对固定外部测试集分别计算五个模型的指标，报告 mean ± SD。
8. 对 MdrDB 单独训练模型并同时报告回归与抗性分类指标。
9. 保存有效 YAML、split membership、依赖版本、checkpoint 和逐样本预测。

## 重要解释边界

前瞻筛选只能调用学生分支。教师侧 residue/atom attribution 依赖 holo 复合物，适合训练后机制解释，
不能把该信息作为未见配体筛选时的输入，否则会形成信息泄漏。`screen.py` 和 `predict.py` 已强制使用
`student_only=True`；`interpret.py` 才会运行教师分支。

