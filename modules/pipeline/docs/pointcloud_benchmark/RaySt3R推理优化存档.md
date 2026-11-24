# RaySt3R 推理优化存档

记录日期：2026年11月22日。状态：方案已存档，暂停速度优化，先评估 RaySt3R 点云接 GraspGenX 后的实际抓取效果。

## 当前基线

已完成的五方法、50场景重建评测中，RaySt3R 常驻生成耗时中位数为1.261秒，P95为1.390秒，F-score@5mm均值为91.41%，重建误差中位数为1.81毫米。耗时包含适配器预处理、文件接口、模型计算和必要的CPU传输；尚无逐阶段耗时测量。

当前模型和DINO均已常驻，已使用inference_mode，未开启可视化或TSDF。RaySt3R主网络已使用BF16自动混合精度，DINO特征提取仍为FP32。配置中的`n_pred_views=5`表示采样网格参数，实际生成21个新视角，加输入视角共22个；已用本地采样函数核对。

## 待开展的优化

| 顺序 | 方法 | 已核查的依据 | 核验方式 |
| --- | --- | --- | --- |
| 1 | 两阶段复用同一张图的DINO特征 | 输入视角预测和新视角预测分别调用特征提取 | 保留两阶段、相同掩码及视角，比较输出和耗时 |
| 2 | 单独提供只返回预测的推理入口 | 当前推理仍计算训练损失及评估指标，指标含多次`.item()`同步 | 保留深度、掩码、置信度，删除不影响预测的计算后核对输出 |
| 3 | RaySt3R注意力使用SDPA融合实现 | 主网络注意力仍手写矩阵乘法及Softmax；本地DINO已经使用SDPA | 核查实际CUDA后端，并比较重建精度、耗时和显存 |
| 4 | DINO使用BF16 | DINO调用位于主网络BF16上下文外 | 几何计算保留FP32，在同50场景比较精度和延迟 |
| 5 | 内存传参、缓存像素网格、减少复制 | 每次保存五个输入文件，并构建两次读取流程 | 先保持原深度量化与预处理等价，再比较输出和时间 |
| 6 | 编译固定形状的网络部分 | 当前未使用torch.compile | 对DINO与主网络分开编译，预热后计时，记录编译成本和显存 |
| 7 | 减少合成视角 | 参数5/4/3分别对应21/13/7个新视角 | 在同50场景比较耗时、整体精度及细长物体和遮挡面的变化 |

先逐段计时，再评估保持现有视角的工程优化，最后评估减少视角。当前没有测得任何优化后的加速倍数。

## 恢复入口

- 冻结基线：`experiments/pointcloud_benchmark/runs/five_methods_50_20261122_01/`。
- 适配器：`experiments/pointcloud_benchmark/model_adapters.py`。
- 两阶段入口：`third_party/rayst3r/eval_wrapper/eval.py`。
- 预处理与DINO：`third_party/rayst3r/utils/batch_prep.py`。
- 推理与评估：`third_party/rayst3r/engine.py`。
- 注意力：`third_party/rayst3r/models/blocks.py`。
- 视角采样：`third_party/rayst3r/eval_wrapper/sample_poses.py`。

恢复优化时另建实验目录与登记，保留本次冻结的五方法结果和源码。

## 参考

- [RaySt3R官方项目](https://rayst3r.github.io/)
- [官方代码及参数说明](https://github.com/Duisterhof/rayst3r)
- [PyTorch 2.7 SDPA](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch 2.7编译接口](https://docs.pytorch.org/docs/2.7/generated/torch.compile.html)
