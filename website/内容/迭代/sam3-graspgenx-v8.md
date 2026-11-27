## 内存状态更新：完整抓取测试

### 本轮结果

148个登记槽全部处理，147个完整结束、1个程序中断、0个未开始。排除共同站位前提未满足和中断场景后，有68个完整配对场景，双方分别成功53次和53次（77.9%／77.9%）。各阶段规划服务耗时合计平均15.41→11.56秒，减少25.0%。

全登记表里原方法54次成功、内存更新53次，差的一次来自香蕉新方法的程序中断；该条保留且未重抓。

使用同一组74个配对场景，新旧方法每场景各一次，共148个登记槽。69个场景各组重新复位仿真并规划抓取；5个原站位前提未满足的场景单列未执行。

两组沿用同一份冻结头部RGB-D、SAM3分割和15毫米／15度NMS后的候选。本轮实际执行开爪、预抓取、按实测状态重规划接近、闭合、按实测状态重规划抬升与3秒保持。成功判据沿用目标最低保持抬升至少1厘米、双指有效接触、各段到位且无禁止接触。

| 方法 | 完整结束 | 程序中断 | 初始规划通过 | 实际执行 | 抓取成功 | 前提未满足未执行 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原方法 | 74/74 | 0 | 68/74（91.9%） | 68 | 54/74（73.0%） | 5 |
| 内存状态更新 | 73/74 | 1 | 68/74（91.9%） | 68 | 53/74（71.6%） | 5 |

### 各段规划计时

| 环节 | 原方法平均／中位／P90 | 内存更新平均／中位／P90 |
| --- | ---: | ---: |
| 初始规划完整请求（含交接与返回） | 12.052／8.177／21.894秒（68次） | 8.344／6.346／14.619秒（68次） |
| 初始规划服务内部 | 12.042／8.171／21.887秒（68次） | 8.317／6.310／14.364秒（68次） |
| 到预抓取后重规划接近 | 1.638／1.619／2.009秒（67次） | 1.552／1.562／1.860秒（67次） |
| 实测闭合后重规划抬升 | 2.021／2.002／2.741秒（59次） | 1.981／1.928／2.793秒（59次） |
| 每场景全部规划服务合计 | 15.410／11.507／25.996秒（68次） | 11.565／9.696／17.335秒（68次） |

主耗时表使用两组同一场景、同一实际发生阶段的配对；中断场景单列，不计入配对时延：M0004_S1_open_banana。

### 失败分布

| 原因 | 原方法 | 内存更新 |
| --- | ---: | ---: |
| 接近未到位 | 1 | 1 |
| 闭合未到位 | 1 | 1 |
| 物体抬升或保持不足 | 6 | 6 |
| 规划不出来 | 1 | 1 |
| 闭合后未形成双指夹持 | 6 | 6 |

### 核验与对照

| 项目 | 结果 |
| --- | --- |
| 任务结果 | 148个登记槽已处理；147完整、1中断、0未开始 |
| 记录完整性 | 272段录像完整解码；完整项命令与回执通过，中断项停止记录单列 |
| 事件适用性 | 发生过的接近、闭合、抬升、保持按实际记录核验；未执行项单列 |
| 关键事件覆盖 | 完整项核查137次初始规划、135次接近重规划、119次抬升重规划；独立验算405个末端点和524条命令路径，解码272段录像 |
| 初始抓法、手臂与关节相同 | 是 |
| 初始命令轨迹相同 | 是 |

完整配对范围内，原方法成功而内存更新失败的场景为0。

完整配对范围内，内存更新成功而原方法失败的场景为0。

### 逐场景结果

| 场景 | 原方法 | 内存更新 |
| --- | --- | --- |
| M0001_S1_open_apple | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0001-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0001-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0001-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0001-optimized-head.mp4) |
| M0002_S1_open_avocado | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0002-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0002-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0002-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0002-optimized-head.mp4) |
| M0003_S1_open_baguette | 前提未满足，未执行 | 前提未满足，未执行 |
| M0004_S1_open_banana | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0004-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0004-baseline-head.mp4) | 程序中断，未重复执行 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0004-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0004-optimized-head.mp4) |
| M0005_S1_open_bar_soap | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0005-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0005-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0005-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0005-optimized-head.mp4) |
| M0006_S1_open_bell_pepper | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0006-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0006-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0006-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0006-optimized-head.mp4) |
| M0007_S1_open_book | 前提未满足，未执行 | 前提未满足，未执行 |
| M0008_S1_open_bottled_water | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0008-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0008-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0008-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0008-optimized-head.mp4) |
| M0009_S1_open_bowl | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0009-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0009-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0009-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0009-optimized-head.mp4) |
| M0010_S1_open_boxed_food | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0010-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0010-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0010-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0010-optimized-head.mp4) |
| M0011_S1_open_bread | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0011-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0011-baseline-head.mp4) | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0011-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0011-optimized-head.mp4) |
| M0012_S1_open_broccoli | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0012-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0012-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0012-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0012-optimized-head.mp4) |
| M0015_S1_open_carrot | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0015-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0015-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0015-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0015-optimized-head.mp4) |
| M0016_S1_open_cell_phone | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0016-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0016-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0016-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0016-optimized-head.mp4) |
| M0018_S1_open_croissant | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0018-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0018-baseline-head.mp4) | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0018-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0018-optimized-head.mp4) |
| M0019_S1_open_cucumber | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0019-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0019-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0019-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0019-optimized-head.mp4) |
| M0020_S1_open_donut | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0020-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0020-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0020-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0020-optimized-head.mp4) |
| M0022_S2_edge_keychain | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0022-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0022-baseline-head.mp4) | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0022-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0022-optimized-head.mp4) |
| M0023_S2_edge_lemon | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0023-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0023-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0023-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0023-optimized-head.mp4) |
| M0024_S2_edge_mango | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0024-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0024-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0024-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0024-optimized-head.mp4) |
| M0026_S2_edge_mug | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0026-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0026-baseline-head.mp4) | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0026-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0026-optimized-head.mp4) |
| M0027_S2_edge_mushroom | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0027-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0027-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0027-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0027-optimized-head.mp4) |
| M0029_S2_edge_orange | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0029-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0029-baseline-head.mp4) | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0029-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0029-optimized-head.mp4) |
| M0030_S2_edge_peach | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0030-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0030-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0030-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0030-optimized-head.mp4) |
| M0031_S2_edge_pear | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0031-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0031-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0031-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0031-optimized-head.mp4) |
| M0032_S2_edge_potato | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0032-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0032-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0032-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0032-optimized-head.mp4) |
| M0033_S2_edge_remote_control | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0033-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0033-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0033-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0033-optimized-head.mp4) |
| M0034_S2_edge_scissors | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0034-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0034-baseline-head.mp4) | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0034-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0034-optimized-head.mp4) |
| M0036_S2_edge_spatula | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0036-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0036-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0036-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0036-optimized-head.mp4) |
| M0038_S2_edge_teapot | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0038-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0038-baseline-head.mp4) | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0038-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0038-optimized-head.mp4) |
| M0039_S2_edge_tomato | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0039-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0039-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0039-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0039-optimized-head.mp4) |
| M0041_S3_sparse_apple | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0041-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0041-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0041-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0041-optimized-head.mp4) |
| M0042_S3_sparse_avocado | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0042-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0042-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0042-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0042-optimized-head.mp4) |
| M0043_S3_sparse_baguette | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0043-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0043-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0043-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0043-optimized-head.mp4) |
| M0044_S3_sparse_banana | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0044-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0044-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0044-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0044-optimized-head.mp4) |
| M0045_S3_sparse_bar_soap | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0045-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0045-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0045-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0045-optimized-head.mp4) |
| M0046_S3_sparse_bell_pepper | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0046-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0046-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0046-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0046-optimized-head.mp4) |
| M0047_S3_sparse_book | 规划不出来 | 规划不出来 |
| M0048_S3_sparse_bottled_water | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0048-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0048-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0048-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0048-optimized-head.mp4) |
| M0050_S3_sparse_boxed_food | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0050-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0050-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0050-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0050-optimized-head.mp4) |
| M0052_S3_sparse_broccoli | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0052-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0052-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0052-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0052-optimized-head.mp4) |
| M0055_S3_sparse_carrot | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0055-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0055-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0055-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0055-optimized-head.mp4) |
| M0056_S3_sparse_cell_phone | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0056-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0056-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0056-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0056-optimized-head.mp4) |
| M0058_S3_sparse_croissant | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0058-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0058-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0058-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0058-optimized-head.mp4) |
| M0059_S3_sparse_cucumber | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0059-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0059-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0059-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0059-optimized-head.mp4) |
| M0060_S3_sparse_donut | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0060-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0060-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0060-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0060-optimized-head.mp4) |
| M0061_S4_clutter_eggplant | 接近未到位 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0061-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0061-baseline-head.mp4) | 接近未到位 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0061-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0061-optimized-head.mp4) |
| M0062_S4_clutter_keychain | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0062-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0062-baseline-head.mp4) | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0062-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0062-optimized-head.mp4) |
| M0063_S4_clutter_lemon | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0063-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0063-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0063-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0063-optimized-head.mp4) |
| M0064_S4_clutter_mango | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0064-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0064-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0064-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0064-optimized-head.mp4) |
| M0065_S4_clutter_milk | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0065-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0065-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0065-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0065-optimized-head.mp4) |
| M0066_S4_clutter_mug | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0066-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0066-baseline-head.mp4) | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0066-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0066-optimized-head.mp4) |
| M0067_S4_clutter_mushroom | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0067-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0067-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0067-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0067-optimized-head.mp4) |
| M0069_S4_clutter_orange | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0069-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0069-baseline-head.mp4) | 物体抬升或保持不足 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0069-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0069-optimized-head.mp4) |
| M0070_S4_clutter_peach | 闭合未到位 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0070-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0070-baseline-head.mp4) | 闭合未到位 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0070-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0070-optimized-head.mp4) |
| M0071_S4_clutter_pear | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0071-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0071-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0071-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0071-optimized-head.mp4) |
| M0072_S4_clutter_potato | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0072-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0072-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0072-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0072-optimized-head.mp4) |
| M0074_S4_clutter_scissors | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0074-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0074-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0074-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0074-optimized-head.mp4) |
| M0076_S4_clutter_spatula | 前提未满足，未执行 | 前提未满足，未执行 |
| M0078_S4_clutter_teapot | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0078-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0078-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0078-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0078-optimized-head.mp4) |
| M0079_S4_clutter_tomato | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0079-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0079-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0079-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0079-optimized-head.mp4) |
| M0081_S5_occluded_apple | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0081-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0081-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0081-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0081-optimized-head.mp4) |
| M0082_S5_occluded_avocado | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0082-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0082-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0082-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0082-optimized-head.mp4) |
| M0084_S5_occluded_banana | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0084-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0084-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0084-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0084-optimized-head.mp4) |
| M0085_S5_occluded_bar_soap | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0085-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0085-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0085-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0085-optimized-head.mp4) |
| M0086_S5_occluded_bell_pepper | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0086-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0086-baseline-head.mp4) | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0086-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0086-optimized-head.mp4) |
| M0087_S5_occluded_book | 前提未满足，未执行 | 前提未满足，未执行 |
| M0089_S5_occluded_bowl | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0089-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0089-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0089-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0089-optimized-head.mp4) |
| M0092_S5_occluded_broccoli | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0092-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0092-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0092-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0092-optimized-head.mp4) |
| M0095_S5_occluded_carrot | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0095-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0095-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0095-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0095-optimized-head.mp4) |
| M0096_S5_occluded_cell_phone | 前提未满足，未执行 | 前提未满足，未执行 |
| M0098_S5_occluded_croissant | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0098-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0098-baseline-head.mp4) | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0098-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0098-optimized-head.mp4) |
| M0099_S5_occluded_cucumber | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0099-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0099-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0099-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0099-optimized-head.mp4) |
| M0100_S5_occluded_donut | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0100-baseline-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0100-baseline-head.mp4) | 抓取成功 · [第三人称](/media/sam3-v8-execution74-20261128-04-m0100-optimized-overview.mp4) · [头部](/media/sam3-v8-execution74-20261128-04-m0100-optimized-head.mp4) |

运行记录：`tmp/state_updates/execution74_20261128_04`。所有视频的路径保存在本轮acceptance.json。

[完整测试报告](/media/sam3-v8-execution74-20261128-04-report.md) · [逐场景统计与核验](/media/sam3-v8-execution74-20261128-04-results.json)

## 内存状态更新：完整规划对照

机器人固定配置保存在内存中，夹爪仅更新原URDF推导的动态参数；场景几何保留在原容量内，通过实际启停标志切换碰撞对象，并及时更新障碍位姿。不变的YAML文本提前生成，每次只替换两个开度数值。配置与回执由单线程后台写入，全部写入并核验后才结束请求。

原五场景、每组重复两次，共20次完整纯规划。两组在共享GPU2交替运行，使用同一常驻规划器、统一预热及原候选顺序。

| 方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 | 累计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原方法 | 10/10 | 7.022秒 | 4.298秒 | 11.455秒 | 70.216秒 |
| 内存状态更新 | 10/10 | 4.403秒 | 3.860秒 | 6.250秒 | 44.033秒 |

### 分项平均耗时

| 环节 | 原方法 | 内存状态更新 |
| --- | ---: | ---: |
| 绑定中其余处理 | 3.5517秒 | 0.0065秒 |
| 创建内存输入 | 0.0000秒 | 0.0390秒 |
| 夹爪动态参数更新 | 0.0000秒 | 0.0252秒 |
| 场景更新 | 0.0000秒 | 0.5658秒 |
| 实测FK与起始碰撞 | 0.0000秒 | 0.0940秒 |
| 接触更新中其余处理 | 0.3882秒 | 0.0007秒 |
| 接触模型动态参数更新 | 0.0000秒 | 0.0035秒 |
| 接触场景更新 | 0.0000秒 | 0.2423秒 |
| 精细几何实测状态更新 | 0.0000秒 | 0.0004秒 |
| 原抓取IK | 0.8585秒 | 1.2746秒 |
| 路径搜索 | 0.1536秒 | 0.1599秒 |
| 预抓取轨迹优化 | 0.2219秒 | 0.1767秒 |
| 接近轨迹优化 | 0.0850秒 | 0.1048秒 |
| 请求模型与资源准入 | 0.0000秒 | 0.0213秒 |
| 记录完成等待与结束核验 | 0.0000秒 | 0.0322秒 |

分项为互斥耗时；原方法的绑定与接触更新尚未拆成上述子项，整组比较以完整请求为准。

### 结果与验证

累计耗时变化-37.29%。完整请求时延门槛通过。

| 项目 | 结果 |
| --- | --- |
| 任务结果 | 20/20个完整纯规划完成 |
| 记录完整性 | 60个独立终点、40条记录轨迹通过 |
| 事件适用性 | 本轮为完整规划与状态更新比较，物理抓取0次 |
| 关键事件覆盖 | 两臂、三种开度、192状态的871球与碰撞判断、四组完整开合爪采样、障碍移动与恢复、写入故障 |
| 抓法与手臂相同 | 是 |
| 关节与记录轨迹逐位相同 | 是 |
| 可选实现采用门槛 | 通过 |

当前默认入口保持。候选姿态、实际开度、871球、TCP43、V2指尖规则与完整路径检查保持。

登记：`tmp/state_updates/memory5_20261127_03`。入口：`modules.curobo.state_updates.runtime.create`。

### 第一版完整结果

第一版20次结果和源码完整保留。以下是第一版自己的同期对照，修订版耗时使用上面的新对照。

| 第一版方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 |
| --- | ---: | ---: | ---: | ---: |
| 原方法 | 10/10 | 6.770秒 | 6.648秒 | 8.865秒 |
| 第一版内存更新 | 10/10 | 6.494秒 | 5.682秒 | 8.906秒 |

第一版累计省4.07%，未达到5%的采用门槛。修订版将YAML不变文本提前生成，并让同一容量中的碰撞对象通过原生启停标志切换，减少重新装载几何。两版累计40个完整请求，新增物理抓取0次。

[实施方法](/media/sam3-v8-memory5-20261127-03-method.md) · [完整报告](/media/sam3-v8-memory5-20261127-03-report.md) · [计时与核验数据](/media/sam3-v8-memory5-20261127-03-results.json)

## 更新后的完整耗时与下一步

修订版在共享GPU2完成20次同期对照。原方法和新方法各10/10规划通过，原抓法、手臂、关节终点与记录轨迹逐位一致。完整请求平均7.022→4.403秒，减少37.29%。

### 完整平均耗时

| 环节 | 原方法 | 内存更新修订版 |
| --- | ---: | ---: |
| 机器人与场景绑定合计 | 3.552秒 | 0.732秒 |
| 接触几何更新合计 | 0.388秒 | 0.247秒 |
| 原抓取IK | 0.859秒 | 1.275秒 |
| 终点补充检查与回退 | 0.177秒 | 0.288秒 |
| 路径搜索与轨迹优化 | 0.460秒 | 0.441秒 |
| 开爪与原地收爪 | 0.282秒 | 0.154秒 |
| 其他已分项的IK、调度与核验 | 0.104秒 | 0.170秒 |
| 其余未分项请求开销 | 1.200秒 | 1.097秒 |
| 完整请求合计 | 7.022秒 | 4.403秒 |

上表互斥项相加等于完整请求。机器人绑定和接触几何更新合计3.940→0.979秒，减少75.16%。其余未分项开销按完整请求减去已记录的互斥分项计算，下一阶段需要继续拆开，不能把它视为已经消失。

### 场景工作量

两版内存更新都实际消费98个候选。第一版10次请求重新装载124套场景；修订版为60套，并完成192次障碍位姿更新和154次原生启停标志更新。原碰撞判断和实时关节状态仍然检查。

第一版仍逐次把约103KB的模型字典转成YAML。修订版预先生成不变文本，运行时只编码两个夹爪开度，后台负责写文件，并在结束请求前等到全部完成。

### 逐场景完整耗时

| 场景 | 原方法两次平均 | 修订版两次平均 |
| --- | ---: | ---: |
| M0001_S1_open_apple | 3.463秒 | 2.772秒 |
| M0020_S1_open_donut | 15.433秒 | 7.856秒 |
| M0044_S3_sparse_banana | 8.622秒 | 5.654秒 |
| M0064_S4_clutter_mango | 3.487秒 | 3.652秒 |
| M0074_S4_clutter_scissors | 4.103秒 | 2.081秒 |

### 下一步优先级

| 顺序 | 内容 | 要解决的耗时 |
| --- | --- | --- |
| 1 | 把请求内的状态映射、候选交接、记录提交及未分项GPU等待分别计时 | 当前其余未分项开销约1.097秒 |
| 2 | 在已验证的内存更新上，比较首候选直接执行、失败后再按4／16扩批的IK策略；保持原精度、开度、评分顺序与回退 | 当前抓取IK约1.275秒 |
| 3 | 继续减少首次场景装载和多套接触视图的重复工作，保留当前碰撞几何 | 当前绑定与接触更新仍约0.979秒 |

当前仍未达到一秒。下一步以整次请求衡量收益，同时保留原成功和完整路径核验。本阶段两个版本累计40次完整纯规划，120个终点和80条记录轨迹核验通过；默认抓取入口保持。

运行记录：`tmp/state_updates/memory5_20261127_03`。

[完整耗时分析](/media/sam3-v8-memory5-20261127-03-analysis.md)

## 固定候选窗口：批量端点与并行路径

新版去掉串行寻找4个合格抓法的过程。只查看下一段16个原候选，在当前需要检查的开度下，批量求抓取、预抓取和抬升关节解，并完成原碰撞检查。已经有多个合格结果时再并行规划；只有一个就立即走原方法。

### 完整请求结果

首次模型加载与统一预热耗时147.27秒，单独记录，不计入下表的单次请求耗时。

五个固定场景，每组重复两次，共30次完整纯规划。各组统一预热，共用原15毫米／15°候选、数值IK设置、871球、TCP43、V2指尖规则及原接近和收爪检查。

本轮在GPU2共享运行。三组在同一卡上交替测试，统一预热，同时记录显存和计算负载。

| 方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 | 累计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原方法 | 10/10 | 7.198秒 | 6.079秒 | 12.328秒 | 71.985秒 |
| 固定窗口批量端点 | 10/10 | 9.235秒 | 8.238秒 | 12.869秒 | 92.353秒 |
| 批量端点＋并行路径4条 | 10/10 | 8.959秒 | 7.626秒 | 12.586秒 | 89.587秒 |

### GPU2运行负载

| 记录项 | 数值 |
| --- | ---: |
| 规划期间采样数 | 207 |
| GPU总利用率中位数（含本轮任务） | 87% |
| 最低空余显存 | 12246 MiB |
| 同时存在的其他计算进程数上限 | 3 |


### 是否仍在额外串行凑批

| 方法 | 实际消费候选数 | 串行抓取终点调用 | 批量计算的抓取终点 | 批量提前算出但未使用 | 为凑批额外串行检查 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原方法 | 98 | 98 | 0 | 0 | 0 |
| 固定窗口批量端点 | 98 | 70 | 84 | 56 | 0 |
| 批量端点＋并行路径4条 | 98 | 70 | 84 | 56 | 0 |

提前批量计算仍可能包含最后没有使用的候选，但不再为这些候选逐个求IK或绑定模型，也不会扩大窗口寻找更多合格抓法。不同开度保持精确分组，没有取整。

### 完整请求分项平均耗时

| 环节 | 原方法 | 批量端点 | 批量端点＋并行路径 |
| --- | ---: | ---: | ---: |
| 模型和场景绑定 | 3.1825秒 | 3.5739秒 | 3.3191秒 |
| 原串行抓取IK | 1.4032秒 | 1.1610秒 | 0.8662秒 |
| 批量抓取IK及碰撞 | 0.0000秒 | 1.1869秒 | 1.3297秒 |
| 批量预抓取IK及碰撞 | 0.0000秒 | 0.3285秒 | 0.3182秒 |
| 批量抬升IK及碰撞 | 0.0000秒 | 0.0219秒 | 0.0280秒 |
| 原预抓取IK | 0.0137秒 | 0.0243秒 | 0.0180秒 |
| 原抬升IK | 0.0323秒 | 0.0307秒 | 0.0242秒 |
| 并行完整路径及核验 | 0.0000秒 | 0.0000秒 | 0.3710秒 |
| 原路径搜索 | 0.1662秒 | 0.2422秒 | 0.1934秒 |
| 原预抓取轨迹优化 | 0.1971秒 | 0.2896秒 | 0.1499秒 |
| 原接近轨迹优化 | 0.0821秒 | 0.1463秒 | 0.1908秒 |
| 接触几何更新 | 0.6427秒 | 0.7911秒 | 0.7831秒 |

### 并行路径实际使用量

| 方法 | 并行批次 | 有效路径行 | 批量通过行 |
| --- | ---: | ---: | ---: |
| 原方法 | 0 | 0 | 0 |
| 固定窗口批量端点 | 0 | 0 | 0 |
| 批量端点＋并行路径4条 | 6 | 20 | 20 |

### 结果分析

- 固定窗口批量端点：累计耗时变化+28.30%；原成功全部保留；未通过累计下降5%、中位下降和P90不增加5%的时延门槛。
- 批量端点＋并行路径4条：累计耗时变化+24.45%；原成功全部保留；未通过累计下降5%、中位下降和P90不增加5%的时延门槛。

从检查数量看，原方法实际检查98个抓取终点；新组合串行检查70个，批量计算84个，合计154个。其中56个批量目标最后没有用到。串行凑批已经去掉，但提前计算的总工作量仍比原方法多。

在本轮计时中，原串行抓取IK平均省下0.537秒，新增的抓取、预抓取和抬升批量检查平均花费1.676秒。新成本超过了省下的部分。

因此本轮保留可选实现，常用入口继续原方法。后续可以先让首个候选完成原检查与路径规划，失败后才对后续候选批量处理；该调整本轮尚未执行。

| 场景／重复 | 原方法 | 批量端点 | 批量端点＋并行路径 | 抓法与手臂一致 |
| --- | --- | --- | --- | --- |
| M0001_S1_open_apple/1 | 通过 | 通过 | 通过 | 是 |
| M0001_S1_open_apple/2 | 通过 | 通过 | 通过 | 是 |
| M0020_S1_open_donut/1 | 通过 | 通过 | 通过 | 是 |
| M0020_S1_open_donut/2 | 通过 | 通过 | 通过 | 是 |
| M0044_S3_sparse_banana/1 | 通过 | 通过 | 通过 | 是 |
| M0044_S3_sparse_banana/2 | 通过 | 通过 | 通过 | 是 |
| M0064_S4_clutter_mango/1 | 通过 | 通过 | 通过 | 是 |
| M0064_S4_clutter_mango/2 | 通过 | 通过 | 通过 | 是 |
| M0074_S4_clutter_scissors/1 | 通过 | 通过 | 通过 | 是 |
| M0074_S4_clutter_scissors/2 | 通过 | 通过 | 通过 | 是 |

### 验收

| 项目 | 结果 |
| --- | --- |
| 任务结果 | 30个完整纯规划槽完成 |
| 记录完整性 | 30条回执、90个独立物理臂终点、60条记录轨迹核验通过 |
| 事件适用性 | 本轮比较规划通过率与完整耗时，物理抓取0次 |
| 关键事件覆盖 | 两臂、三种开度、混合可达/不可达、不同目标初值、固定窗口、单结果立即执行、原评分及回退 |

入口：`modules.curobo.parallel_planning.windowed.create`。登记：`tmp/parallel_planning/shared_gpu2_20261127_01`。50项离线检查通过。

[完整报告](/media/sam3-v8-shared-gpu2-20261127-01-report.md) · [逐组计时与核验](/media/sam3-v8-shared-gpu2-20261127-01-results.json)

## 预抓取并行路径规划

本轮已把预抓取路径接入原生批量PRM与TrajOpt。每次收集4或8个终点合格的抓法，按完全相同的夹爪开度分组，将多组q_pre一起交给GPU生成完整路径，再按原评分顺序选择。

### 完整请求对照

五个固定场景，每组重复两次，共30个完整纯规划请求。两种批容量均提前预热；下表包含双臂准备、候选IK、模型绑定、路径、接近、检查、留档和交接。

| 方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 | 累计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原方法 | 10/10 | 4.110秒 | 3.941秒 | 5.622秒 | 41.100秒 |
| 并行路径4条 | 10/10 | 5.615秒 | 5.603秒 | 6.959秒 | 56.151秒 |
| 并行路径8条 | 10/10 | 6.249秒 | 6.294秒 | 8.361秒 | 62.493秒 |

### 实际并行量与提前检查

| 方法 | 真正的并行批次 | 有效路径行数 | 批量通过行数 | 额外提前检查的候选终点 |
| --- | ---: | ---: | ---: | ---: |
| 并行路径4条 | 6 | 20 | 20 | 82 |
| 并行路径8条 | 8 | 30 | 30 | 132 |

当前六轴模型将夹爪开度锁定为一个值，所以不同开度没有混算。同开度组不足两条时，继续原路径；补齐行不参与选点。批量路径未通过时，仍对同一候选运行原完整路径。

### 单个并行批次耗时

下表为完整一批的中位耗时，包含这一批中的所有有效行与补齐行。

| 方法 | 批量路径搜索 | 批量轨迹优化 | 逐行完整检查 |
| --- | ---: | ---: | ---: |
| 并行路径4条 | 18.3毫秒 | 51.5毫秒 | 146.5毫秒 |
| 并行路径8条 | 28.1毫秒 | 83.7毫秒 | 133.8毫秒 |

### 完整请求的分项平均耗时

| 环节 | 原方法 | 并行4条 | 并行8条 |
| --- | ---: | ---: | ---: |
| 模型和场景绑定 | 1.7673秒 | 2.4172秒 | 2.6468秒 |
| 抓取IK | 0.7925秒 | 1.1709秒 | 1.4262秒 |
| 预抓取IK | 0.0122秒 | 0.1521秒 | 0.1191秒 |
| 抬升IK | 0.0116秒 | 0.0432秒 | 0.0763秒 |
| 并行完整路径及检查 | 0.0000秒 | 0.1314秒 | 0.2454秒 |
| 原路径搜索 | 0.1013秒 | 0.0751秒 | 0.0765秒 |
| 原预抓取轨迹优化 | 0.1047秒 | 0.0644秒 | 0.0594秒 |
| 原接近轨迹优化 | 0.0647秒 | 0.0635秒 | 0.0655秒 |
| 接触几何更新 | 0.3478秒 | 0.4741秒 | 0.5113秒 |

### 结果分析

- 并行路径4条：累计耗时变化+36.62%；原成功全部保留，未通过累计下降5%、中位下降和P90不增加5%的时延门槛。
- 并行路径8条：累计耗时变化+52.05%；原成功全部保留，未通过累计下降5%、中位下降和P90不增加5%的时延门槛。

并行计算节省的时间，需要和提前收集更多合格终点、按开度重新绑定及额外路径计算的成本一起比较。本轮只改变预抓取路径调度，最后接近继续原实现。

并行4条比原方法每请求增加约1.51秒，其中模型／场景绑定增加约0.65秒、抓取IK增加约0.38秒。并行8条增加约2.14秒，其中绑定增加约0.88秒、抓取IK增加约0.63秒。当前按序搜索经常在第一个合格抓法就结束，提前凑齐更多合格终点带来的计算量超过了并行路径节省的时间。因此本轮保留可选入口，常用入口继续原方法。

| 场景／重复 | 原方法 | 并行4条 | 并行8条 | 抓法与手臂一致 |
| --- | --- | --- | --- | --- |
| M0001_S1_open_apple/1 | 通过 | 通过 | 通过 | 是 |
| M0001_S1_open_apple/2 | 通过 | 通过 | 通过 | 是 |
| M0020_S1_open_donut/1 | 通过 | 通过 | 通过 | 是 |
| M0020_S1_open_donut/2 | 通过 | 通过 | 通过 | 是 |
| M0044_S3_sparse_banana/1 | 通过 | 通过 | 通过 | 是 |
| M0044_S3_sparse_banana/2 | 通过 | 通过 | 通过 | 是 |
| M0064_S4_clutter_mango/1 | 通过 | 通过 | 通过 | 是 |
| M0064_S4_clutter_mango/2 | 通过 | 通过 | 通过 | 是 |
| M0074_S4_clutter_scissors/1 | 通过 | 通过 | 通过 | 是 |
| M0074_S4_clutter_scissors/2 | 通过 | 通过 | 通过 | 是 |

### 验收与使用

| 项目 | 结果 |
| --- | --- |
| 任务结果 | 30个完整纯规划槽完成 |
| 记录完整性 | 30条回执、90个物理臂终点和60条记录轨迹核验通过 |
| 事件适用性 | 本轮比较规划通过率和耗时，新增物理抓取0次 |
| 关键事件覆盖 | GPU覆盖两臂、三种预检开度、两种批容量、补齐行和原选点；批量50行全部通过。失败回退和资源异常由离线故障测试覆盖 |

入口：`modules.curobo.parallel_planning.runtime.create`。常用入口保持原实现。本轮登记：`tmp/parallel_planning/paths5_20261127_01`。32项离线检查通过，12组正式预热接口检查通过。

[完整报告](/media/sam3-v8-parallel-path-planning-report.md) · [逐组计时与核验](/media/sam3-v8-parallel-path-planning-results.json)

## 细化NMS与第七项完整规划接入

NMS从30毫米／30°改为15毫米／15°。69例的候选数中位数从54增加到151，保留更多相近但不同的抓法，原评分和宽深修正保持。

第七项在原16/64搜索窗口中批量求抓取、预抓取和抬升终点，再沿原顺序检查完整路径。每候选使用自己的开度，V2指尖网格复核及同候选原方法回退继续执行。两臂Graph在预热时创建，换场景时更新固定缓存，已验证新增障碍能被识别、还原场景后结果保持。

五个固定场景，每组重复两次，本轮共20次完整规划。各组共用相同常驻规划器、C版YAML、原生批量开合爪、逐采样FK及共享内存。下表包含双臂准备、终点检查、完整路径、留档和交接。

| 方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 | 累计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15毫米／15° NMS＋原IK | 10/10 | 4.240秒 | 3.957秒 | 5.887秒 | 42.399秒 |
| 15毫米／15° NMS＋解析批量 | 10/10 | 4.377秒 | 4.040秒 | 5.921秒 | 43.773秒 |

相同细NMS下，解析批量组累计耗时变化为+3.24%。

## 分项平均耗时

分项采用互斥的主机耗时，便于区分解析批量计算、原方法回退和其余路径成本。

| 环节 | 细NMS＋原IK | 细NMS＋解析批量 |
| --- | ---: | ---: |
| 机器人与世界绑定 | 1.8869秒 | 1.8545秒 |
| 开爪检查（扣除子项） | 0.1658秒 | 0.2046秒 |
| 原地收爪检查（扣除子项） | 0.0304秒 | 0.0650秒 |
| 共享场景缓存校验 | 0.0000秒 | 0.0006秒 |
| 解析端点整批查询 | 0.0000秒 | 0.0111秒 |
| 解析分支的原FK与碰撞复核 | 0.0000秒 | 0.0205秒 |
| 原抓取IK | 0.7891秒 | 0.8092秒 |
| 原预抓取IK | 0.0133秒 | 0.0000秒 |
| 原抬升IK | 0.0129秒 | 0.0000秒 |
| 其它原IK回退 | 0.0000秒 | 0.0019秒 |
| 接触几何更新 | 0.3728秒 | 0.3984秒 |
| 路径搜索 | 0.1080秒 | 0.1003秒 |
| 预抓取轨迹优化 | 0.1088秒 | 0.1084秒 |
| 接近轨迹优化 | 0.0678秒 | 0.0636秒 |

## 原方法回退与结果变化

| 解析组事件 | 次数 |
| --- | ---: |
| 抓取终点回退原IK | 84 |
| 抬升终点使用解析解 | 14 |
| 预抓取终点使用解析解 | 12 |
| 预抓取终点回退原IK | 2 |

解析组共检查98次候选，其中84次抓取终点仍回到原数值IK。因此毫秒级批量检查没有消除主要的失败候选求解；逐候选绑定模型和场景仍是完整请求的大头。本轮完成了功能接入，完整耗时没有改善，解析方案保留为独立可选入口，常用入口继续原IK。

细NMS原规划成功是否全部保留：是。
粗NMS沿用前一轮完整结果，本轮只比较相同细NMS下的原IK与解析批量。

| 场景／重复 | 粗NMS原IK | 细NMS原IK | 细NMS解析 | 细NMS两组是否选中同一抓法与手臂 |
| --- | --- | --- | --- | --- |
| M0001_S1_open_apple/1 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0001_S1_open_apple/2 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0020_S1_open_donut/1 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0020_S1_open_donut/2 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0044_S3_sparse_banana/1 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0044_S3_sparse_banana/2 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0064_S4_clutter_mango/1 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0064_S4_clutter_mango/2 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0074_S4_clutter_scissors/1 | 本轮未重跑 | 通过 | 通过 | 是 |
| M0074_S4_clutter_scissors/2 | 本轮未重跑 | 通过 | 通过 | 是 |

## 前一轮三组完整结果

前一轮30次完整规划全部通过。解析组有独立场景副本，平均重复更新约0.206秒，因此平均4.374→4.367秒基本持平。本轮改为直接共享原规划器场景缓存，另做20次同期配对；前一轮记录全部保留。

| 前一轮方法 | 规划通过 | 平均耗时 | 中位耗时 | P90耗时 |
| --- | ---: | ---: | ---: | ---: |
| 30毫米／30° NMS＋原IK | 10/10 | 3.539秒 | 3.361秒 | 5.588秒 |
| 15毫米／15° NMS＋原IK | 10/10 | 4.374秒 | 4.172秒 | 5.927秒 |
| 15毫米／15° NMS＋解析批量 | 10/10 | 4.367秒 | 4.080秒 | 6.058秒 |

[完整接入报告](/media/sam3-v8-nms15-analytic-planning-report.md) · [逐组计时与核验](/media/sam3-v8-nms15-analytic-planning-results.json)

## GPU解析IK与批量碰撞验证

**ARX可以使用解析IK。已完成GPU实现和常驻查询接口。**

真实URDF的第22、23、24关节轴平行，第25、26关节轴相交，可使用[IK-Geo／EAIK](/papers/eaik/)的几何分解。一次并行求256个原目标、每目标最多8个普通分支，再把各分支与对应开爪宽度送进cuRobo的871球碰撞检查。

最终版本用双精度计算解析式，输出单精度关节命令；左右臂共8192个已知可达目标全部找回原分支，329个EAIK精确参考分支全部匹配。三种开度、480个状态与原锁定开度模型的FK及碰撞判断一致。

下表每行是固定256个目标、最多2048分支的整批中位耗时。使用RTX 4090、统一预热及CUDA Graph；每组GPU计时50次。

| 场景／规划手臂 | 解析IK | 原生FK＋世界碰撞 | IK＋FK＋世界碰撞 | IK＋FK＋世界及自碰撞 | CPU接口含上传下载 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 苹果／右 | 1.27毫秒 | 0.87毫秒 | 2.20毫秒 | 2.72毫秒 | 3.86毫秒 |
| 苹果／左 | 1.26毫秒 | 4.16毫秒 | 5.31毫秒 | 5.90毫秒 | 7.38毫秒 |
| 甜甜圈／右 | 1.25毫秒 | 0.46毫秒 | 1.74毫秒 | 2.26毫秒 | 2.66毫秒 |
| 甜甜圈／左 | 1.25毫秒 | 5.26毫秒 | 6.56毫秒 | 7.09毫秒 | 8.34毫秒 |
| 香蕉／右 | 1.19毫秒 | 3.96毫秒 | 5.18毫秒 | 5.71毫秒 | 6.79毫秒 |
| 香蕉／左 | 1.26毫秒 | 0.87毫秒 | 2.20毫秒 | 2.75毫秒 | 4.20毫秒 |
| 芒果／右 | 1.19毫秒 | 4.25毫秒 | 5.42毫秒 | 5.99毫秒 | 7.21毫秒 |
| 芒果／左 | 1.26毫秒 | 2.09毫秒 | 3.34毫秒 | 3.85毫秒 | 4.80毫秒 |
| 剪刀／右 | 1.13毫秒 | 2.85毫秒 | 4.13毫秒 | 4.71毫秒 | 6.83毫秒 |
| 剪刀／左 | 1.13毫秒 | 5.11毫秒 | 6.39毫秒 | 6.96毫秒 | 8.06毫秒 |

CPU接口计时另测10次，包含输入上传、Graph执行、独立输出复制及下载；首次场景装载和Graph捕获单列在逐组JSON中。碰撞时间随障碍网格和球位置变化，解析求解时间相对稳定。

这组测试完成的是终点求解与碰撞模块。当前整次规划还包括开爪检查、预抓取／接近／抬升连续性、V2指尖近桌网格复核及完整路径。因此这里的毫秒值不能替换网页原有约4秒的完整请求计时。

首次创建场景会话并捕获Graph需6.14～6.47秒，后续候选批次复用同一个会话。接入整次规划时，需要同时计入新场景准备和查询开销。

| 验收项目 | 结果 |
| --- | --- |
| 任务结果 | GPU解析求解、原生批量碰撞及常驻接口核验完成 |
| 记录完整性 | 登记、输入、代码、逐组回执及数组散列全部通过 |
| 事件适用性 | 适用于单次终点解析IK与原生碰撞；完整路径和物理结果尚待接入对照 |
| 关键事件覆盖 | 左右臂、原限位、近奇异、无解、无效输入、尾批填充、不同开度、世界/自碰撞及输出所有权 |

[完整方法、接口和核验报告](/media/sam3-v8-gpu-analytic-report.md) · [逐组计时及登记副本](/media/sam3-v8-gpu-analytic-results.json)

## V8当前实验：保留原顺序的计算优化

原序计算优化：规划138/138次，规划核验通过；实抓20/20次，完成、未通过组合启用门槛。

这一轮保持原始抓法评分顺序，只比较计算加速。原来的NMS组合及20次实抓结果完整保留在上一轮章节。

| 项目 | 当前链路 | 本轮优化组 |
| --- | --- | --- |
| 候选选择 | 原始评分顺序 | 相同，NMS关闭 |
| 双臂搜索 | 首轮16、后续64交替 | 相同 |
| 抓取IK | 逐候选 | 每批16个 |
| 配置准备 | 原YAML写出 | C版YAML写出 |
| 网格查询 | 原排列 | 同一网格连续查询 |

先用69份既有观测做138次配对规划，核对原抓法、选臂、关节终点和候选原值。正确性通过，且累计规划省时至少5%、中位数下降、P90增幅不超过5%后，再执行原10个场景的20次实抓对照。

每例使用相同的冻结感知与候选。实抓恢复相同的底盘、升降、物体及关节状态，保持871球、TCP43、V2指尖余量、物理和进入开度。计划抬升3厘米，最低保持抬升1厘米、持续3秒，并满足双指接触及禁止接触检查。初始规划计时包括候选读取、共享内存交接与规划。

## 69场景规划对照

| 方法 | 规划通过 | 累计耗时 | 中位耗时 | P90耗时 |
| --- | ---: | ---: | ---: | ---: |
| 当前链路 | 69/69（100.00%） | 1612.72秒 | 15.08秒 | 67.30秒 |
| 本轮优化组 | 69/69（100.00%） | 619.62秒 | 5.82秒 | 19.86秒 |

累计省时61.58%；138次请求、332016条候选原值和414个关节终点已独立核验。

原抓法与选臂全部保持；关节终点原值有差异，已按原精度独立核验。

两组对应关节终点的最大差为0.00000381弧度。

34个场景的中间候选分支数量或原因记录有差异，其中3个场景的候选失败类型序列不同。最终69个场景选中的抓法和手臂均保持。

## 10场景实际抓取对照

| 方法 | 规划通过 | 实际成功 | 初始规划中位数 | 初始规划P90 |
| --- | ---: | ---: | ---: | ---: |
| 当前链路 | 10/10（100%） | 8/10（80%） | 38.75秒 | 89.58秒 |
| 本轮优化组 | 10/10（100%） | 7/10（70%） | 14.98秒 | 30.68秒 |

初始规划累计省时65.56%，原成功变失败1例，原失败变成功0例。

### 任务与事件核验

| 项目 | 结果 |
| --- | --- |
| 任务完成情况 | 20/20完成，中断0、未开始0。 |
| 记录完整性 | 20个槽位的命令、回执与评估一致；40段原视频解码通过。 |
| 事件适用性 | 按实际轨迹和阶段报告核对已执行动作；提前结束后未执行的抬升、保持单列。 |
| 关键事件覆盖 | 两组逐阶段实际执行数量见下表。 |

| 实际阶段 | 当前链路 | 本轮优化组 |
| --- | ---: | ---: |
| 张开 | 10 | 10 |
| 预抓取 | 10 | 10 |
| 接近 | 10 | 10 |
| 闭合 | 10 | 10 |
| 抬升 | 8 | 8 |
| 保持 | 8 | 8 |

### 杂乱剪刀：同一抓法仍有夹持保持差异

M0074两组都选第166号抓法。预抓取和接近期间，物体逐采样位置完全一致；实际接近轨迹的两组关节最大差约0.00000453弧度。闭合接触后，物体运动开始分岔。

| 观测 | 当前链路 | 本轮优化组 |
| --- | ---: | ---: |
| 闭合期间物体最大位移 | 12.57毫米 | 18.93毫米 |
| 闭合结束两指位置 | 4.58、4.06毫米 | 0.52、0.45毫米 |
| 抬升期双指同时有有效接触的采样 | 723/723 | 315/723 |
| 保持期双指同时有有效接触的采样 | 363/363 | 0/363 |

两组物理配置记录完全相同。新基线的逐采样关节状态和物体位姿，也与上一轮基线完全一致。两组闭合时形成了不同的接触状态；优化组随后在抬升过程中脱离，保持期物体已回到桌面。

本轮去掉NMS后仍有一例退步，需要继续定位同一抓法下的细小指令差异与接触保持过程。现有记录把分歧定位到了闭合接触，具体因果还需要后续固定命令回放来区分。

### 本轮结论与下一步

本轮优化组合保留为可选，默认链路保持。下一步优先固定同一成功抓法的运动命令，隔离微小指令差异与夹持接触稳定性，再决定计算优化如何接入常规抓取。

## 任务与记录

规划阶段逐项核对领取、候选、实际求解配置和独立FK。实抓按实际发生的接近、闭合、抬升与保持动作核对命令、回执和视频。普通失败如实记录，资源、程序或记录故障停止，已领取任务不重试。默认配置保持。

### 逐场景结果和录像

| 场景 | 当前链路 | 本轮优化组 |
| --- | --- | --- |
| M0020_S1_open_donut | 成功 · [第三人称](/media/sam3-v8-raw-order-m0020-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0020-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0020-baseline-initial.png) | 成功 · [第三人称](/media/sam3-v8-raw-order-m0020-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0020-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0020-optimized-initial.png) |
| M0034_S2_edge_scissors | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-raw-order-m0034-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0034-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0034-baseline-initial.png) | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-raw-order-m0034-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0034-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0034-optimized-initial.png) |
| M0074_S4_clutter_scissors | 成功 · [第三人称](/media/sam3-v8-raw-order-m0074-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0074-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0074-baseline-initial.png) | 抬升或保持不足 · [第三人称](/media/sam3-v8-raw-order-m0074-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0074-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0074-optimized-initial.png) |
| M0084_S5_occluded_banana | 成功 · [第三人称](/media/sam3-v8-raw-order-m0084-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0084-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0084-baseline-initial.png) | 成功 · [第三人称](/media/sam3-v8-raw-order-m0084-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0084-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0084-optimized-initial.png) |
| M0085_S5_occluded_bar_soap | 成功 · [第三人称](/media/sam3-v8-raw-order-m0085-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0085-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0085-baseline-initial.png) | 成功 · [第三人称](/media/sam3-v8-raw-order-m0085-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0085-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0085-optimized-initial.png) |
| M0099_S5_occluded_cucumber | 成功 · [第三人称](/media/sam3-v8-raw-order-m0099-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0099-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0099-baseline-initial.png) | 成功 · [第三人称](/media/sam3-v8-raw-order-m0099-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0099-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0099-optimized-initial.png) |
| M0001_S1_open_apple | 成功 · [第三人称](/media/sam3-v8-raw-order-m0001-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0001-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0001-baseline-initial.png) | 成功 · [第三人称](/media/sam3-v8-raw-order-m0001-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0001-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0001-optimized-initial.png) |
| M0044_S3_sparse_banana | 成功 · [第三人称](/media/sam3-v8-raw-order-m0044-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0044-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0044-baseline-initial.png) | 成功 · [第三人称](/media/sam3-v8-raw-order-m0044-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0044-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0044-optimized-initial.png) |
| M0056_S3_sparse_cell_phone | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-raw-order-m0056-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0056-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0056-baseline-initial.png) | 闭合后未形成双指夹持 · [第三人称](/media/sam3-v8-raw-order-m0056-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0056-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0056-optimized-initial.png) |
| M0064_S4_clutter_mango | 成功 · [第三人称](/media/sam3-v8-raw-order-m0064-baseline-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0064-baseline-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0064-baseline-initial.png) | 成功 · [第三人称](/media/sam3-v8-raw-order-m0064-optimized-overview.mp4) · [头部](/media/sam3-v8-raw-order-m0064-optimized-head.mp4) · [初始图](/media/sam3-v8-raw-order-m0064-optimized-initial.png) |

### 失败分布

| 原因 | 当前链路 | 本轮优化组 |
| --- | ---: | ---: |
| 抬升或保持不足 | 0 | 1 |
| 闭合后未形成双指夹持 | 2 | 2 |

### 原成功、新失败的实际对比

左侧为当前链路，右侧为本轮优化组。

[M0074_S4_clutter_scissors · 同步对比](/media/sam3-v8-raw-order-m0074-comparison.mp4)


[分析记录：physical_analysis.json](/media/sam3-v8-raw-order-physical-analysis.json)

[分析记录：scissors_hold_diagnosis.json](/media/sam3-v8-raw-order-scissors-hold-diagnosis.json)

[分析记录：scissors_command_contact_analysis.json](/media/sam3-v8-raw-order-scissors-command-contact-analysis.json)

[本轮说明与结果](/media/sam3-v8-raw-order.md) · [逐例统计和独立核验](/media/sam3-v8-raw-order.json)

[关节差异与中间候选记录](/media/sam3-v8-raw-order-numerical-analysis.json)

## 上一轮：含NMS的优化组合实际抓取

实际抓取对照：20/20次，独立核验完成，优化组合未保留全部原成功。

选取10个场景，两组各执行一次。其中包括全部6个此前NMS改变抓法的场景，并覆盖五种布局。两组恢复相同的最终站位、升降和物体状态，使用相同的冻结感知与原始候选，再各自规划、实际抓取。

| 项目 | 当前链路 | 优化组合 |
| --- | --- | --- |
| 双臂搜索 | 首轮16、后续64交替 | 相同 |
| 抓取IK | 单候选 | 每批16个候选 |
| 候选次序 | 原评分次序 | NMS15代表优先，原候选全部保留 |
| 配置写出 | 原方式 | C版YAML |
| 网格查询 | 原线程排列 | 同一网格连续查询 |

本轮现用链路实际成功8/10，优化组合6/10。优化组合更快，但没有保留全部原成功案例，默认链路保持。

| 方法 | 规划通过 | 实际抓取成功 | 初始规划中位数 | 初始规划P90 |
| --- | ---: | ---: | ---: | ---: |
| 当前链路 | 10/10（100%） | 8/10（80%） | 49.128秒 | 122.686秒 |
| 优化组合 | 10/10（100%） | 6/10（60%） | 17.774秒 | 30.860秒 |

包含NMS及共享内存交接的初始规划，累计省时69.75%。原成功变失败2例，原失败变成功0例。

### 失败分布

| 原因 | 当前链路 | 优化组合 |
| --- | ---: | ---: |
| 接近动作未到位 | 0 | 1 |
| 抬升或保持不足 | 0 | 1 |
| 闭合后未形成双指夹持 | 2 | 2 |

### 逐场景结果与视频

| 场景 | 当前链路 | 优化组合 |
| --- | --- | --- |
| M0020 · 甜甜圈 | 成功，初始规划173.17秒 · [第三人称](/media/sam3-v8-physical-m0020-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0020-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0020-baseline-initial.png) | 接近动作未到位，初始规划29.38秒 · [第三人称](/media/sam3-v8-physical-m0020-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0020-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0020-optimized-initial.png) |
| M0034 · 剪刀 | 闭合后未形成双指夹持，初始规划78.58秒 · [第三人称](/media/sam3-v8-physical-m0034-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0034-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0034-baseline-initial.png) | 抬升或保持不足，初始规划26.39秒 · [第三人称](/media/sam3-v8-physical-m0034-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0034-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0034-optimized-initial.png) |
| M0074 · 剪刀 | 成功，初始规划56.62秒 · [第三人称](/media/sam3-v8-physical-m0074-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0074-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0074-baseline-initial.png) | 闭合后未形成双指夹持，初始规划40.30秒 · [第三人称](/media/sam3-v8-physical-m0074-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0074-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0074-optimized-initial.png) |
| M0084 · 香蕉 | 成功，初始规划41.64秒 · [第三人称](/media/sam3-v8-physical-m0084-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0084-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0084-baseline-initial.png) | 成功，初始规划21.22秒 · [第三人称](/media/sam3-v8-physical-m0084-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0084-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0084-optimized-initial.png) |
| M0085 · 香皂 | 成功，初始规划111.74秒 · [第三人称](/media/sam3-v8-physical-m0085-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0085-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0085-baseline-initial.png) | 成功，初始规划14.33秒 · [第三人称](/media/sam3-v8-physical-m0085-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0085-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0085-optimized-initial.png) |
| M0099 · 黄瓜 | 成功，初始规划12.46秒 · [第三人称](/media/sam3-v8-physical-m0099-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0099-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0099-baseline-initial.png) | 成功，初始规划6.12秒 · [第三人称](/media/sam3-v8-physical-m0099-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0099-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0099-optimized-initial.png) |
| M0001 · 苹果 | 成功，初始规划5.71秒 · [第三人称](/media/sam3-v8-physical-m0001-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0001-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0001-baseline-initial.png) | 成功，初始规划5.02秒 · [第三人称](/media/sam3-v8-physical-m0001-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0001-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0001-optimized-initial.png) |
| M0044 · 香蕉 | 成功，初始规划7.29秒 · [第三人称](/media/sam3-v8-physical-m0044-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0044-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0044-baseline-initial.png) | 成功，初始规划6.08秒 · [第三人称](/media/sam3-v8-physical-m0044-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0044-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0044-optimized-initial.png) |
| M0056 · 手机 | 闭合后未形成双指夹持，初始规划117.08秒 · [第三人称](/media/sam3-v8-physical-m0056-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0056-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0056-baseline-initial.png) | 闭合后未形成双指夹持，初始规划29.81秒 · [第三人称](/media/sam3-v8-physical-m0056-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0056-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0056-optimized-initial.png) |
| M0064 · 芒果 | 成功，初始规划9.01秒 · [第三人称](/media/sam3-v8-physical-m0064-baseline-overview.mp4) · [头部](/media/sam3-v8-physical-m0064-baseline-head.mp4) · [初始图](/media/sam3-v8-physical-m0064-baseline-initial.png) | 成功，初始规划6.85秒 · [第三人称](/media/sam3-v8-physical-m0064-optimized-overview.mp4) · [头部](/media/sam3-v8-physical-m0064-optimized-head.mp4) · [初始图](/media/sam3-v8-physical-m0064-optimized-initial.png) |

### 原成功、新失败的同步对比视频

左侧为现用链路，右侧为优化组合。视频标有实际结果和初始规划耗时，较短的一侧结束后停留在最后一帧。

[甜甜圈（M0020）· 同步对比](/media/sam3-v8-physical-m0020-comparison.mp4)

[剪刀（M0074）· 同步对比](/media/sam3-v8-physical-m0074-comparison.mp4)

### 抓法变化与实际抓取

| 场景 | 原候选 → 新候选 | 抓点位置变化 | 抓点朝向变化 | 当前链路 → 优化组合 |
| --- | ---: | ---: | ---: | --- |
| M0020 · 甜甜圈 | 375 → 380 | 12.25毫米 | 14.89° | 成功 → 接近动作未到位 |
| M0034 · 剪刀 | 128 → 145 | 30.00毫米 | 0.00° | 闭合后未形成双指夹持 → 抬升或保持不足 |
| M0074 · 剪刀 | 166 → 869 | 10.00毫米 | 40.00° | 成功 → 闭合后未形成双指夹持 |
| M0084 · 香蕉 | 110 → 308 | 56.27毫米 | 148.94° | 成功 → 成功 |
| M0085 · 香皂 | 261 → 300 | 15.16毫米 | 7.45° | 成功 → 成功 |
| M0099 · 黄瓜 | 10 → 13 | 70.74毫米 | 12.11° | 成功 → 成功 |
| M0001 · 苹果 | 1 → 1 | 0.00毫米 | 0.00° | 成功 → 成功 |
| M0044 · 香蕉 | 0 → 0 | 0.00毫米 | 0.00° | 成功 → 成功 |
| M0056 · 手机 | 340 → 340 | 0.00毫米 | 0.00° | 闭合后未形成双指夹持 → 闭合后未形成双指夹持 |
| M0064 · 芒果 | 13 → 13 | 0.00毫米 | 0.00° | 成功 → 成功 |

抓法和选臂未变的4个场景，两组均成功3/4，逐例结果一致。换了抓法的6个场景，成功从5/6降至3/6。

### 两例退步发生在哪里

**甜甜圈 M0020：接近时推偏物体，未通过到位检查。** 抓法由原候选375换成380，位置相差12.25毫米、朝向相差14.89°。接近期间物体最大移动量由0.84增至10.32毫米。优化组接近结束的最大关节误差为0.007430弧度，超过原有0.005弧度门槛；因此在闭合之前结束。当前链路则完成了闭合、抬升和保持。

**杂乱剪刀 M0074：到位后闭合，没有形成双指夹持。** 抓法由原候选166换成869，位置相差10.00毫米、朝向相差40.00°。优化组接近末端位置误差约0.000377毫米，到位检查通过；随后闭合未形成双指夹持，未进入抬升。当前链路成功带起并保持。

| 退步场景 | 旧成功候选仍在池中 | NMS后的所在区域 | 旧成功候选的新优先名次（从0计） | 实际先选中的新候选 |
| --- | --- | --- | ---: | ---: |
| M0020 | 是 | 后备区 | 760 | 380 |
| M0074 | 是 | 后备区 | 909 | 869 |

NMS保留全部后备，保证了候选仍可被查询；先接受新代表后，搜索就结束了，旧成功抓法没有获得实际执行机会。两例都保持了原夹爪进入开度：单指0.044米、总宽0.088米。

本轮证据指向候选重排带来的抓法变化。四个抓法未变的对照逐例一致；B16、配置写出和网格排列也一同改变了，单靠这一轮还不能分别计算各项对物理结果的影响。

### 任务、记录与事件覆盖

| 核验项目 | 本轮结果 |
| --- | --- |
| 任务结果 | 20/20完成；中断0，未开始0。两组规划均10/10，实际成功8/10与6/10。 |
| 记录完整性 | 20条领取与结果、20次初始状态恢复、命令与回执逐项核对；40段原视频解码通过。 |
| 事件适用性 | 按实际轨迹与阶段报告核对已发生的动作。1例在接近结束，4例在闭合结束；后续未执行的阶段单列。 |
| 关键事件覆盖 | 初始规划20次，接近重规划20次；抬升重规划15次。逐阶段执行数量见下表。 |

| 实际执行阶段 | 当前链路 | 优化组合 |
| --- | ---: | ---: |
| 张开 | 10 | 10 |
| 到预抓取位 | 10 | 10 |
| 接近 | 10 | 10 |
| 闭合 | 10 | 9 |
| 抬升 | 8 | 7 |
| 保持 | 8 | 7 |

### 下一步建议

保留当前默认链路。下一轮优先保持原始评分顺序，只叠加B16、C版YAML与新网格排列，将提速和候选重排分开验证；先检查本轮两例退步，再决定更大范围测试。NMS代表优先的抓取质量另设对照。

[实施设置与结果](/media/sam3-v8-physical-pilot.md) · [统计与核验数据](/media/sam3-v8-physical-pilot.json)

## 已完成：网格查询定位与线程排列优化

线程排列优化：138/138次，独立核验完成。

原五个场景的10次定位已经完成，30个终点独立核验通过。用每例首批完整IK的实际球和网格，分开测量最近表面搜索和三方向内外判断。

| 场景 | 最近表面搜索 | 三方向内外判断 | 合并查询 | 原完整碰撞派发 |
| --- | ---: | ---: | ---: | ---: |
| M0001_S1_open_apple | 0.898毫秒 | 0.905毫秒 | 1.717毫秒 | 3.150毫秒 |
| M0032_S2_edge_potato | 2.521毫秒 | 0.841毫秒 | 3.603毫秒 | 9.928毫秒 |
| M0044_S3_sparse_banana | 3.802毫秒 | 0.702毫秒 | 4.759毫秒 | 7.371毫秒 |
| M0064_S4_clutter_mango | 5.318毫秒 | 0.804毫秒 | 6.585毫秒 | 8.312毫秒 |
| M0084_S5_occluded_banana | 1.925毫秒 | 0.894毫秒 | 2.836毫秒 | 4.658毫秒 |

四个场景的最近表面搜索更耗时，苹果场景两项接近。本轮只调整GPU任务顺序，让相邻线程连续处理同一个障碍物上的球。所有球与障碍物组合、几何、距离公式、三方向射线及精度门槛保持。

原排列与新排列共用NMS15＋全部原候选后备、B16及C版YAML。完整IK分别预热并保存CUDA Graph。上表为固定查询的GPU分项时间，完整规划请求另表比较。

预检已核对8,919,040个实际IK碰撞球，完整碰撞代价与梯度一致。

| 方法 | 规划通过 | 完整请求中位耗时 | 完整请求P90 | 完整请求累计耗时 |
| --- | ---: | ---: | ---: | ---: |
| 原线程排列 | 69/69 | 6.241秒 | 19.721秒 | 639.918秒 |
| 同一障碍物连续排列 | 69/69 | 5.276秒 | 13.836秒 | 488.707秒 |

完整请求累计省时23.63%。

### 哪个环节变快了

| 环节 | 原排列：每次请求平均 | 新排列：每次请求平均 |
| --- | ---: | ---: |
| 成批抓取IK，含场景碰撞 | 4.1059秒 | 2.0052秒 |
| 纯运动学IK回退 | 0.1339秒 | 0.1329秒 |
| 机器人与场景绑定 | 1.4383秒 | 1.4270秒 |
| 局部接触世界绑定 | 0.3856秒 | 0.3823秒 |
| 预抓取IK | 0.0313秒 | 0.0313秒 |
| 抬升IK | 0.0168秒 | 0.0169秒 |
| 到预抓取位置的路径搜索 | 0.1061秒 | 0.1024秒 |

表中记录各环节实际调用时间，其中部分步骤互相包含。完整请求的总耗时以上一表为准。

### 不同场景的耗时

| 场景类型 | 配对场景数 | 原排列平均每次 | 新排列平均每次 | 累计耗时变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单物体 | 15 | 9.033秒 | 6.872秒 | 减少23.9% |
| 桌沿 | 14 | 8.249秒 | 6.376秒 | 减少22.7% |
| 稀疏摆放 | 15 | 9.439秒 | 7.294秒 | 减少22.7% |
| 杂乱摆放 | 14 | 8.661秒 | 6.824秒 | 减少21.2% |
| 遮挡 | 11 | 11.463秒 | 8.311秒 | 减少27.5% |

各组统计同一批场景的完整请求。每个场景的原抓点、选臂和关节终点另行逐项核验。

414个终点独立核验通过，原候选、抓法、选臂、关节解和候选判断保持。

先导按ABBA／BAAB完成20次。累计至少省5%、中位耗时下降、P90增幅不超过5%，并通过独立核验后，才运行69场景的138次完整对照。

[实施方案](/media/sam3-v8-kernel-layout-plan.md) · [方法与结果](/media/sam3-v8-kernel-layout.md) · [计时与核验数据](/media/sam3-v8-kernel-layout.json)

## 上一轮：三方向射线投票优化

网格查询优化：20/20次，独立核验完成，未扩展69场景。

原五场景10次定位全部通过，30个终点核验一致。五批采样中，碰撞查询占GPU实际内核时间97.7%，CUDA Graph重新捕获0次。

原实现沿三个方向投票，判断点在网格内部还是外部。新实现当前两票一致时立即返回，意见不一致时继续第三票。最近点、射线方向、几何和碰撞标准保持。两组分别保留CUDA Graph，共同使用NMS15＋全部原候选后备、B16及C版YAML。

46125个实际网格查询点的距离和梯度逐位一致；完整场景碰撞代价与梯度复核通过。

| 方法 | 规划通过 | 请求中位耗时 | P90 | 累计耗时 |
| --- | ---: | ---: | ---: | ---: |
| 原三方向完整投票 | 10/10 | 4.686秒 | 18.804秒 | 74.819秒 |
| 一致两票提前返回 | 10/10 | 4.819秒 | 17.995秒 | 72.434秒 |

完整请求累计省时3.19%。

60个终点独立核验通过，原抓点、选臂、关节解和候选判断保持。

本轮累计省时和中位耗时未同时达标，先导在20次结束，保留原组合。

先导五场景按ABBA／BAAB运行20次，关闭性能分析器。累计至少省5%、中位数下降、P90增幅不超过5%，并通过独立核验后，才开展69场景完整对照。默认链路保持，机器人动作0次。

[定位与优化方法、结果](/media/sam3-v8-mesh-vote.md) · [实际计时和进度](/media/sam3-v8-mesh-vote.json) · [GPU内核定位数据](/media/sam3-v8-ik-hotspot.json)

[先导的逐场景效果与下一步分析](/media/sam3-v8-mesh-vote-analysis.md)

## V8当前实验：配置准备提速

配置准备提速完整69场景对照：138/138次，独立核验完成。

上一轮的机器人和场景绑定平均约3秒。本轮把约100KB机器人配置的写出交给C版YAML库，保留原绑定函数、配置读取缓存、机器人参数、FK、碰撞、IK及路径检查。两组共同使用NMS15＋B16。

全部69个场景的138份左右臂配置，加上6组闭爪和目标参与组合，144对CPU核对全部一致。单次配置准备中位141→42毫秒；完整请求收益由本轮测量。

继承109次完成记录，本次只执行原未执行的29次，原批次停止标记保留。M0072这一对跨服务重启，保留完整规划结果，主延时表使用其余68对同次常驻运行的场景。

完整规划结果：原版69/69，C版69/69。

| 方法 | 计时请求 | 计时样本规划通过 | RPC中位耗时 | RPC P90 | RPC累计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原版配置写出 | 68 | 68 | 9.721秒 | 27.208秒 | 965.508秒 |
| C版配置写出 | 68 | 68 | 8.707秒 | 24.587秒 | 866.490秒 |

主表按从发送规划请求到RPC返回计时，包含规划、组件记录与候选快照提交。

414个终点独立核验通过，414对实际配置值一致，累计耗时减少10.26%。

先导按ABBA／BAAB交替，共20次。累计至少省5%、中位数下降、P90增幅不超过5%，三项都满足才另登记完整69场景对照；本轮不自动切换默认链路。

[配置准备提速方法与结果](/media/sam3-v8-binding-serialization.md) · [完整请求进度与计时](/media/sam3-v8-binding-serialization.json) · [144组CPU核对](/media/sam3-v8-binding-serialization-cpu.json)

## V8正在推进：抓取分支批量复核

抓取分支批量复核完整69场景对照与独立核验完成。

同一个抓取位姿往往有多组关节解。原版逐组做FK和碰撞检查；本轮改成批量复核，仍按原顺序取第一个通过的解。两组都使用NMS15＋批量16 IK。原69份观测全部保留，每场景各1次，按AB／BA交替顺序完成138次纯规划对照。

新增计时覆盖每次场景绑定、独立FK、局部碰撞、预抓取与抬升IK、PRM及轨迹优化。

先导两组各10/10通过，60个终点独立核验通过，累计耗时96.56→91.27秒，省时5.47%；中位耗时6.06→6.92秒。完整对照继续检查收益能否保持。

[先导方法与结果](/media/sam3-v8-branch-validation-pilot.md) · [先导数据](/media/sam3-v8-branch-validation-pilot.json)

| 方法 | 完整请求 | 规划通过 | 中位耗时 | P90耗时 | 累计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原版逐分支复核 | 69 | 69 | 9.979秒 | 31.762秒 | 1069.370秒 |
| 批量复核 | 69 | 69 | 11.011秒 | 31.665秒 | 1031.875秒 |

表中每条样本是一整个场景的双臂规划请求，包含开爪、候选搜索、IK、碰撞和路径。

原抓点、选臂、关节终点和候选拒绝分类全部一致，414个终点独立FK／限位核验通过。
累计耗时减少3.51%。

32个场景更快、37个更慢。收益集中在部分慢场景，中位耗时有所增加，P90基本持平；这项实现先保留为可选方案。

完整对照已完成，按全部输入的耗时和稳定性确定后续组合。

[本轮方法与结果](/media/sam3-v8-branch-validation-full69.md) · [先导数据与分项计时](/media/sam3-v8-branch-validation-full69.json)

## V8后续阶段：完整69例定位已完成

已从V7完整138次规划记录中，拆分69个配对场景的耗时和终点拒绝原因。根据定位开展抓取分支批量复核先导，进度见本页最上方。

| NMS15＋B16的环节 | 每场景平均耗时 |
| --- | ---: |
| 双臂开爪检查 | 3.027秒 |
| 候选搜索 | 9.666秒 |
| NMS与候选队列构造 | 0.356秒 |
| 共享候选读取与解码 | 0.226秒 |
| 其余状态准备与请求记录 | 0.398秒 |
| 合计 | 13.674秒 |

候选搜索中的批量抓取IK，GPU事件计时平均6.24秒；这是当前优先核对的计算开销。

| 候选终点检查结果 | 次数 |
| --- | ---: |
| 本次未返回合格运动学解 | 5505 |
| 有解，被桌面碰撞拒绝 | 750 |
| 有解，被非桌面场景碰撞拒绝 | 284 |
| 抓取终点通过 | 108 |

每次绑定、预抓取/抬升IK和PRM的计时已接入本轮先导，抓取分支批量复核正在对照。继续保留原候选、最终精度、871球和V2规则。

[完整69例耗时定位报告](/media/sam3-v8-full69-funnel-diagnosis.md) · [逐例数据与拒绝来源](/media/sam3-v8-full69-funnel-diagnosis.json) · [V7已完成的NMS规划对照](/iterations/sam3-graspgenx/versions/V7/)

## 本版进度

E：原目标IK初值对照3,132/3,132次，比较3°、5°、10°查询范围。

[本轮进度与逐例数据](/media/sam3-v8-index-seed-probe-20261126-02-status.json)

## E近邻初值的实际IK对照

同一261个原始终点，各组重复3次；每目标仍使用32个初值，地图最多提供其中8个。逐次核对初值实际进入原生LM，并独立复算返回关节解。

| 方法 | 求出合格解 | 使用地图初值的调用 | 中位耗时 | P90耗时 | 合计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原IK初值 | 345/783 | 0 | 21.59毫秒 | 41.81毫秒 | 19.41秒 |
| 3°近邻初值 | 345/783 | 0 | 23.01毫秒 | 51.53毫秒 | 20.26秒 |
| 5°近邻初值 | 345/783 | 12 | 22.50毫秒 | 42.36毫秒 | 19.78秒 |
| 10°近邻初值 | 345/783 | 99 | 22.67毫秒 | 41.23毫秒 | 19.45秒 |

表中计时包含查询、初值交接和IK。这一项先比较原目标的运动学求解，是否值得接入完整规划，按本轮耗时及原成功保留情况确定。

四组解出的终点相同，原成功全部保留。当前地图初值策略没有带来总耗时收益，保留可选接口，本轮未追加完整规划。后续优先保留B批量16。


## C上下文复用完整结果

| 方法 | 完整规划次数 | 规划通过 | 中位耗时 | P90耗时 | 合计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 现行逐个检查 | 69 | 69 | 14.18秒 | 81.54秒 | 2008.41秒 |
| 左右臂上下文复用 | 69 | 69 | 15.86秒 | 76.75秒 | 1961.70秒 |

表中时间是一整个场景返回完整规划的耗时，包含双臂搜索、开爪检查、IK、碰撞与路径。P90同样按完整请求统计。

累计耗时降低2.33%，完整重绑定584→414次。

[逐例结果](/media/sam3-v8-binding-reuse-69-20261126-01-status.json) · [方法说明](/media/sam3-v8-binding-reuse-method.md)

中位耗时高于同期控制组，当前保留为可选方案。后续优先组合批量16与其他实测有效的模块。

## D几何IK先导结果

| 方法 | 完整规划次数 | 规划通过 | 中位耗时 | P90耗时 | 合计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 现行逐个检查 | 10 | 10 | 7.75秒 | 67.61秒 | 204.83秒 |
| 几何IK优先，原IK回退 | 10 | 10 | 7.34秒 | 74.19秒 | 217.42秒 |

表中时间是一整个场景返回完整规划的耗时，包含双臂搜索、开爪检查、IK、碰撞与路径。P90同样按完整请求统计。

两组各10/10通过，原抓姿与选臂相同，60个终点独立复算通过。累计慢6.14%，当前不纳入组合，也未启动69场景扩展。

[逐次记录](/media/sam3-v8-geometric-pilot-20261126-01-status.json) · [几何IK说明](/media/sam3-v8-geometric-ik-method.md)

## E地图近邻查询结果

在69份保存观测中，重放原控制组实际检查过的10,322个候选。查询位置范围为2厘米，比较完整三维朝向。

| 查询角度范围 | 找到近邻的候选数 | 比例 |
| --- | ---: | ---: |
| 1.72° | 13 / 10,322 | 0.13% |
| 3.00° | 32 / 10,322 | 0.31% |
| 5.00° | 142 / 10,322 | 1.38% |
| 10.00° | 698 / 10,322 | 6.76% |

1.72°用于邻近初值查询过于严格。本轮扩大到3°、5°、10°，取出附近的关节角，再让IK求解原抓取位姿。最终位置误差2毫米、朝向误差1.72°与原抓姿保持。

直接满足原抓姿精度的地图记录命中0次。近邻查询中位耗时0.529毫秒；旧8192位置近邻加多分支选择为17.37毫秒。

[地图索引实施说明](/media/sam3-v8-index-reuse-method.md)

## B原生批量IK完整结果

| 方法 | 完整规划次数 | 规划通过 | 中位耗时 | P90耗时 | 合计耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 现行逐个检查 | 69 | 69 | 15.56秒 | 75.98秒 | 2151.76秒 |
| 批量16个 | 69 | 69 | 11.03秒 | 56.78秒 | 1445.20秒 |
| 批量32个 | 69 | 69 | 15.47秒 | 54.87秒 | 1650.48秒 |

表中时间是一整个场景返回完整规划的耗时，包含双臂搜索、开爪检查、IK、碰撞与路径。P90同样按完整请求统计。

三组各69/69通过，原抓姿和选臂全部一致。621个终点独立复算通过。批量16合计耗时减少32.84%，中位数降低29.12%；后续组合优先使用批量16。

[完整结果与分析](/media/sam3-v8-native-batch-final.md) · [69例逐项数据](/media/sam3-v8-native-batch-69-20261126-01-status.json) · [621个终点复算](/media/sam3-v8-native-batch-fk-review.json)

## 后续方向：分级筛选与批量路径

V7已完成候选去重与原候选后备的69场景对照，NMS15与B16组合累计省时28.65%，两组各69/69通过。接下来推进完整抓法联合检查，以及直接复用预抓取关节解的小批量路径规划。

| 工作 | 安排 |
| --- | --- |
| 夹爪粗筛和代表候选去重 | V7离线和138次规划完成，保留NMS15；几何硬筛误删3条原成功抓法，暂缓启用 |
| 预抓取、抓取和抬升联合检查 | 基于B16补批量处理，保留同一组关节分支 |
| 少量已知关节终点并行规划 | 优先评估本地批量关节空间接口，减少重复IK |
| Goalset | 单独比较选点与避障表现，保留原路径回退 |

原IK能解出的115个终点中，有85个在2厘米/10°地图邻域里没有记录。地图空值继续表示未知，不能据此删除候选。

[建议核对与实施顺序](/media/sam3-v8-funnel-review.md) · [本地接口与反例数据](/media/sam3-v8-funnel-source-review.json)

## 优化目标

把候选检查与整次规划请求做快，同时保留原抓姿、评分顺序和规划效果。以已启用的双臂交替搜索为基线，分别测量批量IK、几何IK、地图复用和绑定复用，最后组合有效部分。

双臂交替搜索的香蕉428.23→8.55秒已归入[V3双臂版本](/iterations/sam3-graspgenx/versions/V3/)。V8各组都使用相同的首轮16、后续64调度，比较在此基础上还能节省多少时间。

## 优先比较什么

| 分组 | 具体做法 | 想解决的耗时 | 当前进度 |
| --- | --- | --- | --- |
| A：现行基线 | 当前交替搜索、逐候选IK与原碰撞/路径检查 | 建立优化后的真实对照 | 69份回放完成 |
| B：原生批量IK | 一次处理16或32个目标，各自保留编号与结果 | 逐个调用小规模求解的开销 | 207次对照完成，优先保留批量16 |
| C：绑定复用 | 同一请求内复用每臂的模型和场景上下文 | 交替窗口反复绑定世界 | 138次完整对照已完成 |
| D：几何IK | 用EAIK/IK-Geo生成关节分支，再走当前检查 | 六轴数值搜索 | 20次与60个终点复核完成，慢6.14%，未扩69场景 |
| E：索引复用 | 地图见证直接验证或作为原目标初值 | 重复寻找已知解 | E：地图索引复用，原目标IK初值对照已完成 |
| F：有效组合 | 组合单项测得有效的部分 | 最终请求总耗时 | 待各组结果 |

批量IK与地图分开测试，便于看清各自的收益。所有组都按原候选排名决定哪个完整方案先取得执行权。

批量求解同时保留完整场景IK和纯运动学回退两层流程，后者继续经过独立FK、限位与V2近桌检查。

## 批量接口怎样保持逐目标检查

先成批产生并检查原32个初值。每个目标已有合格解时先保留，只有未通过的行继续优化；这样其他失败目标不会冲掉该行已通过的初始解。补齐行不参与选择，返回解继续经过原碰撞、接触和路径检查。

两种批次容量的混合目标检查与独立FK均通过。

[实施说明](/media/sam3-v8-native-batch-method.md) · [接口验证数据](/media/sam3-v8-batch-rows-validation.json)

## 本地已经确认的条件

| 核查项 | 结果 | 下一步 |
| --- | --- | --- |
| cuRobo接口 | 已有批量目标与种子输入；当前辅助IK配置一次1个目标 | 预热16、32目标的独立求解器 |
| ARX运动链 | right_joint22/23/24三个连续转轴平行，腕部25/26轴相交 | 独立CPU适配与463个终点核验通过，测完整规划 |
| 关节见证 | 已有约1056万组前方记录 | 先测命中率和查询/回退合计耗时 |
| 现有地图查询 | 单目标取8192个位置近邻，再比较旋转 | 测量成批查询成本 |
| 现有朝向索引 | 用重力方向做允许底盘转向时的必要条件筛选 | 固定站位查询加入完整旋转比较 |

16/64是两只手的轮换窗口，16/32是GPU一次处理的目标数量，两者分别记录。几何求解返回的角度先转换到原限位允许的等价分支，再独立复算FK；最小二乘结果也按原2毫米、0.03弧度门槛检查。

## 地图怎样接

查询原抓姿附近的关节见证，达到原目标容差的结果进入当前场景碰撞检查；邻近记录作为原目标的求解初值。缺少合格记录时调用现有IK。

地图负责提供运动学解，场景碰撞和完整路径继续检查。先测现有地图的收益，再按具体缺口决定是否补数据。成功解缓存与当前场景碰撞结果分别管理，首轮比较使用冻结缓存。

## 已有问题定位数据

交替优化前的65个配对规划场景中，GraspGenX两臂合计检查12452次抓取终点。下表是候选检查次数：

| 检查结果 | 次数 | 比例 |
| --- | ---: | ---: |
| 当次未得到运动学解 | 10878 | 87.4% |
| 有解后碰撞检查未通过 | 1447 | 11.6% |
| 抓取终点通过 | 127 | 1.0% |

当前交替搜索的完整69份观测已完成B、C同期回放，结果见上方表格。地图主要帮助已有解的复用；批量求解也需要覆盖长串当次未解出的候选。

## 怎样实施和比较

| 顺序 | 工作 | 当前进度 |
| --- | --- | --- |
| 1 | 更新基线，核查文献、接口和ARX几何 | 已完成 |
| 2 | 冻结现行交替输入，建立分项计时 | B已完成，C增加逐次绑定记录 |
| 3 | 原生批量IK与绑定复用分别适配 | B/C完整对照与独立复算完成 |
| 4 | 几何IK适配与FK、限位、多分支检查 | 先导与60终点复核完成，当前方案未提速 |
| 5 | 地图批量查询与见证/初值对照 | E：地图索引复用，原目标IK初值对照已完成 |
| 6 | 全69份观测配对回放，筛选有效组合 | 待登记批次 |
| 7 | 按规划结果确定物理抓取验证 | 待确定 |

使用相同的69份V6保存观测、GraspGenX原候选和机器人状态。5例共同站位失败在74例总体中保留，SAM失败与原中断单列。同卡、统一预热，交替先后顺序；短时计时多次重复。

总请求耗时统计中位数和P90，并拆出解码、查询、世界绑定、开爪、IK、碰撞、路径与交接。规划通过率、原成功输入保留、选中抓法和手臂变化一并记录。每臂871球、TCP43、指尖2毫米规则及场景物理沿用当前配置。

## 接入位置

头部RGB-D → SAM3 → RaySt3R → GraspGenX原候选 → 双臂交替调度 → 本版待比较的IK/复用模块 → 当前碰撞与路径检查 → 选臂抓取。

[[流程卡片]]

## 参考方法与文件

- [cuRoboV2批量IK](/papers/curobo-v2/)
- [MoveIt Cached IK](/papers/cached-ik/)
- [EAIK / IK-Geo几何IK](/papers/eaik/)
- [VAMP-MR向量化检查](/papers/vamp-mr/)：后续FK与碰撞检查加速参考，尚未接入。
- [完整实施方案](/media/sam3-v8-indexed-planning-plan.md)
- [方法调研](/media/sam3-v8-planning-research.md)
- [实验配置](/media/sam3-v8-indexed-planning-config.json)
- [ARX静态几何核查](/media/sam3-v8-arm-geometry-review.json)
- [本地接口与来源记录](/media/sam3-v8-source-review.json)
