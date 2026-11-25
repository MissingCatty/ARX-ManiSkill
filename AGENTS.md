# ARX-ManiSkill 工程约定

- 所有实际项目代码、仿真资源、模型权重及运行产物均归入本工程。
- `modules/<module>/`保存模块代码和模块配置，模块名、代码文件与目录使用英文；禁止在模块目录保存模型权重或实际运行产物。
- `simulation/`保存仿真资产、机器人描述、相机标定与冻结场景；场景构造不能按IK或抓取结果挑选。
- 场景只保留`simulation/scenes/latest200/`这套当前200例；模型与场景分开。所有三维网格、纹理、碰撞网格及机器人模型的实际文件统一归入`simulation/assets/models/`，按资产ID或共享源库存放，不按场景复制。共享定义在`simulation/assets/catalog.json`；场景只保存资产ID、位姿、初始条件和参考观测，通过`arx_project.assets.load_object_catalog`加载。未来新增物体先登记共享资产，再在场景中引用。
- `weights/<method>/`集中存放模块权重、模型配置及相关许可证说明。不得把账号Token、API Key写进工程。
- `tmp/<method>/<run_id>/`保存每次运行的日志、缓存、观测、点云、抓取候选、规划、轨迹、录像和评估。tmp下一级必须是方法或模块英文名。
- `environments/<name>/`保存独立Python环境和工具链，环境依赖清单放在`configs/environments/`。
- `docs/`中的说明文档使用中文文件名。新增模块必须登记到`configs/modules.json`，并使用共享路径工具，不得写用户家目录或旧工程绝对路径。
- 旧工程目录及模块重名入口已退役，不得重建。运行使用本工程内的规范路径；历史记录中的旧路径通过公共读取层映射，不依赖旧目录。
- 当前SAM3主线V4在V3左右臂选择基础上增加底盘与升降站位；每臂871球、TCP43、向下45°头部D405、1毫米接触偏移及原成功判据。原V3固定45.2厘米、200场景与结果保留；V4按最新授权另建100例场景并测试。
- 网站位于根目录`website/`。所有对外发布的图片、视频、表格源数据和附件必须复制为`website/resources/`内的真实文件，不得通过软链接或硬链接依赖实验产物。页面与媒体形成独立发布快照，删除其他模块、资产或tmp后仍能运行；每次内容生成都必须经过该复制步骤。网站发布快照和待办数据库属于本规则中独立保存的例外，不放回tmp。
- 网站发布统一执行媒体压缩：保留网站原图副本，生成WebP和响应式尺寸；视频按质量检查压缩，保留分辨率、帧率和时长，启用MP4前置索引和点击加载；文本生成Brotli/Gzip。派生文件集中在website/resources/optimized，参数在website/media-policy.json，构建时自动检查缓存并补齐。
- 模块内部聚合自身算法与配置；跨模块通过共享数据契约和编排层调用。公共层只管理路径、协议、进程和记录，不导入具体模型。新增算法通过注册表接入，不修改其他算法模块。
- Hunyuan3D-2-Turbo和Hunyuan3D-2mini-Turbo已按用户确认退役，删除代码、权重和专属原始产物，不恢复安装或运行入口。SPAR3D随后已按用户确认退役，删除专用代码、权重、环境和原始产物。RaySt3R使用的共享reconstruction环境保留；网站已有五方法对照及资源副本保留。

- EASG已按用户确认退役：不恢复专用源码、云端VLM入口、source_easg资产和tmp/easg产物，也不加入资源备份。共享候选校验与坐标转换位于modules/maniskill/planning_common，指垫几何位于simulation/robots/arx/finger_pad_geometry.json；共享动作坐标数学位于arx_core的action_frame.py。

- modules/tmp、modules/asset及modules/compat_tmp已退役，不创建或提交这些兼容入口。历史路径通过arx_project.layout.canonical_path和配置映射读取，实际产物、资产和环境分别归入根目录tmp、simulation/assets/models和environments。

- 2026年11月24日用户确认模块目录精简，同时明确当前200场景的定义和参考观测保持原样、继续随Git保存。ManiSkill包内资产、两版cuRobo内置资产及ZeroGrasp旧ARX模型统一放在simulation/assets/models/source_*，通过共享资源索引加载；不得把它们重新复制回模块。可达性实验记录位于tmp/reachability/experiments，旧绝对路径只在读取时映射。OpenArm、Fangzhou/lift2等已删除的无关ZeroGrasp演示工程不恢复。

- 2026年11月24日用户授权继续将指尖近桌接触接回当前871球版本。共享规则位于configs/policies/fingertip_table.json：只对夹指最前端10毫米与桌面上表面开放最多2毫米接触，生效于最后接近、闭合；抬升仅允许退出已有接触。球参数、实际几何、其他物体检查、1毫米物理接触偏移及成功标准保持。tmp/fingertip_contact/v2_871_20261124_02三项接触与退离接口验证完成，正式物体抓取0次；不把接口验证计入抓取成功率，也不据此自动开启200场景批次。01预启动停止记录保留。

- 2026年11月25日用户要求双臂抓取作为SAM3主线V3。入口modules/pipeline/experiments/dual_arm_grasp/run.py，适配器modules/curobo/dual_arm，登记configs/experiments/sam3_graspgenx_v3.json。左右臂分别检查同一原生候选池，择一臂执行；同一世界抓姿不修改。停放臂实测碰撞网格进入规划，选中臂的末端、手指、接触、闭合和抬升全程绑定。原生左右手指凸分解不同，分别读取，不修改物理模型。V2配置快照保留在configs/experiments/history，当前配置已登记V3源码修订。先做右臂柠檬、左臂番茄、自动选臂芒果三项联调，不自动扩展200例。

- V3前三项联调在tmp/dual_arm_grasp/v3_integration_20261125_05完成：右臂柠檬、自动选臂芒果成功；番茄停在SAM3未找到目标，未执行抓取。为补齐左臂物理执行验证，另登记tmp/dual_arm_grasp/v3_left_execution_20261125_01的芒果指定左臂一项，合计三种物体四项联调。原三项结果保留，不重试失败番茄。正式200例预算仍为0。

- 2026年11月25日最新授权为SAM3主线V4：在双臂操作基础上加入升降和底盘自主站位，单独生成100个靠近桌沿的场景并实际测试。此项授权允许新增simulation/scenes/mobile100/，原latest200/作为V3冻结场景保留；新统一场景指针指向mobile100。共享三维资产仍集中存放，不按场景复制。入口modules/pipeline/experiments/mobile_dual_arm，几何模块modules/mobile_manipulation；先观测选站位，完整移动路径检查，到位停稳后重新感知抓取。原资产没有可驱动底盘的活动关节，使用限速SE2根位姿接口；joint4升降保持原PD。物理模型与成功标准不变，共100槽、每例一次，故障停止，不自动重抓。V3的0个新增200例批次授权不限制本次明确授权的V4新100例。

- 同日用户同意优化两臂候选搜索与开爪检查。独立实现为modules/curobo/dual_arm/accelerated_adapter.py，注册表入口accelerated_service.py，配置configs/experiments/dual_arm_acceleration.json。右臂有可行候选后，左臂只检查更优或同名次前缀；全部开爪采样保留，用原生FK和相同871球批量检查。124条历史选臂记录核对一致；replay_20261125_03同卡三组纯规划回放、逐点开爪和中途碰撞测试通过，实际抓取0次。当前活动注册表仍为原实现，切换工具activate_acceleration.py要求原批次完整验收并退出，不能在批中修改或删除STOP。
- V4原100例在tmp/mobile_dual_arm/v4_100_20261125_01第89例接近重规划时发生CUDA非法内存访问，全部自有组件已退出。独立验收为88例完整结束、1例中断、11例未开始、48例成功；262段已完成项视频解码通过。原始summary包含89个结果文件，网站与最终报告按领取/验收记录区分中断，不能把该中断计成普通抓取失败。原STOP保留，不自动重抓或续跑剩余场景；加速默认切换已被原批次停止门控挡住。

- 用户随后认可地图主动选站位，并再次要求保留V2指尖余量。独立实现为modules/mobile_manipulation/inverse_reachability.py及station_search.py，原世界抓姿不变，从前方10559134组关节见证反算底盘XY/转向/升降，批次16、最多原排名前256；完整池回退共用3次原生规划预算。可选规划模块curobo_dual_arm_map复用经FK、限位和碰撞确认的抓取终点关节解；手臂轨迹仍为6自由度。底盘/升降先选站位并检查完整移动路径，到位重新观测。
- 正式策略对齐的纯规划对照为tmp/mobile_manipulation/active_search_replay_20261125_06：双方显式使用interactive优化、同卡同缓存观测、统一预热，苹果22.13→8.33秒、桃子19.86→3.50秒、锅铲94.21→10.16秒，均规划通过、0物理抓取。原生轨迹优化调用约24–34毫秒；整次搜索与单次求解分开统计。锅铲指尖网格深度1.24毫米按V2前端10毫米/最多2毫米规则通过。01预检停止、05未执行登记及02–04诊断结果/源码均保留；原100例及活动规划器指针不变。
- 用户最新要求感知优化先不做，保持SAM3/RaySt3R/GraspGenX及输入采样。只继续定位地图、几何、状态绑定和文件开销。固定57块机器人凸几何复用，场景障碍仍逐例更新；手部碰桌保守包围盒粗筛与高度精算已在478条已存记录对照，掩码/部位/原因一致、深度差小于2e-12米，锅铲几何核验1.83→0.094秒。记录为tmp/mobile_manipulation/broadphase_hand_check_20261125_01；120项检查通过。全量拼接顶点反而慢的方案已撤下，源码留在tmp，不进入默认实现。完整搜索计时仍引用06冻结源，几何组件增量优化单列CPU计时。

- 同日用户明确选择“保留旧批次，直接用新版重测同一组100场景”。新登记为tmp/mobile_dual_arm/v4_map100_20261125_01，使用地图主动站位及非感知加速，逻辑规划角色curobo_dual_arm绑定可选curobo_dual_arm_map；模型、100场景、物理和成功判据与旧批次一致，预算100、每例一次、不自动重试。旧88完整/1中断/11未开始、48成功及STOP原样保留，未补跑旧槽。配置更新仅登记获授权的软件哈希，旧活动配置备份和修订回执在configs/experiments/history。GPU2经独立检查为驱动报告GPU requires reset，非显存占用；本账号不能读取内核日志，未复位或终止其他用户进程。新批次使用重新通过CUDA分配探测的GPU0/1/3，故障调查记录在tmp/resource_diagnostics/gpu2_20261125_01。
- 上述地图100例01登记在模型加载前因GPU0显存不足停止，0次执行。用户随后要求寻找其他可用显卡，现行登记为tmp/mobile_dual_arm/v4_map100_20261125_02：GPU3 RaySt3R、GPU1 SAM3/GraspGenX、GPU0 cuRobo/仿真，GPU2排除。控制器与只读发布跟进分别为arx-v4-map100-20261125-02.service及arx-v4-map100-watch-20261125-02.service；保留旧原100批次和01停止记录，当前进度只读02回执。新配置为configs/experiments/sam3_graspgenx_v4_map100_gpu_reassigned.json，模型与场景/物理/成功标准和100槽预算未变。
- 用户随后授权开始感知优化，并再次明确100例后台继续、不要终止。第一阶段只在modules/rayst3r/optimized/增加可选实现：单次重建内内容核对后的DINO特征复用、去掉损失/评估的纯推理。当前注册表、所有运行中源文件与配置均不改。23项CPU/接口检查通过，包括原生网络和两阶段几何逐元素一致。独立50观测/四实现/三次重复的登记为tmp/rayst3r/compute_optimization_20261125_01，物理与规划0次；服务arx-rayst3r-compute-20261125-01.service先等待当前100例及最终核验发布结束、全部自有组件退出，再检查显卡并执行，不能抢占、停止或重启100例。GPU耗时与真实观测输出一致性尚待该对照，不提前填写加速结果、不自动切换主线。方案在modules/pipeline/docs/pointcloud_benchmark/RaySt3R计算优化.md。

- 2026年11月25日最新指令：旧100组不再恢复。先调通“目标进入双臂共同高可达空间 → 更合适的单臂高可达空间且绑定对应手臂 → 仍失败报无法规划”，然后重新运行同一组完整100例，每例一次，不自动重抓。新配置为configs/experiments/sam3_graspgenx_v4_high100.json；实际开始前完成重新资源探测、组件预热及底盘/升降到位后的新观测与规划交接验证。原机器人、场景、物理和成功标准不变。共同/单臂高区沿用先前推荐操作空间的2厘米体素、每臂至少29组采样/26个朝向桶/4个向下桶，并保留实际原生规划检查；不能在高区失败后退回低可达区最短移动方案。
- 原地图02批次由主机关机/重启中断，封存结果为10完整、1中断、89未开始、6成功；未恢复和重抓。原始88完整/1中断/11未开始、48成功继续保留。用户随后明确删除此前无效0/100批次：v4_map100_20261125_01及其网页批次/专用媒体已核对0领取、0模型启动、0实际执行后删除，清理回执在tmp/mobile_dual_arm/cleanup_zero100_20261125_01/receipt.json。此项删除授权替代之前保留该01停止目录的要求，不用于删除非零批次。
- RaySt3R计算优化50观测×四实现×三次重复已完成，50/50点云一致性检查通过，坐标差不超过1微米，组合优化配对中位加速1.0404倍。原生输入、权重、21视角、预处理和点云后处理不变。该结果允许在下一次新登记使用优化适配器，不修改已封存旧批次。记录为tmp/rayst3r/compute_optimization_20261125_01。

- 高可达站位首次开发预检没有领取正式场景，原因是临时上四分位筛选与已登记操作空间不一致。全部原始文件和STOP已校验归档到tmp/mobile_dual_arm/high_workspace_preflight_20261125_01，空的正式登记已清理。修正后采用原2厘米地图和原推荐区域判据；相同缓存柠檬观测的CPU几何回放在0.581秒找到共同高区的无碰终点和完整移动路径，0次抓取/0次原生规划。后续先使用--development-only的独立0抓取登记完成实机仿真接口预检，再建立新的100槽批次。

- 当前独立开发预检为tmp/mobile_dual_arm/high_workspace_preflight_20261125_02，development_only=true、抓取预算0、正式领取0；控制器arx-v4-workspace-check-20261125-02.service。先等preflight/complete.json通过且services_stopped.json确认组件退出，再创建v4_high100_20261125_02正式100槽，不恢复旧槽。正式发布前缀sam3-v4-high100-20261125-02。开发预检不登记为网页100例批次；当前网页保留原88例和地图02的10例结果。

- 当前正式批次已重新登记并启动为tmp/mobile_dual_arm/v4_high100_20261125_03，控制器arx-v4-high100-20261125-03.service、只读跟进arx-v4-high100-watch-20261125-03.service。只读取03的progress.json/current_trial.json；此前高区01与02均0正式领取，已分别完整归档到high_workspace_preflight_20261125_01和03，原STOP与全部文件保留，空的100例批次和网页条目已清理。源站位联调成功证据仍在high_workspace_preflight_20261125_02，0次抓取、一次共同高区实际站位并停稳、新观测和完整规划交接通过。
- 正式02预检被其他用户新任务占用GPU0/1/2各约7.9GB引发的RaySt3R OOM停止，不是GPU驱动故障；没有终止其他用户任务。独立memory_validation_20261125_01完成50/50观测逐元素完全一致核对，出错大输入重放生成50968点，峰值分配11561.71 MiB。当前注册表启用modules/rayst3r/memory_efficient/adapter.py：保留原生FP32 softmax提升，复用注意力临时缓冲区，每次RPC完成后释放闲置缓存；权重、像素/点数/视角/预后处理保持。验证和激活副本在configs/experiments/history/rayst3r_memory_validation_20261125.json和rayst3r_memory_activation_20261125.json，新的Ray启动门槛按峰值加3GiB上取整为15360MiB。原生加载器改为统一的模块限定导入，避免同文件两份模块全局变量。以上只改计算与加载，不修改机器人、场景物理和成功判据。
- 03开始前保留了一次加载器源登记修订，原登记及原源码在registration_revisions/loader_01，变更在任何服务/正式领取之前完成。源码已冻结，运行期间不再修改已登记Python或配置。网站媒体继续复制成resources真实文件，再统一压缩；正式100例结果不能混入开发预检。

- 当前实际执行只读取tmp/mobile_dual_arm/v4_high100_20261125_04：控制器arx-v4-high100-20261125-04.service、跟进arx-v4-high100-watch-20261125-04.service。GPU2独占RaySt3R推理池；GPU0常驻SAM3、GraspGenX、cuRobo与仿真，GPU1留空。原三卡登记已在首次启动前保存为registration_revisions/two_card_01，实际双卡资源与新CUDA探测在preflight/two_card_assignment.json。每例一次，100例全新同场景，不续跑旧槽。
- 推理池验证为tmp/rayst3r/workspace_validation_20261125_03：通过真实注册表工厂连续两次处理原出错输入，50968点均与已验证输出逐元素相同，调用后保留14336MiB工作池；框架清理闲置缓存后仍保有，卡上剩余5579.69MiB。当前Ray要求启动时至少23000MiB空闲，空卡部署；先前8GiB并占测试未达资源门槛，不作为通过结果。验证副本和配置承接在configs/experiments/history/rayst3r_workspace_validation_20261125.json及rayst3r_workspace_activation_20261125.json。
- 正式04于14:03:08通过全部预检并领取首场景M0001_S1_open_apple，首例已实际抓取成功，随后进入M0002；以动态progress.json为准。原空100例01/02/03目录与网页批次已清理，全部停止证据分别保存在high_workspace_preflight_20261125_01/03/04；原88例及地图02的10例实际结果和STOP仍保留。正式04已冻结源码与配置，不得边跑边修改已登记文件。

- 用户随后明确暂时继续采用从地面到桌面的实心长方体表示桌子，优先保持链路简单，也便于真机感知桌子整体范围。桌底进入、桌板/支撑结构分解与对应感知适配暂缓；当前本体凸包、移动间隙及碰撞检查保持。后续站位仍按此长方体边界选择，不因桌底外观留空而放行底盘穿入。当前100例继续原登记，不启动碰撞模型修改或额外实验。

- 用户要求趁100例运行时定位SAM3空掩码，独立离线记录为tmp/sam3/segmentation_diagnosis_20261125_01。21张存图（13空、8正常）回放与原分数/掩码全部一致；13个原词门槛前最高分掩码参考IoU中位98.44%、最低96.66%，被官方“候选分数×类别存在性”0.5门槛过滤。固定0.5仅换同义描述恢复4/13；96组不存在类别测试中，同义词最高分策略误认5组，降到0.05后恢复13/13但误认43/96，不能据恢复数直接切换。已完成64例的50个非空初次掩码参考IoU均>80%。当前100例运行源码、提示词与门槛均未改，诊断不计物理尝试。用户随后明确将SAM3诊断详细数据放到SAM3主线V5，已从V4正文迁至V5；V5为离线诊断完成、改进待验证，当前100例及原批次仍属V4。

- 用户最新要求统计分母剔除SAM失败。后续进度和网站的主要规划通过率、抓取成功率，统一用“完整结束例数－SAM3最终失败例数”作分母，SAM失败单列。包含初次及站位后重新观测的终止性SAM失败；若unable_to_plan实际是所有已到位观测均SAM失败、且从未调用原生抓取规划，也归SAM失败。途中一次分割失败后恢复并成功的场景仍计入。中断、未开始继续单列；原始逐例回执与成功判定不改。展示统计在website/tools/grasp_metrics.py与各批次publication_statistics.json，网站复制独立统计副本。

- 正式04在第88例cuRobo更新碰撞场景时发生CUDA非法内存访问并停止：87例完整、1例中断、12例未开始、48成功。独立验收通过、390段视频解码正常，旧STOP与全部数据保留。按最新分母剔除20例SAM最终失败（19初次、1站位后），为48/67抓取成功、63/67规划通过。
- 用户随后明确要求“恢复之后继续”。新续跑登记为tmp/mobile_dual_arm/v4_high100_resume_20261125_01，继承原87例完整结果与第88例中断，只允许执行原89—100这12个未开始槽，保留全局序号与随机种子，不重抓旧槽。调度承接模块为modules/pipeline/experiments/mobile_dual_arm/continuation.py；旧执行源与活动配置在旧登记和configs/experiments/history/v4_continuation_20261125_01_previous.json保留，只修改调度和中断统计，模型/物理/长方体桌子/SAM3原提示与0.5门槛不变。资源沿用已验证23000MiB Ray启动门槛，重新CUDA探测后选择GPU2 Ray、GPU1 SAM3/GraspGenX、GPU0 cuRobo/仿真；额外低余量诊断在资源释放后取消，未用于放宽门槛。

- 续跑01在模型/仿真创建前被瞬时显存门控停止，0新增槽、0新增服务，旧STOP和登记保留。当前承接为tmp/mobile_dual_arm/v4_high100_resume_20261125_02，仍只执行原89—100；6项离线承接/故障检查通过。原87条结果、原88中断保留，索引和种子沿用原100场景。启动器arx-v4-high100-resume-launch-20261125-02.service先等待已登记显存门槛（Ray23000MiB，不放宽），再启动arx-v4-high100-resume-20261125-02.service及同名加-watch的只读跟进。正式源码在新登记冻结；模型启动前显存不足只等待，模型或试验开始后的资源/记录故障仍STOP，不重抓。原始大媒体与数组在工程内部复用不可变文件，结果JSON及独立评价单独复制；网站媒体仍全部为独立真实副本。

- 用户要求监测空闲卡并自动启动，独立脚本为modules/pipeline/monitoring/gpu_auto_start.py，配置configs/monitoring/v4_continuation_gpu.json，后台arx-v4-gpu-auto-start-20261125-02.service。每秒扫描显存，达到原门槛后提交已登记启动器；控制器已活动时只更新状态，不重复启动，STOP或已领取新槽后退出时不重启。6项离线启动/故障测试通过，0模型调用。当前续跑02控制器已启动并进入模型预热，监测状态在tmp/resource_monitor/v4_continuation_20261125_02/status.json；不原位修改运行中已登记执行源码。

- 用户随后明确授权 SAM3 分割后改用 ZeroGrasp，与当前 RaySt3R＋GraspGenX 链路比较规划成功率、抓取成功率，并补充延时。独立实验为 SAM3 主线 V6，父版本 V4，与 V5 分割诊断并列；这是本轮实验调用 ZeroGrasp 的明确例外，不修改默认主链。入口 modules/pipeline/experiments/sam3_zerograsp_comparison，配置 configs/experiments/sam3_zerograsp_comparison.json，排队批次 tmp/sam3_zerograsp_comparison/v6_paired_20261125_01。先等当前 V4 续跑完整结束、服务退出并通过独立验收，再冻结全部完整且非 SAM 最终失败的场景；保持同一最终抓取准备站位、SAM掩码、RGB-D及物体/机器人状态。原站位共同失败保留，中断和未开始单列。使用原生碰撞/宽深修正/NMS后的ZeroGrasp候选，无原始候选回退或附加姿态筛选；TCP43、871球、指尖2毫米余量、物理与成功标准均不变，每例一次。18项离线契约/后处理故障检查通过，原生API导入通过；正式进展读取新批次。排队服务 arx-sam3-zerograsp-v6-queue-20261125-01.service，不修改运行中的V4冻结文件。
- 用户要求将GraspGenX官方扫掠区域提取复核先列为夹爪配置待办，当前不修改扫掠参数。网站 gripper-configuration 待办第7项已登记：核对官方全开/半开区域、当前15毫米指腹截取、坐标偏移与开合动画。配置页待办同步进入首页汇总，状态独立存于website/resources/state。

- V4续跑02已结束并通过独立验收：99完整、1原中断、0未开始，54成功，430段录像解码通过；25个SAM最终失败剔除后为69/74规划通过、54/74成功。V6已据此冻结74个配对场景，其中5个站位共同失败、69个可执行生成与抓取测试。控制器arx-sam3-zerograsp-v6-paired-20261125-01.service及同名前缀-watch.service均已启动，预检通过；牛油果左臂首次实际抓取成功，最低保持抬升3.026厘米、3.025秒，两视角录像解码通过。正式进度以tmp/sam3_zerograsp_comparison/v6_paired_20261125_01/progress.json为准，勿新建第二个控制器或修改冻结源。ZeroGrasp附带初始化用的ResNeXt权重真实文件集中到weights/zerograsp/backbone，TorchHub运行缓存为等内容路径别名，承接证明在本轮preflight/backbone_resource.json。

- V6配对测试现已完成74/74，0中断、0未开始，独立验收通过，控制器与发布进程、自有模型服务均已退出。原RaySt3R＋GraspGenX组规划69/74、成功54/74；ZeroGrasp组规划60/74、成功44/74。25个SAM最终失败仍单列剔除，5个共同站位失败保留在74例分母中。本次结束，不重试或追加场景，原结果与冻结源档案保留。
- 用户要求去掉候选池的重复读取、旧路径转换开销，并以后台线程保存候选。已在原V6完成验收并退出后启用候选I/O修复，记录为tmp/candidate_io/activation_20261125_01，旧源码与配置完整保留。候选通过arx_project.artifacts共享内存交接，两臂在modules/curobo/candidate_io/store.py中共享一次解码；每臂只保存小型引用记录，真实旧路径仍兼容。当前高可达站位的GraspGenX转换显式background=True；旧文件工具默认同步接口保留，ZeroGrasp也接入相同后台留档契约。配置注册curobo_dual_arm及curobo_dual_arm_map的candidate_transport_format，文件消费者由客户端等待快照后调用。保存失败向上传播，请求完成前确认记录写入，不自动重试。69条历史候选/选臂检查和4个同卡原生cuRobo回放通过，抓姿与关节解完全一致；此次新增物理抓取0次。详见modules/curobo/docs/候选数据交接与后台留档.md；机器人、物理、候选顺序与碰撞/成功标准不变。

- 用户要求预先建立GraspGenX接类似ZeroGrasp碰撞筛选、宽度深度修正与NMS的独立实验版本，登记为SAM3主线V7，父版本V4，与V5/V6并列。方案为modules/pipeline/docs/mainline_versions/V7_GraspGenX候选后处理实验方案.md，配置configs/experiments/graspgenx_postprocessing.json，准备记录tmp/graspgenx_postprocessing/preparation_20261125_01。当前仅完成源码核对、69份已有候选数量核查与版本登记，适配器尚未实现，正式批次0、物理测试0。先离线比较原候选/仅NMS/ARX几何碰撞和空抓筛选＋NMS，宽度深度修正另组；NMS初始参考3厘米/30°仅作去重，不修改保留抓姿。几何需对齐当前ARX、生成器后移100毫米与TCP43，原生固定夹爪参数不能直接当ARX参数。所有组使用相同候选I/O修复，记录候选减少与成功抓法误删，后处理加规划总耗时配对比较；主链默认不切换，不自动启动物理批次。网站V7及附件均为resources独立副本。

- 用户明确授权将网站待办随Git推送并复制到公网部署。实时SQLite数据库继续留在website/resources/state/，可恢复快照website/resources/state/todos.snapshot.json是Git例外；使用website/tools/todo_snapshot.mjs导出或恢复。新部署首次创建数据库时自动载入快照，已有数据库不因重启而覆盖；显式迁移前保存原数据库备份，保留条目编号、完成状态和删除后的编号序列。部署公网网站需同时同步此快照并核对API结果，不能只复制静态页面。
