# ARX-ManiSkill 工程约定

- E实际IK登记index_seed_probe_20261126_01已在加载前因可用CUDA显存低于原10000MiB门槛停止，started不存在、IK实际0次、3132次未执行，STOP及输入源码保留。无GPU规划或物理动作，不自动重启此登记；状态/承接见stop_summary.json。E的10,322次CPU查询与16项检查已完成，网站明确VAMP尚未实施。当前没有后台V8实验。

- 最新E阶段：index_lookup_20261126_01的10,322个实际历史候选CPU查询全部完成，128个原地图见证与32组旧查询等值核对通过，16项离线检查通过。直接原精度命中0；2厘米近邻在1.72°/3°/5°/10°下命中13/32/142/698。用户指出0.03弧度太紧，已明确区分“近邻初值查询范围3/5/10度”与“最终原目标验收2毫米/0.03弧度”，不修改抓姿。新脚本seed_probe.py准备同一预选261终点×4组×3次的3,132次纯运动学IK对照，物理/仿真/完整规划均0；实际是否启动读active_run、controller和progress，不重复启动。VAMP仅为后续FK/碰撞向量化参考，当前未接入B/C/D。旧登记与STOP、默认V3交替实现继续保留。

- 最新：D先导20次已完成，双方各10/10规划通过，抓法/选臂及候选检查数相同，60个终点独立FK/限位通过；总耗时204.83→217.42秒，慢6.14%，未达到2%收益门槛，未登记69例。D的1434次候选检查里1200个无合格几何分支、198个有分支但终点未过、36个几何终点过，其中22个完整路径失败后回退，最终14个候选走完几何路径。所有D服务已正常退出，无后台实验，不自动重跑。下一阶段E索引复用待实施，B批量16仍优先；物理0，默认链路不变，所有原登记/停止记录保留。

- 最新实际阶段：C完整138次及414个终点独立FK/限位复算已通过，原抓法、选臂和检查数全部一致；累计省时2.33%，中位14.18→15.86秒，暂保留可选而不纳入默认组合。C跟进已完成并发布。D已由geometric_followup_20261126_01自动登记geometric_pilot_20261126_01，20次ABBA/BAAB纯规划先导在GPU3启动，常驻预热和原生正负目标核验通过，开始产生实际规划结果；物理0。B批量16仍优先保留。D运行中全部注册源码/配置冻结，实际进度读取execution.active_run，勿启动重复控制器；20次全部独立验收且省时2%后才扩138次，原停止记录继续保留。

- 2026年11月26日继续V8：C完整对照仍按原158预算后台运行；D几何IK已在独立环境geometric_ik完成EAIK 1.2.2适配。27项新检查、256个完整机器人随机状态及原69份规划的207个终点核验通过，全部207终点有合格几何分支。新可选代码位于modules/curobo/indexed_planning/geometric，原IK、模型、场景、871球、TCP43和V2规则保持。无合格几何分支继续原IK；几何关节解后续路径不通过，先完整尝试同一原候选，再推进评分排名。队列geometric_followup_20261126_01等C独立验收/发布/退出，再登记D先导20次，符合原成功保留及省时2%门槛后扩138次。D物理/感知/仿真预算均0，不自动切换默认。CPU核验最新03，旧01与02及停止标记保留；实际阶段读execution.active_run和对应跟进状态，不修改正在登记的源。

- 2026年11月26日最新进展：B原生批量IK完整69份观测×三组207次规划已完成，各组69/69通过，原抓姿及选臂全部一致；621个终点独立FK与限位复算通过，批量16合计省时32.84%，后续组合优先16。结果与公网发布见native_batch_69_20261126_01及native_batch_followup_20261126_02。用户要求结束后继续，当前推进C上下文复用：可选binding_service.py为每臂保留开放/闭合上下文、共享纯运动学求解器，只在完整请求内按完全相同状态复用，复用仍重查网格、回执、FK与起始碰撞。20项新故障测试加原31项共51项通过。当前先导binding_reuse_pilot_20261126_01登记五场景ABBA/BAAB共20次规划，跟进binding_followup_20261126_01；原成功和抓法保留、独立复算通过、重绑定减少且总耗时降低至少2%后才登记69×两组138次。C使用逐个IK单独比较，物理、感知、仿真均0，不自动切换默认。全部旧登记与STOP保留，不修改运行中注册源码，实时状态读取execution.active_run。

- 2026年11月26日用户“开始吧”授权启动V8实施与纯规划比较。首阶段原生批量IK已建立可选模块curobo_indexed_planning，完整场景IK与纯运动学回退一起批量化，默认链路仍使用V3交替搜索。20项新离线检查及11项原交替回归（含72组排序子例）通过。当前预检读取configs/experiments/indexed_grasp_planning.json的execution.active_run；native_batch_pilot_20261126_01在模型配置构造阶段停止，正式规划0次，STOP及源码保留；02、03的配置/批次接口停止也保留，03有1次完整控制请求与1次接口中断，不纳入配对结果；04的混合批次检查停止也保留。当前05完成五份观测×control/batch16/batch32的15次纯规划预检，三组各5/5通过，原抓姿与手臂全部一致；45个终点独立FK及限位通过。批量按行保留已通过LM初值，仅优化未通过行，原32初值预算与标准保持；直接SDK及修正接口对照在ik_rows_20261126_03/04。后台followup服务在独立FK及控制成功保留通过后自动登记69份观测×三组的207次回放，阶段状态读取native_batch_followup_20261126_02/status.json。完整回放native_batch_69_20261126_01已登记并启动，69份冻结观测共207次规划，无感知、仿真及物理动作，不重抓原V6。原默认等价的native_grasp_ik钩子与可选模块登记已保存前值并更新配置散列，承接记录为history/indexed_planning_optional_registration_20261126.json；物理与模型参数均未变。批量方法比较总耗时和原成功输入保留，不以单次IK计时替代总收益，不自动提升为默认。

- 用户于2026年11月26日要求交替优化记录到V3，并调研V8的优化方法。V3已单列双臂交替搜索方法与19次纯规划验证，原4项物理联调保留；V6仍保留原74例成绩与计时，并链接V3模块记录。V8方案以现行16/64交替调度为基线，比较原生批量IK、绑定复用、EAIK/IK-Geo几何IK、地图见证与有效组合。当前URDF的22/23/24轴平行、25/26轴线相交，静态核查保存在tmp/indexed_grasp_planning/research_20261126_01；本轮V8原生IK/规划及物理抓取均0，适配器尚未实施。配置与中文方案已更新，原方案副本保留。后续先实现并验证批量完整场景IK与纯运动学回退，再比较其他支路，不改变原抓姿、评分与碰撞规则。

- 2026年11月26日用户要求“先优化交替”已完成并启用：当前curobo_dual_arm_map使用interleaved_map_service.py，首轮各16个、随后每批64个交替检查，各臂保存游标；只剩一臂时直接检查剩余竞争前缀。原抓姿、评分、同名次关节位移比较和全部碰撞/路径门控保持。28项离线检查及72组排序子例通过。tmp/dual_arm_interleaving/replay_20261126_01完成5对/10次同卡规划，香蕉428.23→8.55秒；02新增5次16/64规划，五组输入的规划成败、选中臂及原抓姿仍全部一致。window_sizes_20261126_01追加4次同模型池AB/BA对照，盒装食品固定16批次19.87秒、后续64批次17.89秒，窗口8→4、检查114次保持。合计19次纯规划，0仿真实例、0物理抓取、0感知模型推理。默认切换记录在02/activation和configs/experiments/history/dual_arm_interleaving_activation_20261126.json；current configuration核验通过，新抓取预算0。旧V6完整成绩、录像、STOP及各轮原始记录保留，V8索引/批量IK仍未实施。

- V6完整74例对照已于2026年11月26日完成，最终记录tmp/sam3_zerograsp_comparison/v6_complete74_20261126_01：两组各74条完整结果，新增两例ZeroGrasp补跑均成功；RaySt3R＋GraspGenX规划69/74、抓取55/74，ZeroGrasp规划59/74、抓取45/74。148条完整结果独立验收通过，256段录像全部解码正常，模型服务均已退出。历史两条中断保存在historical_attempts，旧批次和STOP保留，不再追加或重抓。最终报告modules/pipeline/docs/mainline_versions/V6_共享内存对照最终结果.md；公网发布与Git推送状态读取本批final_publication.json、git_push.json及队列status.json。V8仍为已登记待实现方案，未因V6结束自动启动。

- 2026年11月26日最新授权：等GraspGenX原74例结束，再补跑ZeroGrasp两条历史中断M0042_S3_sparse_avocado和M0060_S3_sparse_donut，使两组各完整74例，最终更新公网并推送Git。此授权替代两例旧禁止重试规则，其他场景不重跑；普通抓取失败算完整结果。新增独立编排modules/pipeline/experiments/sam3_zerograsp_completion，配置configs/experiments/sam3_zerograsp_complete74.json；队列arx-v6-complete74-20261126-01.service等待前驱控制器、发布与模型退出且验收通过，之后登记tmp/sam3_zerograsp_comparison/v6_complete74_20261126_01。146条完整结果继承，旧两条中断复制到historical_attempts，所有旧结果与STOP保留。模型、物理、候选、成功标准与显存门槛保持；只补两例各一次。最终148条完整结果独立验收后发布V6最终报告及补跑录像，保留V8和公网待办，然后提交并验证远端Git。队列状态tmp/sam3_zerograsp_completion/complete74_20261126_01/status.json；不得重复启动或误把等待视为故障。

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
- 公网部署主机已由用户明确提供为root@117.72.104.29；非敏感配置记录在configs/deployments/public_website.json，凭据不进入工程。网站路径/opt/arx-maniskill/website，服务arx-website.service以arx-web运行，Node在/opt/arx-maniskill/environments/node/bin/node；后端127.0.0.1:18786，经现有Nginx代理公开访问。保持CLARO_HOST及本机HTTPS代理的待办同源支持，每次部署先在独立暂存目录验证，旧网站与数据库保存到deployment-backups，再切换并核对公网待办。流程见docs/公网网站部署与待办同步.md。

- 用户最新授权两组同时复测原74例，主要比较I/O优化后的延时并增加失败分布。新批次tmp/sam3_zerograsp_comparison/v6_io74_parallel_20261125_01，入口modules/pipeline/experiments/sam3_zerograsp_comparison/io_parallel，配置sam3_zerograsp_io74_parallel.json；控制器arx-v6-io74-parallel-20261125-01.service及同名-watch.service。两法均重新推理与抓取，固定同一SAM3、RGB-D、最终站位和物体状态；GraspGenX不加后处理，ZeroGrasp保留原生后处理，强制共享内存、后台留档、一次解码。两组各74条结果，5个共同站位失败保留，最多各69个物理槽；每场景每方法一次，原结果/STOP保留。Ray启动23000MiB与原单组件门槛不放宽，模型前资源不足由后台等待，两组预热就绪后共同放行；资源/记录/配置故障停止后续调度。6项新离线契约检查及12项前驱软件承接核对通过。GPU3显示空闲但独立CUDA分配未通过，记录在preflight/ray_device_probe.json；调度器会跳过近期分配失败的设备并尝试健康组合，不排查或复位显卡。首次源登记与0模型/0领取下的资源门控修订保存在registration_revisions/admission_01。网页V6当前内容替换为本轮进度与结果，旧页及媒体归档，V7仍为待实现计划；发布跟进使用已授权SSH控制连接同步公网内容，保留公网实时待办数据库。

- 用户随后明确“一张卡好了就先测，不必须两个同时测”，撤销上述强制同时放行和同一规划GPU要求。同一74例登记在0模型/0领取下修订为独立先就绪先测，原登记与源码保存在registration_revisions/independent_02；不删STOP、不新增场景或重试槽。ZeroGrasp允许在至少18000MiB空闲的一张卡上运行生成/规划/仿真，逐组件仍检查原14000/10000MiB；GraspGenX采用Ray独立23000MiB卡及至少16000MiB执行卡，其他原门槛不变。各组独立预热检查并放行，另一组缺资源不阻挡已就绪组，网页记录分组进度和实际设备；6项修订后的离线契约检查通过。

- 上述I/O复测在ZeroGrasp完成32例、领取第33例后发生Warp碰撞网格分配显存不足并停止：21成功、25规划通过，32完整/1中断/41未开始；GraspGenX尚未启动。原STOP、登记与全部结果保留，独立记录核验通过。用户明确要求继续，新承接目录为tmp/sam3_zerograsp_comparison/v6_io74_continue_20261125_01，只执行ZeroGrasp剩余41例与GraspGenX74例，原第33例中断不重试；统计承接旧32例并单列中断。规划服务通过io_parallel/planner_service.py可选包装释放不再活跃的Warp网格缓存，不改原规划器、当前障碍、碰撞球、抓姿、物理或成功标准；各新槽领取前检查1536MiB余量。13项离线故障检查通过，100次真实CUDA图网格查询和缓存生命周期检查通过，实际GPU缓存用量保持1876字节；验证0抓取。后续只读取新登记进度，不重启旧控制器。

- 承接01于22:26停止：ZeroGrasp累计44完整/2中断/28未开始、36规划通过、29成功；GraspGenX5完整/0中断/69未开始、4规划通过且成功。GraspGenX在下一槽领取前触发1536MiB显存余量门控，总调度按原故障规则中断正在执行的ZeroGrasp甜甜圈一例；本轮没有新的CUDA分配/OOM报错。所有模型与调度均已退出，STOP保留。离线核验原有两处程序问题已修复：继承候选按已登记前驱定位原始绝对路径凭据并核对副本哈希；GraspGenX原生转换在独立解释器中重建，隔离两个同名common模块。旧失败验收和前后审查源码保存在acceptance_revisions/audit_01，2项离线故障检查通过。重新核验49完整/2中断/97未开始、80段视频全部通过，record_integrity=true；执行登记和STOP散列未变，未新增抓取或重启实验。

- 用户再次明确继续，当前新承接为tmp/sam3_zerograsp_comparison/v6_io74_continue_20261125_02，继承44+5条完整结果及两条中断，只新增ZeroGrasp28槽、GraspGenX69槽，每个未领取场景仍一次。旧STOP/验收不改。分组显卡从提交时即预留，加载与执行期间不分配给另一自有分组；完成退出后释放预留。进度新增in_progress，正在执行的槽不再临时计为中断。17项离线检查通过；模型、机器人、物理、候选、成功标准、显存门槛及故障停止规则不变，正式源码在新登记冻结。

- 上述承接02最终为ZeroGrasp72完整/2历史中断/0待测、43成功；GraspGenX30完整/0中断/44待测，再次在领取下一例前因显存余量门控停止，结果与STOP保留，独立验收通过。沿用用户持续完成同一74例的授权，承接03仅补GraspGenX未领取的44槽，ZeroGrasp只继承结果、不加载模型；18项离线检查通过。入口arx-v6-io74-continue-20261125-03.service及同名-watch.service，原冻结RaySt3R后端不变；启动等待独立加速接口核验释放GPU。当前进度仍读取tmp/sam3_zerograsp_comparison/active_io_repeat.json，未重试任何旧领取或中断。

- 用户要求加速点云重建，独立可选实现为modules/rayst3r/fast_inference，基于PyTorch SDPA融合注意力、同次DINO特征复用及省去推理损失/指标。原21视角、输入分辨率、FP32 DINO、BF16主网络、权重和筛选保留；融合输出存在浮点差异，未宣称逐点相同。tmp/rayst3r/inference_acceleration_20261125_01已完成50观测×4实现×3次计时，原重建中位1.2466秒→0.6606秒，配对加速1.89倍；完整表面F-score@5mm均值91.414%→91.395%，CD中位1.806→1.789毫米，按预登记几何门槛50组核验通过。独立三组实际头部观测工作接口1.2855→0.6883秒，相机/世界坐标、21视角及模型只加载一次契约通过；7项CPU检查通过，新增物理抓取和规划0。已验证配置和可选模块定义在configs/experiments/rayst3r_fast_inference_profile.json及rayst3r_fast_inference_module.json，原configs/modules.json没有切换，原74例仍使用冻结后端。报告modules/rayst3r/docs/重建推理加速.md，峰值计算显存约11.3GiB，服务14GiB工作区与原启动门槛保留。

- 2026年11月26日用户明确将可达性地图的数据库索引、关节解复用与批量IK检查方案作为实验版本。登记为SAM3主线V8，父版本V4，与V7后处理并列；配置configs/experiments/indexed_grasp_planning.json，方案modules/pipeline/docs/mainline_versions/V8_可达性索引与批量IK加速实验方案.md，准备记录tmp/indexed_grasp_planning/preparation_20261126_01。当前仅问题定位和方案登记，适配器待实现，V8原生规划与物理抓取预算均为0，不改当前V6运行登记与默认链路。原抓姿、评分、顺序与来源编号保持；查询命中需独立FK及限位核验，邻近解只辅助求解原目标，未覆盖保留原IK，当前场景碰撞与完整路径另验。先比较原实现、索引复用、索引复用加分批终点检查，首轮地图与缓存冻结，同卡统一预热，记录查询及各规划阶段耗时、调用数和规划通过率。网页V8与附件真实复制到resources，旧V6/V7结果和节点保留。
