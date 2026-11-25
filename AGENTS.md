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
