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
- 当前主线配置为871碰撞球、TCP43、固定45.2厘米、向下45°头部D405、1毫米接触偏移及独立采样200场景。保持机器人实际几何、限位和成功判据。配置整理不授权自动运行新抓取批次。
- 网站位于根目录`website/`。所有对外发布的图片、视频、表格源数据和附件必须复制为`website/resources/`内的真实文件，不得通过软链接或硬链接依赖实验产物。页面与媒体形成独立发布快照，删除其他模块、资产或tmp后仍能运行；每次内容生成都必须经过该复制步骤。网站发布快照和待办数据库属于本规则中独立保存的例外，不放回tmp。
- 网站发布统一执行媒体压缩：保留网站原图副本，生成WebP和响应式尺寸；视频按质量检查压缩，保留分辨率、帧率和时长，启用MP4前置索引和点击加载；文本生成Brotli/Gzip。派生文件集中在website/resources/optimized，参数在website/media-policy.json，构建时自动检查缓存并补齐。
- 模块内部聚合自身算法与配置；跨模块通过共享数据契约和编排层调用。公共层只管理路径、协议、进程和记录，不导入具体模型。新增算法通过注册表接入，不修改其他算法模块。
- Hunyuan3D-2-Turbo和Hunyuan3D-2mini-Turbo已按用户确认退役，删除代码、权重和专属原始产物，不恢复安装或运行入口。SPAR3D随后已按用户确认退役，删除专用代码、权重、环境和原始产物。RaySt3R使用的共享reconstruction环境保留；网站已有五方法对照及资源副本保留。

- EASG已按用户确认退役：不恢复专用源码、云端VLM入口、source_easg资产和tmp/easg产物，也不加入资源备份。共享候选校验与坐标转换位于modules/maniskill/planning_common，指垫几何位于simulation/robots/arx/finger_pad_geometry.json；共享动作坐标数学位于arx_core的action_frame.py。

- modules/tmp、modules/asset及modules/compat_tmp已退役，不创建或提交这些兼容入口。历史路径通过arx_project.layout.canonical_path和配置映射读取，实际产物、资产和环境分别归入根目录tmp、simulation/assets/models和environments。

- 2026年11月24日用户确认模块目录精简，同时明确当前200场景的定义和参考观测保持原样、继续随Git保存。ManiSkill包内资产、两版cuRobo内置资产及ZeroGrasp旧ARX模型统一放在simulation/assets/models/source_*，通过共享资源索引加载；不得把它们重新复制回模块。可达性实验记录位于tmp/reachability/experiments，旧绝对路径只在读取时映射。OpenArm、Fangzhou/lift2等已删除的无关ZeroGrasp演示工程不恢复。

- 2026年11月24日用户授权继续将指尖近桌接触接回当前871球版本。共享规则位于configs/policies/fingertip_table.json：只对夹指最前端10毫米与桌面上表面开放最多2毫米接触，生效于最后接近、闭合；抬升仅允许退出已有接触。球参数、实际几何、其他物体检查、1毫米物理接触偏移及成功标准保持。tmp/fingertip_contact/v2_871_20261124_02三项接触与退离接口验证完成，正式物体抓取0次；不把接口验证计入抓取成功率，也不据此自动开启200场景批次。01预启动停止记录保留。
