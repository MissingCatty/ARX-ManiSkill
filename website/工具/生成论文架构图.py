"""Draw original, editable SVG method diagrams for the four paper records."""
from html import escape
from pathlib import Path
from fontTools import subset
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

DEST = Path(__file__).resolve().parents[1] / '内容/资源/论文架构图'
FONT = TTFont(Path(__file__).resolve().parents[2] / '缓存/对比视频字体/NotoSansCJKsc-Regular.otf')
GLYPHS, CMAP = FONT.getGlyphSet(), FONT.getBestCmap()
UNITS = FONT['head'].unitsPerEm
COLORS = {
    'input': ('#f3f5f9', '#d8dfe9', '#526079'),
    'model': ('#eef1ff', '#c9d2fb', '#4057af'),
    'geometry': ('#eaf7f5', '#b6dfd6', '#236e62'),
    'memory': ('#fff5e5', '#eed1a0', '#93611d'),
    'output': ('#f1edfb', '#d7cbee', '#735196'),
}


class Diagram:
    def __init__(self, name, title, subtitle, height=760):
        self.name, self.height = name, height
        self.edges, self.nodes = [], []
        self.glyphs = {}
        self.title, self.subtitle = title, subtitle

    def text(self, x, y, value, size=17, color='#667085', anchor='middle', weight=400, max_width=None):
        names = [CMAP[ord(ch)] for ch in value]
        advances = [GLYPHS[name].width for name in names]
        width = sum(advances)*size/UNITS
        if max_width and width>max_width:size*=max_width/width; width=max_width
        left = x-width/2 if anchor=='middle' else x
        uses, offset = [], 0
        for ch,name,advance in zip(value,names,advances):
            key=f'g{ord(ch):x}'
            if key not in self.glyphs:
                pen=SVGPathPen(GLYPHS);GLYPHS[name].draw(pen)
                self.glyphs[key]=pen.getCommands()
            uses.append(f'<use href="#{key}" x="{offset}"/>');offset+=advance
        stroke=f' stroke="{color}" stroke-width="15"' if weight>=650 else ''
        return (f'<g aria-label="{escape(value,quote=True)}" fill="{color}"{stroke} '
                f'transform="translate({left} {y}) scale({size/UNITS} {-size/UNITS})">'+''.join(uses)+'</g>')

    def box(self, x, y, w, title, detail, kind='model', h=88):
        fill, stroke, ink = COLORS[kind]
        body = [f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
                self.text(x+w/2, y+34, title, 21, ink, weight=650, max_width=w-24)]
        for i, line in enumerate(detail.split('\n')):
            body.append(self.text(x+w/2, y+61+i*23, line, 17, '#536076', max_width=w-24))
        body.append('</g>')
        self.nodes.append(''.join(body))

    def arrow(self, points, label=None, at=None, feedback=False):
        color = '#a77b37' if feedback else '#8591ac'
        dash = ' stroke-dasharray="7 5"' if feedback else ''
        marker = 'feedback' if feedback else 'arrow'
        coords = ' '.join(f'{x},{y}' for x,y in points)
        self.edges.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                          f'stroke-width="2" stroke-linejoin="round"{dash} marker-end="url(#{marker})"/>')
        if label:
            x,y=at
            # Keep labels legible where data paths cross the white canvas.
            width = max(58, len(label)*17+16)
            self.edges.append(f'<rect x="{x-width/2}" y="{y-19}" width="{width}" height="25" rx="5" fill="white"/>')
            self.edges.append(self.text(x,y,label,16,color))

    def save(self, note):
        title=escape(self.title)
        heading=self.text(40,46,self.title,28,'#26344d','start',700)
        subheading=self.text(40,78,self.subtitle,17,'#798398','start')
        footer=self.text(40,self.height-25,note,16,'#798398','start')
        glyph_paths=''.join(f'<path id="{key}" d="{path}"/>' for key,path in self.glyphs.items())
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{self.height}" '
             f'viewBox="0 0 1100 {self.height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{title}：论文主要方法架构图</title>',
             f'<desc id="desc">{escape(self.subtitle)}。{escape(note)}</desc>',
             '<defs>'+glyph_paths+'<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
             '<path d="M0 0 L9 4.5 L0 9" fill="#8591ac"/></marker>'
             '<marker id="feedback" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
             '<path d="M0 0 L9 4.5 L0 9" fill="#a77b37"/></marker></defs>',
             '<g font-family="Noto Sans CJK SC, Microsoft YaHei, sans-serif">',
             f'<rect width="1100" height="{self.height}" rx="16" fill="white"/>',
             heading,subheading,
             *self.edges,*self.nodes,
             footer,'</g></svg>']
        DEST.mkdir(parents=True,exist_ok=True)
        (DEST/f'{self.name}-architecture.svg').write_text('\n'.join(svg)+'\n')


def zerograsp():
    d=Diagram('zerograsp','ZeroGrasp','联合重建形状与抓取，再用重建几何细化位姿',800)
    d.box(40,112,270,'单视角观测','RGB-D · 实例掩码 · 相机内参','input')
    d.box(410,112,270,'实例八叉树','回投点云 + RGB 图像特征')
    d.box(790,112,270,'八叉树 CVAE','条件先验 · 潜变量 z')
    d.box(40,270,270,'三维遮挡场','射线检测：自身 / 物间遮挡','geometry')
    d.box(790,270,270,'多物体编码器','3D Transformer · 邻近关系')
    d.box(410,405,270,'联合解码器','融合潜变量、关系与遮挡')
    d.box(190,540,310,'形状预测','SDF · 表面法向','geometry')
    d.box(600,540,310,'抓取预测','方向 · 旋转 · 宽度 · 深度 · 分数','output')
    d.box(340,667,420,'原生抓取细化 → 候选','接触点校正宽度 / 深度 + 几何碰撞检查','output')
    d.arrow([(310,156),(410,156)])
    d.arrow([(680,156),(790,156)])
    d.arrow([(175,200),(175,270)])
    d.arrow([(925,200),(925,270)],'潜特征 ℓ',(975,242))
    d.arrow([(925,200),(1090,200),(1090,380),(650,380),(650,405)],'潜变量 z',(1000,380))
    d.arrow([(175,358),(175,449),(410,449)])
    d.arrow([(925,358),(925,449),(680,449)])
    d.arrow([(480,493),(480,516),(345,516),(345,540)])
    d.arrow([(610,493),(610,516),(755,516),(755,540)])
    d.arrow([(345,628),(345,647),(465,647),(465,667)])
    d.arrow([(755,628),(755,647),(635,647),(635,667)])
    d.save('依据论文 §3 绘制；这里的碰撞检查针对预测夹爪与重建几何。')


def graspcl():
    d=Diagram('graspcl','GraspCL','示教补充抓法，执行经验改变候选评分；部署时固定编码器',780)
    d.box(40,112,270,'当前场景点云','单视角观测','input')
    d.box(790,112,270,'用户示教','抓取位姿 + 局部点云','input')
    d.box(40,265,270,'几何候选采样','接触法向采样 / 俯抓')
    d.box(410,265,270,'示教召回与迁移','FPFH 匹配 + 点到平面 ICP')
    d.box(790,265,270,'召回记忆','存储示教抓法与局部形状','memory')
    d.box(40,430,270,'候选合并与筛选','运动学可行性 + 碰撞')
    d.box(410,430,270,'局部几何编码','抓取坐标系 · BPS → 32 维')
    d.box(790,430,270,'记忆评分与排序','近邻证据 → Beta 后验均值','output')
    d.box(410,620,270,'评分记忆','离线数据 + 新结果 + 示教','memory')
    d.box(790,620,270,'执行并获取结果','成功 / 失败','output')
    d.arrow([(175,200),(175,265)])
    d.arrow([(175,200),(175,230),(545,230),(545,265)])
    d.arrow([(925,200),(925,265)])
    d.arrow([(790,309),(680,309)])
    d.arrow([(175,353),(175,430)])
    d.arrow([(545,353),(545,388),(175,388),(175,430)])
    d.arrow([(310,474),(410,474)])
    d.arrow([(680,474),(790,474)])
    d.arrow([(925,518),(925,620)])
    d.arrow([(790,664),(680,664)],'写入标签',(735,647),True)
    d.arrow([(680,664),(735,664),(735,540),(855,540),(855,518)],'检索近邻',(746,580))
    d.arrow([(1060,309),(1090,309),(1090,742),(545,742),(545,708)],'示教也进入评分记忆',(800,742),True)
    d.save('依据论文 §3、附录 D / F 绘制；实线为推理数据流，虚线为记忆更新。')


def sam3():
    d=Diagram('sam3','SAM 3','概念提示驱动实例分割；视频中结合检测与带记忆的跟踪',760)
    d.box(40,112,270,'图像 / 当前视频帧','视觉编码器','input')
    d.box(790,112,270,'概念提示','文本短语 / 正负图像示例','input')
    d.box(410,265,270,'提示条件化特征','视觉与提示编码 · 注意力融合')
    d.box(100,420,350,'实例检测与掩码头','目标查询 → 位置 / 掩码 / 实例分数')
    d.box(650,420,350,'全局存在性判断','presence token → 概念存在分数')
    d.box(100,610,350,'图像实例分割结果','实例分数 × 概念存在分数','output')
    d.box(650,610,350,'视频跟踪器与记忆','传播掩码 · 匹配检测 · 维护身份','memory')
    d.arrow([(175,200),(175,230),(475,230),(475,265)])
    d.arrow([(925,200),(925,230),(615,230),(615,265)])
    d.arrow([(475,353),(475,387),(275,387),(275,420)])
    d.arrow([(615,353),(615,387),(825,387),(825,420)])
    d.arrow([(275,508),(275,610)])
    d.arrow([(825,508),(825,558),(345,558),(345,610)])
    d.arrow([(450,654),(650,654)],'当前检测',(550,635))
    d.arrow([(1000,654),(1052,654),(1052,584),(930,584),(930,610)],'历史帧',(1040,565),True)
    d.save('依据论文 §2–3 绘制；另支持点 / 框 / 掩码交互式细化指定实例。')


def claro():
    d=Diagram('claro','CLARO','VLM 根据当前场景和交互历史选择工具，执行后重新观察与规划',800)
    d.box(40,112,300,'任务与当前观测','语言目标 + RGB-D','input')
    d.box(720,112,340,'交互历史记忆','图像 · 工具回执 · 计划 / 场景判断','memory')
    d.box(360,265,380,'VLM 规划器','选择操作对象、工具和下一步计划')
    d.box(40,430,230,'survey_scene','场景图 · 物体命名 / 位置','geometry')
    d.box(300,430,230,'locate_object','SAM 3 掩码 → 三维点云','geometry')
    d.box(570,430,230,'grasp','GraspGen 六自由度抓取')
    d.box(830,430,230,'push','生成推移方向与三维路点')
    d.box(360,620,380,'执行与场景更新','移走挡物 / 推开挡物 / 抓取目标','output')
    d.arrow([(190,200),(190,235),(460,235),(460,265)])
    d.arrow([(890,200),(890,235),(640,235),(640,265)])
    d.arrow([(550,353),(550,390),(155,390),(155,430)])
    d.arrow([(550,390),(415,390),(415,430)])
    d.arrow([(550,390),(685,390),(685,430)])
    d.arrow([(550,390),(945,390),(945,430)])
    d.arrow([(155,518),(155,570),(285,570),(285,320),(360,320)],'感知结果',(210,550),True)
    d.arrow([(415,518),(415,550),(285,550)],feedback=True)
    d.arrow([(685,518),(685,620)])
    d.arrow([(945,518),(945,575),(685,575),(685,620)])
    d.arrow([(360,664),(20,664),(20,156),(40,156)],'重新观测',(130,642),True)
    d.arrow([(740,664),(1080,664),(1080,156),(1060,156)],'追加动作与结果',(970,642),True)
    d.save('依据论文 §3 绘制；实线为工具调用与执行，虚线为反馈。感知工具返回规划器后再选择动作。')


def update_site_font():
    root=Path(__file__).resolve().parents[1]
    target=root/'页面/notosc.woff'
    previous=TTFont(target)
    characters=set(previous.getBestCmap());previous.close()
    for path in (root/'内容').rglob('*'):
        if path.suffix in {'.md','.json'}:
            characters.update(map(ord,path.read_text()))
    for path in list((root/'页面').rglob('*')) + list(root.glob('*.mjs')):
        if path.suffix in {'.mjs','.jsx'}:
            characters.update(map(ord,path.read_text()))
    font=TTFont(Path(__file__).resolve().parents[2] / '缓存/对比视频字体/NotoSansCJKsc-Regular.otf')
    selected=subset.Subsetter()
    selected.populate(unicodes=characters & set(font.getBestCmap()))
    selected.subset(font);font.flavor='woff'
    temporary=target.with_suffix('.pending');font.save(temporary);temporary.replace(target)


if __name__=='__main__':
    for draw in [zerograsp,graspcl,sam3,claro]:draw()
    update_site_font()
    print(f'已生成 4 张论文架构图：{DEST}')
