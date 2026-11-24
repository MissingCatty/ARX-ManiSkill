"""Draw the local ARX workflow and distinguish its two validated upstream routes."""
from html import escape
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paper_diagrams',ROOT/'工具/生成论文架构图.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
TEXT=module.Diagram('unused','','')
DEST=ROOT/'内容/资源/CLARO流程/CLARO复现流程.svg'
WORKFLOW=json.loads((ROOT/'内容/流程/CLARO流程.json').read_text())
LINKS={s['number']:f'/iterations/claro/steps/{s["id"]}/' for s in WORKFLOW['steps']}

COLORS={
    'input':('#f1f4f9','#d7deea','#42546f'),
    'vision':('#edf2ff','#bdcdf5','#3558a5'),
    'geometry':('#eaf7f3','#b4dacb','#236b58'),
    'execute':('#edf7f5','#acd2ca','#24645b'),
    'fixed':('#fff5e6','#e8c991','#97651c'),
    'pending':('#f5f5f7','#d6d8df','#747b89'),
}
nodes=[];edges=[]


def text(x,y,value,size=20,color='#556174',anchor='middle',bold=False,width=None):
    return TEXT.text(x,y,value,size,color,anchor,700 if bold else 400,width)


def box(x,y,w,h,title,lines,kind='geometry',href=None):
    fill,border,ink=COLORS[kind]
    dash=' stroke-dasharray="7 5"' if kind=='pending' else ''
    body=[f'<g role="group" aria-label="{escape(title,quote=True)}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{fill}" stroke="{border}" stroke-width="1.5"{dash}/>',
          text(x+w/2,y+34,title,25,ink,bold=True,width=w-28)]
    for i,line in enumerate(lines):body.append(text(x+w/2,y+63+i*25,line,18,'#526174',width=w-26))
    node=''.join(body)+'</g>'
    href=href or LINKS.get(title.split()[0])
    if href:node=f'<a href="{escape(href,quote=True)}" target="_top" aria-label="查看{escape(title,quote=True)}详情">'+node+'</a>'
    nodes.append(node)


def arrow(points,label=None,label_at=None,kind='normal'):
    color={'normal':'#8190a6','fixed':'#bc893c','pending':'#9aa0ad','aux':'#a1aaba'}[kind]
    dash=' stroke-dasharray="6 5"' if kind in ['fixed','pending','aux'] else ''
    edges.append(f'<polyline points="'+ ' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" marker-end="url(#{kind}-arrow)"{dash}/>')
    if label:
        x,y=label_at;w=max(50,len(label)*17+18)
        edges.append(f'<rect x="{x-w/2}" y="{y-19}" width="{w}" height="25" rx="5" fill="white"/>')
        edges.append(text(x,y,label,17,color))


def main():
    box(60,110,540,90,'01  腕部 RGB ＋任务目标',['同一观测中的原目标描述与可见物体'],'input')
    box(660,110,290,90,'几何辅助输入',['同帧深度 · 相机标定','基准任务提供目标实例掩码'],'input',LINKS['01'])
    box(60,245,540,90,'02  检查原目标能否直接抓取',['目标 RGB-D／掩码 → ZeroGrasp 候选＋cuRobo'],'geometry')
    box(660,245,290,90,'可执行：直接抓取',['执行原目标抓取并核验','不进入移障分支'],'execute',LINKS['02'])

    box(60,395,540,100,'03  survey_scene：生成场景图',['Qwen4-Max 从原始腕部 RGB 输出 YAML','物体 ID／外观／二维位置／相邻与遮挡关系'],'vision')
    box(60,540,540,100,'04  VLM 选择一个挡物',['任务目标＋场景图＋直接抓取失败回执','输出 locate_object 与具体物体 ID、名称'],'vision')
    box(60,685,540,100,'05  SAM3 定位挡物实例',['场景图二维点 → 实例掩码与边界框','结合深度与标定 → 可见三维点云'],'vision')

    box(660,395,290,180,'13 例实测分支',['根据历史碰撞证据','预先固定一个挡物','使用仿真实例掩码','未在线运行左侧三步'],'fixed',LINKS['实测分支'])
    box(660,635,290,150,'左侧视觉模块的进度',['首批 3 例固定图验证','场景图、选挡与定位已保存','尚未串入 13 例自动选挡'],'vision','/iterations/v0.2/')

    box(60,830,540,90,'06  ZeroGrasp 生成挡物抓点',['输入挡物 RGB-D／掩码，使用原生后处理候选'],'geometry')
    box(60,965,540,90,'07  cuRobo 规划并复核运动',['抓取与预抓取 IK · 场景碰撞 · 进场和接近路径'],'geometry')
    box(60,1100,540,90,'08  移走挡物',['夹起 → 搬运 → 放下 → 松爪与撤离'],'execute')
    box(60,1235,540,100,'09  重新采集腕部观测',['新 RGB-D／原目标掩码','更新三维碰撞世界，供后续规划使用'],'input')
    box(60,1380,540,90,'10  重新抓取原目标',['新观测 → ZeroGrasp 新抓点 → cuRobo 规划'],'geometry')
    box(60,1515,540,90,'11  执行与独立结果核验',['闭合、抬升、持续保持，检查接触与实际物体高度'],'execute')
    box(660,1005,290,180,'执行段的实际结果',['移障 → 重观测 → 重抓','I0069 完成全链路','13 例实际成功仍为 1/13','后续离线恢复另行记录'],'execute',LINKS['11'])
    box(60,1660,890,90,'待接入：多轮动态工具编排',['根据工具回执更新语义场景图、重新选挡并安排下一轮动作'],'pending',LINKS['待接入'])

    arrow([(330,200),(330,245)])
    arrow([(600,290),(660,290)],'可执行',(630,279))
    arrow([(330,335),(330,395)],'直接抓取受阻',(330,369))
    arrow([(585,335),(635,335),(635,365),(805,365),(805,395)],kind='fixed')
    arrow([(330,495),(330,540)])
    arrow([(330,640),(330,685)])
    arrow([(330,785),(330,830)])
    arrow([(805,575),(805,600),(630,600),(630,875),(600,875)],kind='fixed')
    arrow([(950,155),(978,155),(978,735),(600,735)],'深度与标定',(890,720),kind='aux')
    arrow([(685,200),(630,200),(630,220),(550,220),(550,245)],kind='aux')
    for bottom,top in [(920,965),(1055,1100),(1190,1235),(1335,1380),(1470,1515)]:arrow([(330,bottom),(330,top)])
    arrow([(330,1605),(330,1660)],kind='pending')

    heading=text(40,45,'CLARO 本地复现流程',31,'#26344d','start',True)
    subtitle=text(40,80,'从腕部 RGB 开始；视觉选挡验证与固定挡物实测分开标注',20,'#65738a','start')
    note=text(40,1794,'蓝色：首批视觉验证　金色虚线：13 例固定挡物路线　灰色虚线：待接入模块',18,'#65738a','start',width=920)
    markers=[]
    for name,color in [('normal','#8190a6'),('fixed','#bc893c'),('pending','#9aa0ad'),('aux','#a1aaba')]:
        markers.append(f'<marker id="{name}-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9" fill="{color}"/></marker>')
    glyphs=''.join(f'<path id="{key}" d="{path}"/>' for key,path in TEXT.glyphs.items())
    svg='\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1830" viewBox="0 0 1000 1830" role="img" aria-labelledby="title description">',
        '<title id="title">CLARO 本地复现：从腕部 RGB 到移障、重新观测和原目标重抓</title>',
        '<desc id="description">依次展示直接抓取判断、Qwen4-Max 场景图、VLM 选挡、SAM3 定位、ZeroGrasp、cuRobo、移障、重观测、重抓和独立判定。蓝色视觉模块仅完成首批三例固定图验证；13例固定挡物实测使用金色路线和仿真实例掩码。多轮动态场景图更新与工具编排仍待接入。</desc>',
        '<defs>'+glyphs+''.join(markers)+'</defs>',
        '<rect width="1000" height="1830" rx="16" fill="white"/>',heading,subtitle,*edges,*nodes,note,'</svg>'])
    DEST.parent.mkdir(parents=True,exist_ok=True);DEST.write_text(svg+'\n')
    print('流程图已生成',DEST,'bytes',DEST.stat().st_size)


if __name__=='__main__':main()
