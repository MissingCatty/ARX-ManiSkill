"""Query advisory position/orientation coverage and original-model IK seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from 公共 import ROOT,read,save
from 能力索引 import CapabilityMap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--地图',type=Path,default=ROOT/'产物/首版_01')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--目标',type=Path);group.add_argument('--位置',type=float,nargs=3)
    parser.add_argument('--坐标系',choices=['body4','world'],default='body4')
    parser.add_argument('--输出',type=Path)
    args=parser.parse_args();atlas=CapabilityMap(args.地图)
    if args.目标:result=atlas.query(read(args.目标))
    else:
        position=np.array(args.位置)
        if args.坐标系=='world':position=(np.linalg.inv(atlas.base)@np.r_[position,1])[:3]
        result=atlas.coverage(position)
    result['map_load_seconds']=atlas.load_seconds
    if args.输出:save(args.输出,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
