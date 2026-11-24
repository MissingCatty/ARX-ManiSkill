"""Register usable existing local object USDs without authoring their physics."""
import argparse
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,digest,read,sha,write
from .asset_contract import source_contract
from .sampler import composed_mesh


def geometry_key(mesh):
    # Exact composed render geometry only; no claims of near-shape equivalence.
    return digest({'v':np.round(np.asarray(mesh.vertices),8).tolist(),'f':np.asarray(mesh.faces).tolist()})


def discover(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    original=read(ROOT/'runs/data_factory_20261030_15/catalog.json')['objects']
    existing_names={a['object_id'] for a in original};keys=set()
    for a in original:
        mesh,_=composed_mesh(a['usd_path']);keys.add(geometry_key(mesh))
    base=ROOT/'asset/tabletop_cache/Assets/Isaac/6.0/Isaac/IsaacLab/Arena/assets/object_library/srl_robolab_assets/objects'
    catalog=[];receipts=[]
    for path in sorted(base.rglob('*.usd')):
        row={'path':str(path),'name':path.stem,'available':False}
        if path.stem in existing_names:row['reason']='existing catalog object';receipts.append(row);continue
        try:
            contract=source_contract(path)
            mesh,evidence=composed_mesh(path);size=np.asarray(mesh.extents)
            if min(size)<.002 or max(size)>.65:raise ValueError('Outside registered tabletop object dimensions 2mm..650mm')
            # Large bodies can have thin rims/handles; aperture applies to local
            # opposing contacts, never the object's entire bounding box.
            key=geometry_key(mesh)
            if key in keys:row['reason']='exact composed geometry already present';receipts.append(row);continue
            keys.add(key)
            # A new scene version gets an explicit source-pivot-aware placement.
            placement={'position':[4.048,2.08,float(.679-mesh.bounds[0,2]+.012)],'orientation':[1.,0.,0.,0.]}
            catalog.append({'name':path.stem,'path':str(path),'container_path':str(path),'source':str(path),
                'size_m':size.tolist(),'mass_kg':contract['expected_mass_kg'],'placement':placement,
                'usage_conditions':'Existing NVIDIA Isaac object-library asset; retain upstream asset licensing/terms',
                'category':path.relative_to(base).parts[0],
                'physics_contract':contract,'registered_original_sha256':sha(path),
                'new_scene_version':'sdg_stage1_additional_objects_20261031',
                'exact_visual_geometry_key':key,'near_shape_deduplication':'Not established'})
            row.update(available=True,size_m=size.tolist(),mass_kg=contract['expected_mass_kg'])
        except (ValueError,RuntimeError) as exc:row['reason']=str(exc)
        receipts.append(row)
    write(out/'catalog.json',catalog)
    write(out/'receipt.json',{'candidates':receipts,'accepted_additional_objects':len(catalog),
        'original_objects_unchanged':len(original),'geometry_dedup':'Exact composed render mesh; no near-shape guarantee',
        'physics_mutations':False,'source':'Existing local USD properties only, no inferred replacement mass/friction/inertia'})
    print('ADDITIONAL_OBJECTS',len(catalog),[a['name'] for a in catalog])
    return catalog


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();discover(a.out)
