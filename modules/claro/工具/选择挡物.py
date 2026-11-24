"""根据同一帧腕部 RGB 生成场景图并选择一个挡物。"""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import time

import httpx
import yaml


ENDPOINT = 'https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1'
MODEL = 'qwen4-max'
RELATIONS = {'on_top_of', 'under', 'partially_occluded_by', 'next_to', 'left-of', 'right-of', 'behind'}
SURVEY = '''Read the unannotated oblique wrist image of an ARX tabletop. Produce a scene graph in YAML, with scene_summary and objects. Use one entry per physical tabletop instance, including obj_id (positive integer), name, desc, point_norm ([row, col] in [0, 1], with origin at upper left and the point on visible pixels of that object), position_description, and relations (pairs [relation_type, other_obj_id]). Relations can be on_top_of, under, partially_occluded_by, next_to, left-of, right-of, or behind. Distinguish similar instances by color, shape, or their place in the image. Exclude the robot hand and background. Output strict YAML without Markdown fences.'''
PLANNER = '''You choose the next tool for an ARX robot retrieving the named target after direct grasp planning failed. Use the current wrist RGB and latest scene graph to choose one visible blocker whose removal could expose the target or clear the approach. Choose locate_object with the exact object_name and object_id from the graph, or survey_scene if the evidence is insufficient. Use only one JSON object with tool and a brief evidence-based belief. Include both object_name and object_id for locate_object. Output strict JSON without Markdown fences.'''


def strip_fence(text, syntax):
    match = re.fullmatch(rf'```(?:{syntax})?\s*\n(.*?)\n```', text.strip(), flags=re.I | re.S)
    return (match.group(1), True) if match else (text.strip(), False)


def parse_graph(body):
    graph = yaml.safe_load(body)
    if not isinstance(graph, dict) or not isinstance(graph.get('scene_summary'), str):
        raise ValueError('场景图缺少概述')
    objects = graph.get('objects')
    if not isinstance(objects, list) or not objects:
        raise ValueError('场景图缺少可见物体')
    ids = set()
    for obj in objects:
        identifier = obj['obj_id']
        point = obj['point_norm']
        if type(identifier) is not int or identifier <= 0 or identifier in ids:
            raise ValueError('场景图的实例编号无效或重复')
        if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in ('name', 'desc', 'position_description')):
            raise ValueError('场景图缺少有效实例描述')
        if (not isinstance(point, list) or len(point) != 2
                or not all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in point)):
            raise ValueError('场景图二维点无效')
        ids.add(identifier)
    for obj in objects:
        if not isinstance(obj.get('relations'), list):
            raise ValueError('场景图的物体关系无效')
        for relation, other in obj['relations']:
            if relation not in RELATIONS or type(other) is not int or other not in ids or other == obj['obj_id']:
                raise ValueError('场景图关系引用了无效实例')
    return graph


def credential(config_path):
    config = json.loads(config_path.read_text())
    if config['base_url'].rstrip('/') != ENDPOINT:
        raise ValueError('云端地址与本轮登记的 Token Plan 端点不符')
    path = Path(config['keys_file']).expanduser()
    if path.stat().st_mode & 0o077:
        raise ValueError('凭据文件权限须为 600')
    keys = json.loads(path.read_text())
    if not isinstance(keys, list) or len(keys) != 1 or not isinstance(keys[0], str) or not keys[0].strip():
        raise ValueError('凭据文件应包含一个可用的 Key')
    return keys[0].strip()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--trial', required=True, type=Path)
    cli.add_argument('--task', required=True)
    cli.add_argument('--config', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    stage = json.loads((args.trial / 'plan_result.json').read_text())
    image = args.trial / 'camera/color.png'
    rgb = image.read_bytes()
    data = {'scene': args.trial.name.split('_')[0], 'model': MODEL, 'image': str(image.resolve()),
            'image_sha256': hashlib.sha256(rgb).hexdigest(), 'direct_plan_success': stage['success'],
            'direct_plan_reason': stage.get('reason'), 'task': args.task, 'ground_truth_in_model_input': False,
            'simulation_or_motion_executed': False, 'calls': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    def save():
        temp = args.output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temp.replace(args.output)

    if stage['success']:
        data['status'] = 'skip_direct_plan_passed'
        save()
        print(data['scene'], data['status'])
        return
    if stage.get('reason') != 'no_feasible_native_post_or_recalled_candidate':
        raise ValueError('当前直接规划失败未分类为无可执行抓取方案')

    content = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(rgb).decode()}}
    key = credential(args.config)

    def ask(client, tool, system, prompt):
        rec = dict(tool=tool, system=system, prompt=prompt, image=str(image.resolve()))
        data['calls'].append(rec)
        save()
        start = time.monotonic()
        try:
            response = client.post(ENDPOINT + '/chat/completions', headers={'Authorization': 'Bearer ' + key}, json={
                'model': MODEL,
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': [content, {'type': 'text', 'text': prompt}]}],
                'temperature': 0.0, 'max_tokens': 1800, 'enable_thinking': False, 'stream': False,
            })
        except httpx.RequestError as error:
            rec['request_error'] = type(error).__name__
            save()
            raise RuntimeError('Token Plan 请求未完成') from None
        rec.update(http_status=response.status_code, wall_seconds=round(time.monotonic() - start, 3))
        save()
        if response.status_code != 200:
            raise RuntimeError(f'Token Plan 返回 HTTP {response.status_code}')
        payload = response.json()
        choice = payload['choices'][0]
        rec.update(response_model=payload.get('model'), finish_reason=choice.get('finish_reason'),
                   output=choice['message']['content'], usage=payload.get('usage'))
        save()
        if rec['finish_reason'] != 'stop' or not isinstance(rec['output'], str):
            raise ValueError('模型回复不完整')
        return rec

    with httpx.Client(timeout=60., trust_env=False, follow_redirects=False) as client:
        survey = ask(client, 'survey_scene', SURVEY, 'Survey the visible tabletop instances and their relationships.')
        body, wrapped = strip_fence(survey['output'], 'yaml|yml')
        graph = parse_graph(body)
        survey['parse'] = {'outer_markdown_fence_removed': wrapped, 'passed': True, 'parsed_scene_graph': graph}
        prompt = (f'Task: {args.task}\nTOOL RESULT: direct grasp planning failed.\n'
                  f'TOOL RESULT: survey_scene succeeded.\n{body}')
        plan = ask(client, '挡物选择', PLANNER, prompt)
        body, wrapped = strip_fence(plan['output'], 'json')
        decision = json.loads(body)
        if decision['tool'] not in {'locate_object', 'survey_scene'} or not isinstance(decision.get('belief'), str):
            raise ValueError('模型选择工具或理由无效')
        if decision['tool'] == 'locate_object':
            matches = [r for r in graph['objects'] if r['obj_id'] == decision.get('object_id')
                       and r['name'] == decision.get('object_name')]
            if len(matches) != 1:
                raise ValueError('模型选择的挡物无法绑定场景图实例')
        plan['parse'] = {'outer_markdown_fence_removed': wrapped, 'passed': True}
        data['decision'] = decision
        data['status'] = 'selected' if decision['tool'] == 'locate_object' else 'needs_observation'
        save()
        print(data['scene'], json.dumps(decision, ensure_ascii=False))


if __name__ == '__main__':
    main()
