"""Publish the verified 200-scene dataset as lazy, clickable head-view mosaics."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scene_annotations import mark_target, resize_mask

SITE = Path(__file__).resolve().parents[1]
MAINLINE = SITE.parent / 'modules/pipeline'
DATASET_CONFIG = MAINLINE / 'configs/scene_dataset.json'
DEST = SITE / '内容/资源/scene-gallery'
FONT = SITE.parent / '缓存/对比视频字体/NotoSansCJKsc-Regular.otf'
TILE_WIDTH, PHOTO_HEIGHT, LABEL_HEIGHT = 240, 180, 26
COLUMNS, ROWS = 8, 5
LAYOUTS = {'S1_open': '开阔桌面', 'S2_edge': '靠近桌边', 'S3_sparse': '稀疏混放',
           'S4_clutter': '中度杂乱', 'S5_occluded': '部分遮挡'}
LABELS = {'apple': '苹果', 'banana': '香蕉', 'orange': '橙子', 'pear': '梨', 'peach': '桃子',
          'lemon': '柠檬', 'mango': '芒果', 'avocado': '牛油果', 'bell_pepper': '甜椒',
          'carrot': '胡萝卜', 'corn': '玉米', 'cucumber': '黄瓜', 'eggplant': '茄子',
          'mushroom': '蘑菇', 'onion': '洋葱', 'potato': '土豆', 'tomato': '番茄',
          'broccoli': '西兰花', 'bread': '面包', 'baguette': '法棍', 'croissant': '羊角面包',
          'donut': '甜甜圈', 'canned_food': '食品罐头', 'boxed_food': '盒装食品',
          'bottled_water': '瓶装水', 'milk': '牛奶', 'yogurt': '酸奶', 'mug': '马克杯',
          'bowl': '碗', 'soap_dispenser': '洗手液瓶', 'bar_soap': '肥皂', 'spatula': '锅铲',
          'scissors': '剪刀', 'teapot': '茶壶', 'candle': '蜡烛', 'book': '书',
          'remote_control': '遥控器', 'cell_phone': '手机', 'sponge': '海绵', 'keychain': '钥匙串'}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = read_json(DATASET_CONFIG)
    root = Path(config['dataset']).resolve()
    verification = read_json(root / 'verification.json')
    manifest = read_json(root / 'manifest.json')
    registration = read_json(root / 'registration.json')
    entries = manifest['schedule']
    assert verification['passed'] and verification['scenes'] == 200
    assert verification['construction_independent_of_ik']
    frozen = read_json(root / 'frozen_dataset.json')
    assert frozen['construction_independent_of_ik'] and frozen['IK_evaluations_at_freeze'] == 0
    assert frozen['files']['manifest.json'] == digest(root / 'manifest.json')
    assert manifest['registration_sha256'] == digest(root / 'registration.json')
    assert Counter(e['layout'] for e in entries) == dict.fromkeys(LAYOUTS, 40)
    assert len(entries) == len({e['id'] for e in entries}) == 200
    assert len({e['model_id'] for e in entries}) == 80
    assert len({e['semantic_class'] for e in entries}) == 40
    DEST.mkdir(parents=True, exist_ok=True)
    assets, groups, source_bytes = {}, [], 0

    def save_image(image, name, quality):
        payload = io.BytesIO()
        image.save(payload, 'JPEG', quality=quality, optimize=True, progressive=True)
        data = payload.getvalue()
        filename = f'{name}-{hashlib.sha256(data).hexdigest()[:12]}.jpg'
        target = DEST / filename
        if not target.exists():
            target.write_bytes(data)
        assets[filename] = str(target.relative_to(SITE / '内容/资源'))
        return {'url': '/media/' + filename, 'bytes': len(data)}

    font = ImageFont.truetype(str(FONT), 18)
    for layout, title in LAYOUTS.items():
        cases = sorted((e for e in entries if e['layout'] == layout), key=lambda e: e['id'])
        assert len({e['semantic_class'] for e in cases}) == 40
        sheet = Image.new('RGB', (COLUMNS * TILE_WIDTH, ROWS * (PHOTO_HEIGHT + LABEL_HEIGHT)), '#f0f5f6')
        group = {'id': layout, 'title': title, 'cases': []}
        for index, entry in enumerate(cases):
            folder = root / 'cases' / entry['id']
            admission = read_json(folder / 'admission.json')
            construction = read_json(folder / 'construction_check.json')
            assert admission['passed'] and admission['entry'] == entry and construction['passed']
            assert construction['construction_independent_of_ik'] and construction['IK_queries'] == 0
            source = folder / 'camera/color.png'
            source_sha = digest(source)
            assert source_sha == admission['camera_input_sha256']['color.png']
            with Image.open(source) as original:
                image = original.convert('RGB')
            assert image.size == (640, 480)
            mask_source = folder / 'target_gt.png'
            with Image.open(mask_source) as original_mask:
                mask = np.asarray(original_mask.convert('L')) > 0
            assert mask.shape == (image.height, image.width) and mask.any()
            category = LABELS[entry['semantic_class']]
            marked, annotation = mark_target(image, mask, category, FONT)
            model = entry['model_id'].split('__')[-1]
            item = {'id': entry['id'], 'label': entry['id'].split('_')[0],
                    'target': category + ' · ' + model, 'model_id': entry['model_id'],
                    'semantic_class': entry['semantic_class'], 'seed': entry['seed'],
                    'source': str(source.relative_to(MAINLINE)), 'source_sha256': source_sha,
                    'camera': 'head', 'width': image.width, 'height': image.height,
                    'target_marked': True, 'target_mask_source': str(mask_source.relative_to(MAINLINE)),
                    'target_mask_sha256': digest(mask_source), 'annotation': annotation}
            item.update(save_image(marked, 'scene-' + item['label'].lower(), 88))
            source_bytes += source.stat().st_size
            cell = Image.new('RGB', (TILE_WIDTH, PHOTO_HEIGHT + LABEL_HEIGHT), '#f0f5f6')
            preview = ImageOps.contain(image, (TILE_WIDTH, PHOTO_HEIGHT), Image.Resampling.LANCZOS)
            preview, _ = mark_target(preview, resize_mask(mask, preview.size), category, FONT, thumbnail=True)
            cell.paste(preview, ((TILE_WIDTH - preview.width) // 2, (PHOTO_HEIGHT - preview.height) // 2))
            ImageDraw.Draw(cell).text((8, PHOTO_HEIGHT + 2), item['label'] + ' · ' + category,
                                      fill='#16343b', font=font)
            sheet.paste(cell, ((index % COLUMNS) * TILE_WIDTH, (index // COLUMNS) * (PHOTO_HEIGHT + LABEL_HEIGHT)))
            group['cases'].append(item)
        group['sheet'] = {**save_image(sheet, 'sheet-' + layout[:2].lower(), 76),
                          'width': sheet.width, 'height': sheet.height}
        groups.append(group)
    result = {'groups': groups, 'assets': assets, 'registered': 200, 'screenshots': 200, 'missing': 0,
              'categories': 40, 'models': 80, 'camera': 'head', 'dataset': str(root),
              'annotated_screenshots': 200,
              'annotation': 'Yellow external visible-instance contours and Chinese target labels from aligned simulator target masks',
              'annotation_source_sha256': digest(Path(__file__).with_name('scene_annotations.py')),
              'dataset_registration_sha256': digest(root / 'registration.json'),
              'dataset_verification_sha256': digest(root / 'verification.json'),
              'construction_independent_of_ik': True,
              'dataset_freeze_sha256': digest(root / 'frozen_dataset.json'),
              'generator_source_sha256': digest(Path(__file__)),
              'thumbnail_bytes': sum(g['sheet']['bytes'] for g in groups),
              'large_image_bytes': sum(c['bytes'] for g in groups for c in g['cases']),
              'source_png_bytes': source_bytes, 'lift_joint_m': registration['lift_joint_m']}
    target = SITE / '内容/场景图集.json'
    temp = target.with_suffix('.json.pending')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temp.replace(target)
    print(json.dumps({k: v for k, v in result.items() if k not in ['groups', 'assets']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
