"""Evaluate DAPO binary and multi-type tasks in one dataset pass."""

import argparse
import json
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DATASET_NAMES, build_dataset, validate_dataset
from .model import ModelConfig, build_model
from .metrics import summarize_records
from .evaluation_protocol import DAPO_DEFECT_NAMES, dapo_anomaly_map
from .multitype import (PROTOCOLS, MultitypeRecords, defect_id, multitype_maps,
                        product_labels, validate_multitype_partition)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', help='Experiment JSON required when loading an original DAPO checkpoint')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--dataset', required=True, choices=DATASET_NAMES)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', choices=('both', 'binary', 'multi-type'), default='both',
                        help='Default: both binary and multi-type evaluation')
    parser.add_argument('--defect-names', help='Override binary target vocabulary with a JSON list of defect names')
    parser.add_argument('--scratch-dir', help='Parent directory for temporary multi-type prediction files')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    partition = validate_dataset(args.data_root, args.dataset)
    binary_enabled = args.tasks in ('both', 'binary')
    typed_enabled = args.tasks in ('both', 'multi-type')
    if args.defect_names and not binary_enabled:
        parser.error('--defect-names applies only to binary evaluation')
    if typed_enabled:
        validate_multitype_partition(args.dataset, partition)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'blocks' in saved:
        config = ModelConfig(**saved['model_config'])
    else:
        if not args.config:
            parser.error('--config is required for original DAPO checkpoints')
        experiment = json.loads(Path(args.config).read_text())
        config = ModelConfig(**experiment['model'])
    binary_names = list(DAPO_DEFECT_NAMES[args.dataset])
    if args.defect_names:
        binary_names = json.loads(Path(args.defect_names).read_text())
        if (not isinstance(binary_names, list) or not binary_names
                or any(not isinstance(name, str) or not name.strip() for name in binary_names)
                or len(set(binary_names)) != len(binary_names)):
            parser.error('--defect-names must contain a nonempty JSON list of unique strings')
    typed_names = PROTOCOLS[args.dataset]['defect_names'] if typed_enabled else []
    # Encode the union once. Each task selects its own anchors before softmax.
    names = list(dict.fromkeys((binary_names if binary_enabled else []) + typed_names))
    model, preprocess = build_model(config, names, args.device)
    if 'blocks' in saved:
        model.load_block_state_dict(saved['blocks'])
    else:
        model.load_dapo_checkpoint(saved)
    model.eval()
    dataset = build_dataset(args.data_root, args.dataset, preprocess, config.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    records = {}
    counts = {product: len(rows) for product, rows in partition.items()}
    binary_indices = [0] + [names.index(name) + 1 for name in binary_names] if binary_enabled else []
    with tempfile.TemporaryDirectory(prefix='clip-ad-eval-', dir=args.scratch_dir) as scratch:
        typed_records = MultitypeRecords(scratch, counts, args.dataset) if typed_enabled else None
        try:
            with torch.no_grad():
                for batch in tqdm(loader):
                    outputs = model(batch['img'].to(args.device))
                    product = batch['class_name'][0]
                    mask = (batch['mask'][0, 0] > 0.5).numpy()
                    if binary_enabled:
                        binary = model.readout(
                            outputs['image_embeddings'], outputs['patch_embeddings'],
                            outputs['text_embeddings'][binary_indices], (config.image_size, config.image_size))
                        record = records.setdefault(product, {'labels': [], 'scores': [], 'masks': [], 'maps': []})
                        record['labels'].append(batch['anomaly'].item())
                        record['scores'].append(binary['image_logits'].softmax(-1)[0, 1].item())
                        record['masks'].append(mask)
                        record['maps'].append(dapo_anomaly_map(binary)[0].cpu().numpy())
                    if typed_enabled:
                        labels = product_labels(args.dataset, product)
                        indices = [0] + [names.index(label) + 1 for label in labels[1:]]
                        maps = multitype_maps(outputs, indices, (config.image_size, config.image_size),
                                              config.temperature)
                        label_id = defect_id(args.dataset, product, batch['defect_type'][0],
                                             batch['anomaly'].item())
                        typed_records.append(product, mask.astype('uint8') * label_id,
                                             maps[0].permute(1, 2, 0).cpu().numpy())
            binary_result = None
            if binary_enabled:
                metrics, macro = summarize_records(records)
                binary_result = dict(classes=metrics, macro=macro)
            typed_result = typed_records.summarize() if typed_enabled else None
        finally:
            if typed_records is not None:
                typed_records.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        dataset=args.dataset, defect_names=binary_names if binary_enabled else typed_names,
        evaluation_protocol='dapo_binary_multitype_v1', tasks=args.tasks,
        vocabulary_source=('custom' if args.defect_names else 'DAPO/test.py')
                          if binary_enabled else 'DAPO/test_multi_defect.py',
        pixel_score='sum_layers_sum_defect_probabilities' if binary_enabled
                    else 'product_softmax_then_upsample_then_sum_layers',
        checkpoint=str(Path(args.checkpoint).resolve()), epoch=saved.get('epoch'),
        data_root=str(Path(args.data_root).resolve()),
        sample_counts=counts,
    )
    if binary_enabled:
        report['binary'] = binary_result
        # Preserve the original report paths for existing comparison scripts.
        report.update(binary_result)
    if typed_enabled:
        typed_result.update(defect_names=typed_names, vocabulary_source='DAPO/test_multi_defect.py',
                            includes_normal=True, f1_rule='argmax_macro',
                            undefined_auroc='null; excluded from macro',
                            pixel_score='product_softmax_then_upsample_then_sum_layers')
        report['multi_type'] = typed_result
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key]['macro'] for key in ('binary', 'multi_type') if key in report}, indent=2))


if __name__ == '__main__':
    main()
