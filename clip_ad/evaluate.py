"""Server binary evaluation using the same composed model as training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .data import build_dataset
from .model import ModelConfig, build_model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', help='Experiment JSON required when loading an original DAPO checkpoint')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--defect-names', help='JSON list of candidate defect names; default: training names')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'blocks' in saved:
        config = ModelConfig(**saved['model_config'])
        names = saved['defect_names']
    else:
        if not args.config:
            parser.error('--config is required for original DAPO checkpoints')
        experiment = json.loads(Path(args.config).read_text())
        config = ModelConfig(**experiment['model'])
        names = experiment['defect_names']
    if args.defect_names:
        names = json.loads(Path(args.defect_names).read_text())
    model, preprocess = build_model(config, names, args.device)
    if 'blocks' in saved:
        model.load_block_state_dict(saved['blocks'])
    else:
        model.load_dapo_checkpoint(saved)
    model.eval()
    dataset = build_dataset(args.data_root, args.dataset, preprocess, config.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    records = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            outputs = model(batch['img'].to(args.device))
            record = records.setdefault(batch['class_name'][0], {'labels': [], 'scores': [], 'masks': [], 'maps': []})
            record['labels'].append(batch['anomaly'].item())
            record['scores'].append(outputs['image_logits'].softmax(-1)[0, 1].item())
            record['masks'].append((batch['mask'][0, 0] > 0.5).numpy())
            record['maps'].append(outputs['binary_probabilities'][0, 1].cpu().numpy())
    metrics = {}
    for name, record in records.items():
        def pair(truth, scores):
            truth, scores = np.asarray(truth).ravel(), np.asarray(scores).ravel()
            if np.unique(truth).size < 2:
                return {'auroc': None, 'ap': None}
            return {'auroc': float(roc_auc_score(truth, scores)), 'ap': float(average_precision_score(truth, scores))}
        metrics[name] = {'image': pair(record['labels'], record['scores']), 'pixel': pair(record['masks'], record['maps'])}
    if not metrics:
        raise ValueError('Evaluation dataset is empty')
    macro = {}
    for level in ('image', 'pixel'):
        macro[level] = {}
        for metric in ('auroc', 'ap'):
            values = [result[level][metric] for result in metrics.values() if result[level][metric] is not None]
            macro[level][metric] = float(np.mean(values)) if values else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(dataset=args.dataset, defect_names=names, classes=metrics, macro=macro), indent=2) + '\n')
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
