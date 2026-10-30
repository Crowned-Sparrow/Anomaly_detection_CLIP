"""Server training entry point using the composed model and objectives."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset
from .losses import build_loss_composer, typed_targets
from .model import ModelConfig, build_model, checkpoint_payload


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, help='Override total epochs, e.g. 1 for a smoke test')
    parser.add_argument('--resume', help='Resume a clip_ad checkpoint, including optimizer and RNG')
    parser.add_argument('--init-dapo', help='Initialize learned blocks from an original DAPO checkpoint')
    args = parser.parse_args()
    if args.resume and args.init_dapo:
        parser.error('--resume and --init-dapo are mutually exclusive')
    experiment = json.loads(Path(args.config).read_text())
    seed = experiment['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = ModelConfig(**experiment['model'])
    model, preprocess = build_model(config, experiment['defect_names'], args.device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=experiment['learning_rate'], betas=(0.5, 0.999))
    criterion = build_loss_composer(experiment).to(args.device)
    start = 0
    if args.init_dapo:
        model.load_dapo_checkpoint(torch.load(args.init_dapo, map_location='cpu', weights_only=False))
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['model_config'] != checkpoint_payload(model)['model_config'] or saved['experiment'] != experiment:
            raise ValueError('Resume requires the same model and experiment configuration')
        model.load_block_state_dict(saved['blocks'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['epoch'] + 1
        torch.set_rng_state(saved['rng']['torch'])
        random.setstate(saved['rng']['python'])
        np.random.set_state(saved['rng']['numpy'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    dataset = build_dataset(args.data_root, experiment['source_dataset'], preprocess, config.image_size)
    loader = DataLoader(dataset, batch_size=experiment['batch_size'], shuffle=True, num_workers=args.workers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
    epochs = args.epochs if args.epochs is not None else experiment['epochs']
    if epochs <= start:
        raise ValueError('Total epochs must exceed the checkpoint epoch')
    if any(not 0 <= value <= len(model.defect_names) for value in experiment['label_ids'].values()):
        raise ValueError('Source label IDs must index normal + candidate defect channels')
    print(f'Trainable parameters: {sum(p.numel() for p in model.trainable_parameters()):,}')
    for epoch in range(start, epochs):
        model.train()
        sums = {}
        progress = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in progress:
            images = batch['img'].to(args.device)
            targets = typed_targets(batch['mask'].to(args.device), batch['anomaly'].to(args.device),
                                    batch['defect_type'], experiment['label_ids'])
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(images), targets)
            losses['total'].backward()
            optimizer.step()
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + value.detach().item()
            progress.set_postfix(loss=losses['total'].detach().item())
        metrics = {name: value / len(loader) for name, value in sums.items()}
        if model.fusion is not None:
            metrics['spectral_gates'] = model.fusion.gates.detach().tanh().cpu().tolist()
        rng = dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        torch.save(checkpoint_payload(model, optimizer=optimizer.state_dict(), epoch=epoch,
                                     experiment=experiment, rng=rng), output / f'epoch_{epoch}.pth')
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(epoch=epoch, **metrics)) + '\n')
        print(metrics)


if __name__ == '__main__':
    main()
