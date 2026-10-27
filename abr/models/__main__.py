"""Construct, inspect, or smoke-test small decision models. No ABR training yet."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .factory import build_model, MODEL_KINDS
from .checkpoint import load_model, save_initial_model
from .policy import DecisionPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Initialize random weights, explicitly marked untrained')
    init.add_argument('--kind', choices=MODEL_KINDS, default='gru')
    init.add_argument('--history', type=int, default=8)
    init.add_argument('--seed', type=int, default=0)
    init.add_argument('--output', required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--checkpoint', required=True)
    rollout = sub.add_parser('rollout', help='Diagnostic integration test, not a trained-policy benchmark')
    rollout.add_argument('--checkpoint', required=True)
    rollout.add_argument('--dataset', required=True)
    rollout.add_argument('--media')
    rollout.add_argument('--mode', choices=['vod', 'live'], default='vod')
    rollout.add_argument('--config')
    rollout.add_argument('--split', choices=['train', 'validation'], default='validation')
    rollout.add_argument('--seed', type=int, default=0)
    rollout.add_argument('--allow-untrained', action='store_true')
    rollout.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        if args.command == 'init':
            if args.seed < 0:
                raise ValueError('Seed must be nonnegative')
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(args.seed)
                model = build_model({'kind': args.kind, 'history_length': args.history})
            metadata = save_initial_model(model, args.output, seed=args.seed)
            print(json.dumps({'output': args.output, **metadata}, indent=2))
            return
        model, metadata = load_model(args.checkpoint)
        if args.command == 'inspect':
            print(json.dumps(metadata, indent=2))
            return
        if metadata['training_status'] == 'untrained' and not args.allow_untrained:
            raise ValueError('This checkpoint is UNTRAINED. Use --allow-untrained for integration testing only')
        from ..data import Dataset
        from ..media import Media
        from ..env import StreamingEnv, StreamConfig
        from ..benchmark import evaluate
        cfg = json.loads(Path(args.config).read_text()) if args.config else {}
        cfg['mode'] = args.mode
        env = StreamingEnv(Dataset.load(args.dataset), media=Media.load(args.media) if args.media else None,
                           config=StreamConfig(**cfg), split=args.split, history=model.config.history_length)
        artifact = {**metadata, 'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    'purpose': 'UNTRAINED architecture integration test; no policy quality claim'}
        report = evaluate(env, baseline='model', seed=args.seed, policy=DecisionPolicy(model), policy_metadata=artifact)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'output': str(output), 'training_status': 'untrained',
                          'episodes': len(report['episodes']), 'completion_rate': report['completion_rate']}))
    except (ValueError, KeyError, TypeError) as error:
        p.exit(2, f'Model input error: {error}\n')


if __name__ == '__main__':
    main()
