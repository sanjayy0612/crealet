"""Freeze validation selection, then evaluate additional sessions once."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.benchmark import evaluate
from abr.data import Dataset
from abr.env import StreamingEnv, QoEWeights
from abr.learning_experiment import make_env
from abr.models.checkpoint import load_model
from abr.models.safety import ActionSafety, allowed_actions
from abr.models.train_dueling_dqn import evaluate_policy, baseline_summary, checkpoint_eligible


def reference(dataset, split, safety):
    report = evaluate(make_env(dataset, split), baseline='model',
                      policy=lambda obs, cfg: int(np.flatnonzero(allowed_actions(obs, safety))[-1]),
                      policy_metadata={'kind': 'highest_allowed_without_learning'})
    return baseline_summary(report)


def main():
    torch.set_num_threads(1)
    output = Path('runs/learning-ablation')
    inputs = json.loads((output / 'selection_inputs.json').read_text())
    summaries = {}
    for variant in ('baseline', 'credit', 'queue'):
        rows = [r for r in inputs if r['variant'] == variant]
        summaries[variant] = {'median_validation_reward': float(np.median([r['selected_validation']['mean_reward'] for r in rows])),
                              'median_validation_bitrate_kbps': float(np.median([r['selected_validation']['mean_played_bitrate_kbps'] for r in rows])),
                              'all_have_eligible_checkpoints': all(r['accepted_checkpoint'] for r in rows)}
    selected = max(summaries, key=lambda name: summaries[name]['median_validation_reward'])
    frozen = {'selection_basis': 'median original validation reward across seeds7,11; acceptance limits unchanged',
              'selected_variant': selected, 'validation': summaries,
              'checkpoints': [{'variant': r['variant'], 'seed': r['seed'], 'path': r['accepted_checkpoint'],
                               'sha256': hashlib.sha256(Path(r['accepted_checkpoint']).read_bytes()).hexdigest()} for r in inputs],
              'fresh_results_used_for_selection': False}
    # Persist the decision and exact artifacts BEFORE opening the fresh dataset.
    (output / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    fresh = Dataset.load('data/learning-holdout/dataset.json')
    original = Dataset.load('data/hard_envivio/dataset.json')
    safety = ActionSafety()
    rule = reference(fresh, 'test', safety)
    validation_rule = reference(original, 'validation', safety)
    throughput = baseline_summary(evaluate(StreamingEnv(fresh, split='test', weights=QoEWeights(rebuffer=12)), baseline='throughput'))
    limits = {e['trace']: min(8., e['rebuffer_s']) for e in throughput['episodes']}
    results = []
    for row in inputs:
        model, metadata = load_model(row['accepted_checkpoint'])
        env = make_env(fresh, 'test', model.config.queue_features)
        report = evaluate_policy(env, model, safety=safety, stall_limits_s=limits)
        results.append({'variant': row['variant'], 'seed': row['seed'], 'checkpoint_episode': metadata['training']['episode'],
                        'fresh': report, 'fresh_targets_met': checkpoint_eligible(report, 3.)})
    beats_rule = all(r['selected_validation']['mean_reward'] > validation_rule['mean_reward'] + 1e-6
                     for r in inputs if r['variant'] == selected)
    passes_fresh = all(r['fresh_targets_met'] for r in results if r['variant'] == selected)
    report = {'frozen_selection': frozen, 'fresh_dataset_sha256': fresh.fingerprint,
              'validation_highest_allowed': validation_rule,
              'fresh_highest_allowed': rule, 'fresh_original_throughput': throughput,
              'results': results, 'promotion_recommended': beats_rule and passes_fresh,
              'promotion_reason': 'Requires beating highest-allowed on validation in both seeds and passing fresh playback targets; fresh outcomes never select another arm.'}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'validation': summaries, 'selected_variant': selected,
                      'validation_rule_reward': validation_rule['mean_reward'],
                      'fresh_rule': {k: v for k, v in rule.items() if k != 'episodes'},
                      'fresh': [{k: v for k, v in r.items() if k != 'fresh'} |
                                {'metrics': {k: v for k, v in r['fresh'].items() if k != 'episodes'}} for r in results],
                      'promotion_recommended': beats_rule and passes_fresh}, indent=2))


if __name__ == '__main__':
    main()
