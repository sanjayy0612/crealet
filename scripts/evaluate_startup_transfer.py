"""Freeze all validation-selected runs before evaluating unused sessions/content."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.benchmark import evaluate
from abr.data import Dataset
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.startup_training import make_env
from scripts.evaluate_request_recovery import compact


def main():
    torch.set_num_threads(1)
    out = Path('runs/startup-ladder')
    protocol = json.loads((out / 'protocol.json').read_text())
    training = [json.loads((out / f'seed-{seed}' / 'metrics.json').read_text()) for seed in protocol['fixed_training']['seeds']]
    frozen = {'selection': 'Highest original validation reward among eligible checkpoints; no transfer outcome selects a checkpoint',
              'seeds': [{'seed': r['seed'], 'path': r['selected_checkpoint'],
                         'sha256': hashlib.sha256(Path(r['selected_checkpoint']).read_bytes()).hexdigest() if r['selected_checkpoint'] else None,
                         'validation_reward': r['selected_validation_reward'], 'rule_reward': r['rule_validation_reward']} for r in training]}
    # Persist exact choices BEFORE opening/scoring new-session transfer datasets.
    (out / 'frozen-selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    safety = ActionSafety(**protocol['safety'])
    policies = {'highest_allowed': lambda obs, cfg: int(np.flatnonzero(allowed_actions(obs, safety))[-1]), 'lowest': lambda obs, cfg: 0}
    checkpoints = {'old_frozen': protocol['checkpoint']} | {f"seed_{r['seed']}": r['selected_checkpoint'] for r in training if r['selected_checkpoint']}
    for name, path in checkpoints.items():
        model, _ = load_model(path)
        policies[name] = DecisionPolicy(model, safety=safety)
    results = []
    for content in ('envivio', 'sintel'):
        data = Dataset.load(f'data/transfer-sessions/{content}.json')
        for name, policy in policies.items():
            report = compact(evaluate(make_env(data, 'test', protocol), baseline='model', policy=policy,
                                      policy_metadata={'kind': name, 'checkpoint': checkpoints.get(name), 'media': content}))
            report.update(content=content, policy=name)
            full = [e for e in report['episodes'] if e['completed']]
            report['summary'] = {'complete': len(full), 'sessions': len(report['episodes']),
                                  'targets_met': sum(e['playback_targets_met'] for e in report['episodes']),
                                  'mean_reward_all_rollouts': float(np.mean([e['reward'] for e in report['episodes']])),
                                  'mean_bitrate_full_playbacks': float(np.mean([e['metrics']['mean_played_bitrate_kbps'] for e in full])) if full else None,
                                  'mean_stalls_full_playbacks': float(np.mean([e['metrics']['rebuffer_s'] for e in full])) if full else None}
            results.append(report)
            print(json.dumps({'content': content, 'policy': name, **report['summary']}), flush=True)
    comparisons = []
    for record in training:
        name = f"seed_{record['seed']}"
        checks = []
        for content in ('envivio', 'sintel'):
            candidate = next((r for r in results if r['content'] == content and r['policy'] == name), None)
            rule = next(r for r in results if r['content'] == content and r['policy'] == 'highest_allowed')
            quality = candidate is not None and all(e['playback_targets_met'] for e in candidate['episodes'])
            no_extra_stalls = candidate is not None and all(a['metrics']['rebuffer_s'] <= b['metrics']['rebuffer_s'] + 1e-9
                                                           for a, b in zip(candidate['episodes'], rule['episodes']))
            beats = quality and no_extra_stalls and candidate['summary']['mean_reward_all_rollouts'] > rule['summary']['mean_reward_all_rollouts'] + 1e-6
            checks.append({'content': content, 'targets_met': bool(quality), 'no_extra_stalls': bool(no_extra_stalls), 'beats_rule': bool(beats)})
        comparisons.append({'seed': record['seed'], 'validation_beats_rule': record['beats_rule_on_validation'],
                            'transfer': checks})
    promotion = len(training) == 3 and all(c['validation_beats_rule'] and all(t['beats_rule'] for t in c['transfer']) for c in comparisons)
    payload = {'frozen_selection': frozen, 'training': training, 'results': results, 'comparisons': comparisons,
               'promotion_recommended': promotion, 'selection_after_transfer': False,
               'limits': ['Four new sessions but same historical ferry route', 'Sintel is a 52.2 s silent trailer, not a full-length live test',
                          '8 kbps 96x54 is intentionally very low visual quality', 'Finite trace failures are retained, not repeated/padded',
                          'Three bounded training seeds, not a broad algorithm benchmark']}
    (out / 'transfer.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'promotion_recommended': promotion, 'comparisons': comparisons}))


if __name__ == '__main__':
    main()
