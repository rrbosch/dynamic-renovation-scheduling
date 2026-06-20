"""Algorithm grouping metadata helpers.

Pure functions (only json/hashlib/copy) that derive human-readable grouping
metadata from a config dict. Kept dependency-free so lightweight consumers such
as ``vis/comparison_dashboard.py`` can import them without pulling in env/agent
modules (numba, numpy, training.trainer, ...).

``experiments.configs`` re-exports these and bakes the results into every
``config.json`` at build time.
"""
from __future__ import annotations

import copy
import hashlib
import json


# Discriminating `extra` keys per direct (fixed-param) heuristic agent_type,
# used by compute_algorithm_label. Optuna-tuned variants are labelled separately
# (by n_tune) under the 'optuna_heuristic' branch.
_DIRECT_HEURISTIC_LABEL_KEYS = {
    'leadtime':       ('lead_epochs', 'repair_lead', 'restrict_lead'),
    'netconcurrency': ('threshold', 'max_concurrent', 'spread_penalty'),
    'holding':        ('threshold', 'max_concurrent', 'defer_window', 'restrict_flow_quantile'),
    'valuedensity':   ('max_concurrent', 'risk_weight', 'degrad_weight', 'threshold'),
    'worstfirst':     ('max_concurrent', 'threshold', 'use_length'),
}


def compute_algorithm_class(config_dict: dict) -> str:
    """Human-readable algorithm family name from a config dict."""
    agent = config_dict.get('agent', {})
    at = agent.get('agent_type', '')
    extra = agent.get('extra', {})
    if at in ('reactive', 'optuna_heuristic'):
        ht = extra.get('heuristic_type', 'reactive')
        return ht  # 'reactive', 'paced', 'reactiveperasset'
    if at == 'paced':
        return 'paced'
    return at  # 'dqn', 'adp', 'actor_critic', 'ppo', 'rollout', 'dcl', etc.


def compute_algorithm_id(config_dict: dict) -> str:
    """Deterministic 12-char hex hash of config minus seed/run_name/output paths."""
    cleaned = copy.deepcopy(config_dict)
    for key in ('seed', 'run_name', 'algorithm_class', 'algorithm_id',
                'algorithm_label', 'instance_id'):
        cleaned.pop(key, None)
    return hashlib.sha256(
        json.dumps(cleaned, sort_keys=True).encode()
    ).hexdigest()[:12]


def compute_algorithm_label(config_dict: dict) -> str:
    """Short human-readable string encoding discriminating hyperparameters."""
    agent = config_dict.get('agent', {})
    at = agent.get('agent_type', '')
    extra = agent.get('extra', {})
    training = config_dict.get('training', {})
    parts: list[str] = []

    if at in ('reactive', 'optuna_heuristic'):
        ht = extra.get('heuristic_type', 'reactive')
        if at == 'optuna_heuristic':
            # Optuna-tuned: label by tuning params, not thresholds
            n_tuning = extra.get('n_tuning_episodes', '')
            if n_tuning:
                parts.append(f"n_tune={n_tuning}")
        elif ht == 'reactive':
            for k in ('threshold', 'repair_threshold', 'restrict_threshold'):
                if k in extra:
                    parts.append(f"{k.replace('_threshold', '').replace('threshold', 'thresh')}={extra[k]}")
        elif ht == 'paced':
            for k in ('threshold', 'pace_threshold'):
                if k in extra:
                    parts.append(f"{k.replace('_threshold', '').replace('threshold', 'thresh')}={extra[k]}")
        elif ht == 'reactiveperasset':
            n_tuning = extra.get('n_tuning_episodes', '')
            if n_tuning:
                parts.append(f"n_tune={n_tuning}")
        if not parts:
            parts.append('default')
    elif at == 'paced':
        for k in ('threshold', 'pace_threshold'):
            if k in extra:
                parts.append(f"{k.replace('_threshold', '').replace('threshold', 'thresh')}={extra[k]}")
        if not parts:
            parts.append('default')
    elif at in ('adp', 'dqn', 'actor_critic'):
        vf = agent.get('value_fn', 'xgboost')
        ag = agent.get('action_gen', 'local_search')
        parts.append(vf)
        parts.append(ag.replace('_', ''))
        trunc = training.get('truncation_mode', '')
        if trunc:
            parts.append(f"trunc={trunc}")
        buf = training.get('buffer_strategy', '')
        if buf and buf != 'fifo':
            parts.append(f"buf={buf}")
    elif at == 'ppo':
        hd = extra.get('hidden_dims', [])
        if hd:
            parts.append(f"h={'x'.join(str(d) for d in hd)}")
    elif at in ('rollout', 'sequential_rollout'):
        nr = extra.get('n_rollouts', '')
        if nr:
            parts.append(f"nroll={nr}")
        # initial_action ('empty' / 'policy') is the only difference between
        # otherwise-identical rollout configs — keep it in the label.
        init = extra.get('initial_action', '')
        if init:
            parts.append(f"init={init}")
    elif at == 'dcl':
        pt = extra.get('policy_type', 'xgboost')
        parts.append(f"pol={pt}")
    elif at in _DIRECT_HEURISTIC_LABEL_KEYS:
        for k in _DIRECT_HEURISTIC_LABEL_KEYS[at]:
            if k in extra:
                parts.append(f"{k.replace('_threshold', '').replace('threshold', 'thresh')}={extra[k]}")
        if not parts:
            parts.append('default')
    else:
        parts.append('default')

    return '_'.join(str(p) for p in parts)
