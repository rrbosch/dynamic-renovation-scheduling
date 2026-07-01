"""Experiment configuration dataclasses with JSON serialization."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import warnings
import numpy as np
from dataclasses import dataclass, field

from env.mdp import EnvConfig, InfraEnv
from env.network import load_sioux_falls, load_amsterdam, NetworkData
from env.tap import make_tap
from training.trainer import TrainingConfig
from agents.base import Agent


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    agent_type: str                      # 'reactive', 'paced', 'adp', 'dqn', 'actor_critic', 'marl'
    value_fn: str = 'xgboost'           # 'xgboost', 'neural', 'ranking'
    action_gen: str = 'local_search'    # 'local_search', 'sequential', 'bdq'
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'agent_type': self.agent_type, 'value_fn': self.value_fn,
                'action_gen': self.action_gen, 'extra': self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> 'AgentConfig':
        return cls(
            agent_type=d['agent_type'],
            value_fn=d.get('value_fn', 'xgboost'),
            action_gen=d.get('action_gen', 'local_search'),
            extra=d.get('extra', {}),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    network: str          # 'sioux_falls' or 'amsterdam'
    tap_backend: str      # 'fast' (default), 'null', 'surrogate'
    seed: int
    run_name: str
    instance: str         # path to instance JSON, relative to project root
    training: dict        # TrainingConfig fields as dict
    agent: AgentConfig

    def to_json(self) -> str:
        d = {
            'network': self.network,
            'tap_backend': self.tap_backend,
            'seed': self.seed,
            'run_name': self.run_name,
            'instance': self.instance,
            'training': self.training,
            'agent': self.agent.to_dict(),
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> 'ExperimentConfig':
        d = json.loads(s)
        return cls(
            network=d['network'],
            tap_backend=d['tap_backend'],
            seed=d['seed'],
            run_name=d['run_name'],
            instance=d['instance'],
            training=d['training'],
            agent=AgentConfig.from_dict(d['agent']),
        )

    @classmethod
    def from_file(cls, path: str) -> 'ExperimentConfig':
        with open(path) as f:
            return cls.from_json(f.read())


# ---------------------------------------------------------------------------
# Algorithm grouping metadata
# ---------------------------------------------------------------------------

# Pure metadata helpers live in a dependency-free module so lightweight
# consumers (e.g. vis/comparison_dashboard.py) can import them without pulling
# in env/agent modules. Re-exported here for existing callers.
from experiments.algorithm_meta import (  # noqa: E402,F401
    compute_algorithm_class,
    compute_algorithm_id,
    compute_algorithm_label,
)


def compute_output_path(config_dict: dict, experiment_name: str, seed: int) -> str:
    """Build nested output path: results/{experiment}/{class}/{label}/seed_{seed}/"""
    alg_class = config_dict.get('algorithm_class', 'unknown')
    alg_label = config_dict.get('algorithm_label', 'default')
    return os.path.join('results', experiment_name, alg_class, alg_label, f'seed_{seed}')


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_experiment(config: ExperimentConfig):
    """
    Construct (InfraEnv, Agent, Trainer) from ExperimentConfig.
    This is the single wiring point; run.py only calls this function.
    All environment parameters are read from the instance file.
    For agent_type='ppo' returns a PPOTrainer instead of Trainer.
    """
    from training.trainer import Trainer
    from utils.logging import RunLogger

    def _arr(v, n):
        """Convert scalar or list to numpy array of length n."""
        if isinstance(v, (int, float)):
            return np.full(n, float(v))
        return np.array(v, dtype=float)

    # 1. Load instance first so n_assets can drive network construction
    with open(config.instance) as f:
        inst = json.load(f)

    # Evaluation tail length (years) is instance-level. Old instances lacking the field
    # fall back to 1x mean expected lifespan = mean(beta/alpha0).
    if 'T_tail' in inst:
        _T_tail = float(inst['T_tail'])
    else:
        _beta = np.asarray(inst['beta'], dtype=float)
        _alpha0 = np.asarray(inst['alpha0'], dtype=float)
        _T_tail = float(np.mean(_beta / _alpha0))

    # 2. Network — pass n_assets (and optional explicit high-synergy link
    #    selection) from instance. Each asset is one bidirectional link.
    if config.network == 'sioux_falls':
        network = load_sioux_falls(
            n_assets=inst['n_assets'], asset_links=inst.get('asset_links'))
    elif config.network == 'amsterdam':
        network = load_amsterdam()
    else:
        raise ValueError(f"Unknown network: {config.network!r}")

    n_assets = network.n_assets

    if inst.get('network') and inst['network'] != config.network:
        warnings.warn(
            f"Instance network={inst['network']!r} != config network={config.network!r}"
        )

    # 3. EnvConfig — all parameters from instance
    d_init = np.array(inst['d_init'], dtype=float) if inst.get('d_init') is not None else None

    env_config = EnvConfig(
        n_assets=n_assets,
        dt=inst['dt'],
        T=round(inst['years'] / inst['dt']),
        gamma=inst['gamma'] ** inst['dt'],
        d_fail=inst['d_fail'],
        eta_ren=inst['eta_ren'],
        eta_load=inst['eta_load'],
        restrict_degrad_multiplier=float(inst.get('restrict_degrad_multiplier', 0.5)),
        mu_h=_arr(inst['mu_h'], n_assets),
        sigma_h=_arr(inst['sigma_h'], n_assets),
        delta_repair=inst['delta_repair'],
        alpha0=_arr(inst['alpha0'], n_assets),
        beta=_arr(inst['beta'], n_assets),
        c_ren=_arr(inst['c_ren'], n_assets),
        c_rep=_arr(inst['c_rep'], n_assets),
        asset_lengths_m=_arr(inst['asset_lengths_m'], n_assets),
        vot=float(inst.get('vot', 10.76)),
        traffic_cost_factor=float(inst.get('traffic_cost_factor', 1.0)),
        risk_base=float(inst.get('risk_base', 10_000.0)),
        d_init=d_init,
        allow_repair=bool(inst.get('allow_repair', True)),
        allow_restrict=bool(inst.get('allow_restrict', True)),
    )

    # 4. TAP
    if config.tap_backend == 'surrogate':
        from env.surrogate import SurrogateTAP
        tap = SurrogateTAP(network)
    else:
        tap = make_tap(network, backend=config.tap_backend)

    # 5. Environment
    env = InfraEnv(network, tap, env_config, rng_seed=config.seed)
    env._tap_backend_name = config.tap_backend  # stored for parallel worker reconstruction

    # 6. Training config dict (needed by agent builder for n_workers)
    tc_dict = config.training

    # 7. Agent
    _n_workers = int(tc_dict.get('n_workers', 1))
    agent = _build_agent(config.agent, env, config.seed, n_workers=_n_workers)
    # ADP 'policy' action-search init: seed the local search from the same heuristic
    # used to warmstart the buffer (training.warmstart). Built here because _build_agent
    # only sees the `agent` block, not training.warmstart.
    from agents.dqn import ADPAgent
    if isinstance(agent, ADPAgent) and agent.init_action_mode == 'policy':
        _ws_cfg = tc_dict.get('warmstart')
        if _ws_cfg is None:
            raise ValueError(
                "agent.extra.init_action='policy' requires a training.warmstart "
                "heuristic config (the search seed reuses the warmstart heuristic)."
            )
        agent.warmstart_policy = _build_agent(AgentConfig.from_dict(_ws_cfg), env, config.seed)
    # Stable config hash (excludes 'seed' so reseeded runs don't auto-resume mismatched checkpoints)
    _cfg_dict_no_seed = {k: v for k, v in json.loads(config.to_json()).items() if k != 'seed'}
    _config_hash = hashlib.sha256(
        json.dumps(_cfg_dict_no_seed, sort_keys=True).encode()
    ).hexdigest()[:16]
    # Handle legacy 'bootstrap_truncation' key -> new 'truncation_mode'
    if 'truncation_mode' in tc_dict:
        _trunc_mode = tc_dict['truncation_mode']
        # Normalize legacy value
        if _trunc_mode == 'bootstrapped':
            _trunc_mode = 'bootstrap'
    elif 'bootstrap_truncation' in tc_dict:
        _trunc_mode = 'bootstrap' if tc_dict['bootstrap_truncation'] else 'none'
    else:
        _trunc_mode = 'bootstrap'

    training_config = TrainingConfig(
        time_budget=float(tc_dict.get('time_budget', 3600.0)),
        n_episodes=int(tc_dict.get('n_episodes', 10_000_000)),
        eval_interval=tc_dict.get('eval_interval', 50),
        eval_interval_seconds=float(tc_dict.get('eval_interval_seconds', 0.0)),
        update_interval=tc_dict.get('update_interval', 10),
        truncation_mode=_trunc_mode,
        T_tail=_T_tail,
        buffer_capacity=tc_dict.get('buffer_capacity', 200_000),
        buffer_strategy=tc_dict.get('buffer_strategy', 'fifo'),
        n_eval_episodes=tc_dict.get('n_eval_episodes', 10),
        n_warmstart_states=int(tc_dict.get('n_warmstart_states', 0)),
        warmstart_agent_config=tc_dict.get('warmstart', None),
        checkpoint_interval=int(tc_dict.get('checkpoint_interval', 0)),
        checkpoint_interval_seconds=float(tc_dict.get('checkpoint_interval_seconds', 1800.0)),
        config_hash=_config_hash,
        n_workers=int(tc_dict.get('n_workers', 1)),
    )

    # 8. Logger
    cfg_dict = json.loads(config.to_json())
    cfg_dict['instance_id'] = inst.get('instance_id')   # None for old instances without ID
    cfg_dict['algorithm_class'] = compute_algorithm_class(cfg_dict)
    cfg_dict['algorithm_id'] = compute_algorithm_id(cfg_dict)
    cfg_dict['algorithm_label'] = compute_algorithm_label(cfg_dict)
    run_dir = os.path.join('results', config.run_name)
    logger = RunLogger(run_dir)
    logger.write_config(cfg_dict)

    # 9. Trainer — special cases first, then general Trainer
    if config.agent.agent_type == 'optuna_heuristic':
        from agents.heuristics import ReactiveAgent, PacedAgent
        from experiments.optuna_heuristic_search import OptunaHeuristicTrainer
        extra = config.agent.extra or {}
        heuristic_type = extra.get('heuristic_type', 'reactive')
        param_space = extra.get('param_space', {})

        _REACTIVE_DEFAULT_SPACE = {
            'threshold':          {'type': 'float', 'low': 0.3,  'high': 1.0},
            'repair_threshold':   {'type': 'float', 'low': 0.1,  'high': 0.95},
            'restrict_threshold': {'type': 'float', 'low': 0.1,  'high': 0.95},
        }
        _PACED_DEFAULT_SPACE = {
            'threshold':      {'type': 'float', 'low': 0.3, 'high': 1.0},
            'pace_threshold': {'type': 'float', 'low': 0.0, 'high': 1.0},
        }

        if heuristic_type == 'reactive':
            if not param_space:
                param_space = _REACTIVE_DEFAULT_SPACE
            agent = ReactiveAgent(threshold=1.0, env_config=env.config)
            def agent_factory(params, _cfg=env.config):
                return ReactiveAgent(
                    threshold=params['threshold'],
                    env_config=_cfg,
                    repair_threshold=params.get('repair_threshold'),
                    restrict_threshold=params.get('restrict_threshold'),
                )
        elif heuristic_type == 'paced':
            if not param_space:
                param_space = _PACED_DEFAULT_SPACE
            agent = PacedAgent(threshold=1.0, env_config=env.config)
            def agent_factory(params, _cfg=env.config):
                return PacedAgent(
                    threshold=params['threshold'],
                    env_config=_cfg,
                    pace_threshold=params.get('pace_threshold', 0.5),
                )
        elif heuristic_type == 'reactiveperasset':
            from agents.heuristics import PerAssetReactiveAgent
            n = env.config.n_assets
            if not param_space:
                for i in range(n):
                    param_space[f'repair_threshold_{i}']   = {'type': 'float', 'low': 0.0, 'high': 1.1}
                    param_space[f'restrict_threshold_{i}'] = {'type': 'float', 'low': 0.0, 'high': 1.1}
                    param_space[f'renovate_threshold_{i}'] = {'type': 'float', 'low': 0.0, 'high': 1.0}
            agent = PerAssetReactiveAgent(
                thresholds=np.ones((n, 3), dtype=float),
                env_config=env.config,
            )
            def agent_factory(params, _cfg=env.config, _n=n):
                thr = _perasset_thresholds_from_extra(params, _n)  # (N, 3)
                return PerAssetReactiveAgent(thresholds=thr, env_config=_cfg)
        elif heuristic_type == 'leadtime':
            from agents.heuristics import LeadTimeAgent
            _T = env.config.T
            if not param_space:
                param_space = {
                    'lead_epochs':   {'type': 'float', 'low': 0.0, 'high': 0.25 * _T},
                    'repair_lead':   {'type': 'float', 'low': 0.0, 'high': 0.25 * _T},
                    'restrict_lead': {'type': 'float', 'low': 0.0, 'high': 0.25 * _T},
                }
            agent = LeadTimeAgent(lead_epochs=0.0, env_config=env.config)
            def agent_factory(params, _cfg=env.config):
                return LeadTimeAgent(
                    lead_epochs=params['lead_epochs'],
                    env_config=_cfg,
                    repair_lead=params.get('repair_lead'),
                    restrict_lead=params.get('restrict_lead'),
                )
        elif heuristic_type == 'netconcurrency':
            from agents.heuristics import NetConcurrencyAgent
            _flow = _asset_flow_proxy(env)
            if not param_space:
                param_space = {
                    'threshold':      {'type': 'float', 'low': 0.3, 'high': 1.0},
                    'max_concurrent': {'type': 'int',   'low': 1,   'high': env.config.n_assets},
                    'spread_penalty': {'type': 'float', 'low': 0.0, 'high': 1.0},
                }
            agent = NetConcurrencyAgent(threshold=1.0, env_config=env.config, asset_flow=_flow)
            def agent_factory(params, _cfg=env.config, _flow=_flow):
                return NetConcurrencyAgent(
                    threshold=params['threshold'],
                    env_config=_cfg,
                    max_concurrent=params.get('max_concurrent', 3),
                    spread_penalty=params.get('spread_penalty', 0.0),
                    asset_flow=_flow,
                )
        elif heuristic_type == 'holding':
            from agents.heuristics import HoldingAgent
            _flow = _asset_flow_proxy(env)
            _T = env.config.T
            if not param_space:
                param_space = {
                    'threshold':              {'type': 'float', 'low': 0.3, 'high': 1.0},
                    'max_concurrent':         {'type': 'int',   'low': 1,   'high': env.config.n_assets},
                    'defer_window':           {'type': 'float', 'low': 0.0, 'high': 0.25 * _T},
                    'restrict_flow_quantile': {'type': 'float', 'low': 0.0, 'high': 1.0},
                }
            agent = HoldingAgent(threshold=1.0, env_config=env.config, asset_flow=_flow)
            def agent_factory(params, _cfg=env.config, _flow=_flow):
                return HoldingAgent(
                    threshold=params['threshold'],
                    env_config=_cfg,
                    max_concurrent=params.get('max_concurrent', 3),
                    defer_window=params.get('defer_window', 4.0),
                    restrict_flow_quantile=params.get('restrict_flow_quantile', 0.5),
                    asset_flow=_flow,
                )
        elif heuristic_type == 'valuedensity':
            from agents.heuristics import ValueDensityAgent
            if not param_space:
                param_space = {
                    'max_concurrent': {'type': 'int',   'low': 1,   'high': env.config.n_assets},
                    'risk_weight':    {'type': 'float', 'low': 0.0, 'high': 2.0},
                    'degrad_weight':  {'type': 'float', 'low': 0.0, 'high': 2.0},
                    'threshold':      {'type': 'float', 'low': 0.0, 'high': 1.0},
                }
            agent = ValueDensityAgent(env_config=env.config)
            def agent_factory(params, _cfg=env.config):
                return ValueDensityAgent(
                    env_config=_cfg,
                    max_concurrent=params.get('max_concurrent', 3),
                    risk_weight=params.get('risk_weight', 1.0),
                    degrad_weight=params.get('degrad_weight', 1.0),
                    threshold=params.get('threshold', 0.0),
                )
        elif heuristic_type == 'worstfirst':
            from agents.heuristics import WorstFirstAgent
            if not param_space:
                param_space = {
                    'max_concurrent': {'type': 'int',         'low': 1, 'high': env.config.n_assets},
                    'threshold':      {'type': 'float',       'low': 0.3, 'high': 1.0},
                    'use_length':     {'type': 'categorical', 'choices': [True, False]},
                }
            agent = WorstFirstAgent(env_config=env.config)
            def agent_factory(params, _cfg=env.config):
                return WorstFirstAgent(
                    env_config=_cfg,
                    max_concurrent=params.get('max_concurrent', 3),
                    threshold=params.get('threshold', 0.5),
                    use_length=params.get('use_length', True),
                )
        else:
            raise ValueError(f"Unknown heuristic_type: {heuristic_type!r}")

        trainer = OptunaHeuristicTrainer(
            env=env, agent=agent, logger=logger,
            base_seed=config.seed,
            time_budget=int(tc_dict.get('time_budget', 86400)),
            n_tuning_episodes=int(extra.get('n_tuning_episodes', 30)),
            early_stopping_seconds=int(tc_dict.get('early_stopping_seconds', 3600)),
            param_space=param_space,
            agent_factory=agent_factory,
            checkpoint_interval_seconds=float(tc_dict.get('checkpoint_interval_seconds', 1800.0)),
            config_hash=_config_hash,
            T_tail=_T_tail,
        )
        return env, agent, trainer

    if config.agent.agent_type == 'ppo':
        from training.ppo_trainer import PPOConfig, PPOTrainer
        from env.tap import NullTAP
        import dataclasses

        ppo_config = PPOConfig(
            n_episodes=int(tc_dict.get('n_episodes', 1000)),
            eval_interval=int(tc_dict.get('eval_interval', 50)),
            n_eval_episodes=int(tc_dict.get('n_eval_episodes', 10)),
            time_budget=float(tc_dict.get('time_budget', 3600.0)),
            T_tail=_T_tail,
            curriculum_phase0_episodes=int(tc_dict.get('curriculum_phase0_episodes', 0)),
            curriculum_phase1_plateau_window=int(tc_dict.get('curriculum_phase1_plateau_window', 5)),
            curriculum_phase1_plateau_tol=float(tc_dict.get('curriculum_phase1_plateau_tol', 0.01)),
            curriculum_reset_critic=bool(tc_dict.get('curriculum_reset_critic', False)),
        )

        curriculum_env = None
        heuristic_agent = None
        if ppo_config.curriculum_phase0_episodes > 0:
            simplified_config = dataclasses.replace(env_config, traffic_cost_factor=0.0)
            curriculum_env = InfraEnv(network, NullTAP(), simplified_config,
                                      rng_seed=config.seed + 9999)
            # Phase-0 imitation target: configurable via training.curriculum_heuristic
            # ({"agent_type": ..., "extra": {...}}, same schema as warmstart). Built on
            # the simplified curriculum env. Falls back to a hardcoded reactive heuristic
            # for configs that don't specify one.
            _ch_cfg = tc_dict.get('curriculum_heuristic')
            if _ch_cfg is not None:
                heuristic_agent = _build_agent(
                    AgentConfig.from_dict(_ch_cfg),
                    curriculum_env, config.seed,
                )
            else:
                from agents.heuristics import ReactiveAgent
                heuristic_agent = ReactiveAgent(
                    threshold=0.99,
                    env_config=simplified_config,
                    repair_threshold=0.9,
                    restrict_threshold=0.7,
                )

        trainer = PPOTrainer(agent, env, ppo_config, logger,
                             curriculum_env=curriculum_env,
                             heuristic_agent=heuristic_agent)
    elif config.agent.agent_type == 'dcl':
        from training.dcl_trainer import DCLConfig, DCLTrainer
        dx = config.agent.extra or {}
        dcl_config = DCLConfig(
            n_rounds=int(dx.get('n_rounds', 3)),
            samples_per_round=int(dx.get('samples_per_round', 2000)),
            collect_steps=int(dx.get('collect_steps', 0)),
            rollout_horizon=dx.get('rollout_horizon', None),
            n_rollouts=int(dx.get('n_rollouts', 30)),
            rollout_selection=dx.get('rollout_selection', 'fixed'),
            sh_budget_per_arm=dx.get('sh_budget_per_arm', None),
            p_threshold=float(dx.get('p_threshold', 0.02)),
            min_rollouts=int(dx.get('min_rollouts', 20)),
            max_rollouts=int(dx.get('max_rollouts', 100)),
            rollout_batch=int(dx.get('rollout_batch', 5)),
            action_threshold=float(dx.get('action_threshold', 0.0)),
            initial_action=dx.get('initial_action', 'policy'),
            value_fn_kind=dx.get('value_fn', 'xgboost'),
            finite_horizon=bool(dx.get('finite_horizon', True)),
            eval_interval=int(tc_dict.get('eval_interval', 1)),
            n_eval_episodes=int(tc_dict.get('n_eval_episodes', 10)),
            time_budget=float(tc_dict.get('time_budget', 86400.0)),
            T_tail=_T_tail,
            seed=config.seed,
            config_hash=_config_hash,
            n_workers=int(tc_dict.get('n_workers', 1)),
            tap_backend=config.tap_backend,
        )
        trainer = DCLTrainer(agent, env, dcl_config, logger)
    else:
        trainer = Trainer(agent, env, training_config, logger, seed=config.seed,
                          tap_backend=config.tap_backend)

    return env, agent, trainer


def _check_extra_keys(agent_type: str, extra: dict, allowed: set) -> None:
    """Fail loudly on unrecognised keys in an agent's `extra` block.

    Config values are read via `extra.get(key, default)`, so a misspelled or
    renamed key is otherwise *silently ignored* and the agent falls back to its
    default — e.g. `max_steps` instead of `rollout_horizon` once silently
    disabled the rollout lookahead and made the agent ~10x worse than its base
    policy. Validating the keys up front turns that class of mistake into an
    immediate, explicit error instead of a corrupted experiment.
    """
    unknown = set(extra) - set(allowed)
    if not unknown:
        return
    import difflib
    lines = []
    for key in sorted(unknown):
        hint = difflib.get_close_matches(key, allowed, n=1)
        suffix = f" (did you mean {hint[0]!r}?)" if hint else ""
        lines.append(f"  - {key!r}{suffix}")
    raise ValueError(
        f"Unknown key(s) in agent.extra for agent_type={agent_type!r}:\n"
        + "\n".join(lines)
        + f"\nValid keys: {sorted(allowed)}"
    )


# Recognised `extra` keys per agent_type. Used by _check_extra_keys to reject
# typos/renamed keys instead of silently falling back to defaults. Extend this
# when adding a new `extra.get(...)` read for one of these agent types.
_ROLLOUT_EXTRA_KEYS = {
    'n_rollouts', 'rollout_horizon', 'rollout_seed', 'action_threshold',
    'initial_action', 'rollout_selection', 'p_threshold', 'min_rollouts',
    'max_rollouts', 'rollout_batch', 'rollout_policy',
}
_DCL_EXTRA_KEYS = {
    # decomposition + classifier
    'action_search', 'policy_type', 'value_fn', 'finite_horizon',
    'use_global_context', 'hidden_dims', 'policy_lr', 'policy_epochs',
    'policy_batch_size', 'clf_kwargs', 'class_weight', 'heuristic_policy',
    # approximate-policy-iteration loop
    'n_rounds', 'samples_per_round', 'collect_steps',
    # rollout-improvement oracle
    'rollout_horizon', 'n_rollouts', 'rollout_selection', 'sh_budget_per_arm',
    'p_threshold', 'min_rollouts', 'max_rollouts', 'rollout_batch',
    'action_threshold', 'initial_action',
}
_REACTIVE_EXTRA_KEYS = {'threshold', 'repair_threshold', 'restrict_threshold'}
_PACED_EXTRA_KEYS = {'threshold', 'pace_threshold'}
_LEADTIME_EXTRA_KEYS = {'lead_epochs', 'repair_lead', 'restrict_lead'}
_NETCONCURRENCY_EXTRA_KEYS = {'threshold', 'max_concurrent', 'spread_penalty'}
_HOLDING_EXTRA_KEYS = {'threshold', 'max_concurrent', 'defer_window',
                       'restrict_flow_quantile'}
_VALUEDENSITY_EXTRA_KEYS = {'max_concurrent', 'risk_weight', 'degrad_weight', 'threshold'}
_WORSTFIRST_EXTRA_KEYS = {'max_concurrent', 'threshold', 'use_length'}
_CLAIRVOYANT_EXTRA_KEYS = {'use_dp', 'warm_start', 'max_sweeps', 'time_budget_s',
                          'n_grid', 'nf_max'}
_EXPLORE_FLIP_EXTRA_KEYS = {'base', 'p_base', 'p_high', 'd_ref', 'act_bias', 'renovate_bias'}
_FLIP_SEED_SALT = 0x666C6970  # 'flip' — keep the flip rng stream independent
_MIXTURE_EXTRA_KEYS = {'policies'}
# Allowed keys for one entry of a mixture's `policies` list.
_MIXTURE_ENTRY_KEYS = {'weight', 'agent_type', 'extra', 'value_fn', 'action_gen'}
_MIXTURE_SAMPLED_ENTRY_KEYS = {'weight', 'agent_type', 'renovate_range',
                               'repair_gap_range', 'restrict_gap_range'}
_MIXTURE_SEED_SALT = 0x6D6978  # 'mix' — keep the mixture rng independent of the flip rng


def _asset_flow_proxy(env: InfraEnv) -> np.ndarray:
    """Static per-asset nominal-capacity TAP flow, shape (N,).

    Solved once (not per step) so network-aware heuristics (NetConcurrencyAgent,
    HoldingAgent) can rank/spread renovations without calling TAP inside act().
    """
    flows = np.asarray(env.tap_fn.solve(env.network.nominal_capacities))
    return flows[env.network.asset_indices]


def _perasset_extra_keys(n: int) -> set:
    """Allowed `extra` keys for a `reactiveperasset` heuristic: 3 thresholds per asset."""
    return {f'{p}_{i}' for i in range(n)
            for p in ('repair_threshold', 'restrict_threshold', 'renovate_threshold')}


def _perasset_thresholds_from_extra(extra: dict, n: int) -> np.ndarray:
    """Map flat per-asset threshold keys into the (N, 3) array PerAssetReactiveAgent expects.

    Columns are [repair, restrict, renovate] (see PerAssetReactiveAgent docstring).
    Used both by _build_agent (warmstart / rollout base policy) and the optuna agent_factory.
    """
    return np.column_stack([
        [extra[f'repair_threshold_{i}']   for i in range(n)],
        [extra[f'restrict_threshold_{i}'] for i in range(n)],
        [extra[f'renovate_threshold_{i}'] for i in range(n)],
    ])  # shape (N, 3)


def _mixture_policy_factory(p: dict, env: InfraEnv, seed: int):
    """Build a ``factory(rng) -> Agent`` for one entry of a mixture's ``policies`` list.

    ``agent_type == 'reactive_sampled'`` is a pseudo-type valid ONLY inside a mixture:
    each episode it samples thresholds ``res <= rep <= ren`` (renovate from
    ``renovate_range``; repair/restrict offset below it by ``repair_gap_range`` /
    ``restrict_gap_range``) and returns a fresh ``ReactiveAgent``. Any other agent_type
    is a normal spec, built once (the factory ignores ``rng`` and returns it)."""
    pt = p.get('agent_type')
    if pt == 'reactive_sampled':
        unknown = set(p) - _MIXTURE_SAMPLED_ENTRY_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in mixture 'reactive_sampled' entry: {sorted(unknown)}; "
                f"valid: {sorted(_MIXTURE_SAMPLED_ENTRY_KEYS)}")
        from agents.heuristics import ReactiveAgent
        cfg = env.config
        ren_r = tuple(p.get('renovate_range', (0.55, 1.0)))
        rep_g = tuple(p.get('repair_gap_range', (0.05, 0.30)))
        res_g = tuple(p.get('restrict_gap_range', (0.05, 0.30)))

        def factory(rng, cfg=cfg, ren_r=ren_r, rep_g=rep_g, res_g=res_g):
            ren = float(rng.uniform(*ren_r))
            rep = max(0.0, ren - float(rng.uniform(*rep_g)))
            res = max(0.0, rep - float(rng.uniform(*res_g)))
            return ReactiveAgent(ren, cfg, repair_threshold=rep, restrict_threshold=res)
        return factory

    unknown = set(p) - _MIXTURE_ENTRY_KEYS
    if unknown:
        raise ValueError(
            f"Unknown key(s) in mixture policy entry (agent_type={pt!r}): {sorted(unknown)}; "
            f"valid: {sorted(_MIXTURE_ENTRY_KEYS)}")
    sub = _build_agent(AgentConfig.from_dict(p), env, seed)   # 'weight' ignored by from_dict
    return lambda rng, sub=sub: sub


def _build_agent(agent_config: AgentConfig, env: InfraEnv, seed: int, n_workers: int = 1) -> Agent:
    at = agent_config.agent_type
    extra = agent_config.extra

    if at in ('reactive', 'optuna_heuristic'):
        # 'optuna_heuristic' carries tuning-only keys (param_space, etc.) consumed
        # in build_experiment, so only validate the plain reactive case.
        if at == 'reactive':
            _check_extra_keys(at, extra, _REACTIVE_EXTRA_KEYS)
        from agents.heuristics import ReactiveAgent
        return ReactiveAgent(
            threshold=extra.get('threshold', 0.7),
            env_config=env.config,
            repair_threshold=extra.get('repair_threshold', None),
            restrict_threshold=extra.get('restrict_threshold', None),
        )

    if at == 'paced':
        _check_extra_keys(at, extra, _PACED_EXTRA_KEYS)
        from agents.heuristics import PacedAgent
        return PacedAgent(threshold=extra.get('threshold', 0.7), pace_threshold=extra.get('pace_threshold'),
                          env_config=env.config)

    if at == 'reactiveperasset':
        from agents.heuristics import PerAssetReactiveAgent
        n = env.config.n_assets
        _check_extra_keys(at, extra, _perasset_extra_keys(n))
        thr = _perasset_thresholds_from_extra(extra, n)
        return PerAssetReactiveAgent(thresholds=thr, env_config=env.config)

    if at == 'leadtime':
        _check_extra_keys(at, extra, _LEADTIME_EXTRA_KEYS)
        from agents.heuristics import LeadTimeAgent
        return LeadTimeAgent(
            lead_epochs=extra.get('lead_epochs', 4.0),
            env_config=env.config,
            repair_lead=extra.get('repair_lead', None),
            restrict_lead=extra.get('restrict_lead', None),
        )

    if at == 'netconcurrency':
        _check_extra_keys(at, extra, _NETCONCURRENCY_EXTRA_KEYS)
        from agents.heuristics import NetConcurrencyAgent
        return NetConcurrencyAgent(
            threshold=extra.get('threshold', 0.7),
            env_config=env.config,
            max_concurrent=extra.get('max_concurrent', 3),
            spread_penalty=extra.get('spread_penalty', 0.0),
            asset_flow=_asset_flow_proxy(env),
        )

    if at == 'holding':
        _check_extra_keys(at, extra, _HOLDING_EXTRA_KEYS)
        from agents.heuristics import HoldingAgent
        return HoldingAgent(
            threshold=extra.get('threshold', 0.7),
            env_config=env.config,
            max_concurrent=extra.get('max_concurrent', 3),
            defer_window=extra.get('defer_window', 4.0),
            restrict_flow_quantile=extra.get('restrict_flow_quantile', 0.5),
            asset_flow=_asset_flow_proxy(env),
        )

    if at == 'valuedensity':
        _check_extra_keys(at, extra, _VALUEDENSITY_EXTRA_KEYS)
        from agents.heuristics import ValueDensityAgent
        return ValueDensityAgent(
            env_config=env.config,
            max_concurrent=extra.get('max_concurrent', 3),
            risk_weight=extra.get('risk_weight', 1.0),
            degrad_weight=extra.get('degrad_weight', 1.0),
            threshold=extra.get('threshold', 0.0),
        )

    if at == 'worstfirst':
        _check_extra_keys(at, extra, _WORSTFIRST_EXTRA_KEYS)
        from agents.heuristics import WorstFirstAgent
        return WorstFirstAgent(
            env_config=env.config,
            max_concurrent=extra.get('max_concurrent', 3),
            threshold=extra.get('threshold', 0.5),
            use_length=extra.get('use_length', True),
        )

    if at in ('adp', 'dqn', 'actor_critic'):
        finite_horizon = extra.get('finite_horizon', True)
        vf = _build_value_fn(agent_config.value_fn, finite_horizon=finite_horizon)
        # (c) Advantage target: subtract a per-epoch baseline b(t) from the ADP
        # cost-to-go target so the value fn resolves the act-vs-wait signal
        # instead of the dominant time trend. predict adds b(t) back, so the
        # action-generator Q reconstruction stays a valid cost estimate (XGB+NN).
        if extra.get('advantage_baseline', False):
            vf.set_baseline_enabled(True)
        ag = _build_action_gen(agent_config.action_gen,
                               log_q_breakdown=extra.get('log_q_breakdown', False))

        from training.trainer import TrainingConfig  # reuse default

        if at == 'adp':
            from agents.dqn import ADPAgent
            # init_action: 'empty' (search from do-nothing) | 'policy' (search from
            # the warmstart heuristic's action). The warmstart_policy itself is
            # attached in build_experiment, which can see training.warmstart.
            return ADPAgent(vf, ag, env, TrainingConfig(), finite_horizon=finite_horizon,
                            init_action_mode=extra.get('init_action', 'empty'),
                            n_step=int(extra.get('n_step', 0)))

        if at == 'dqn':
            from agents.dqn import DQNAgent
            return DQNAgent(vf, ag, env, TrainingConfig(), finite_horizon=finite_horizon)

        # actor_critic — build an ADPAgent as the value-based critic
        from agents.dqn import ADPAgent
        adp = ADPAgent(vf, ag, env, TrainingConfig(), finite_horizon=finite_horizon)
        from agents.fn.policy import PolicyNetwork
        from agents.actor_critic import ActorCriticAgent
        input_dim = (5 * env.config.n_assets + 1) if finite_horizon else 5 * env.config.n_assets
        pnet = PolicyNetwork(input_dim=input_dim, n_assets=env.config.n_assets,
                             hidden_dims=extra.get('hidden_dims', [256, 256]),
                             T=env.config.T, finite_horizon=finite_horizon)
        return ActorCriticAgent(adp, pnet,
                                patience=extra.get('patience', 20))

    if at == 'ppo':
        from agents.ppo import PPOAgent
        finite_horizon = extra.get('finite_horizon', True)
        input_dim = (5 * env.config.n_assets + 1) if finite_horizon else 5 * env.config.n_assets
        ppo_kwargs = extra.get('ppo_kwargs', {})
        return PPOAgent(
            env=env,
            input_dim=input_dim,
            n_assets=env.config.n_assets,
            hidden_dims=extra.get('hidden_dims', [256, 256]),
            finite_horizon=finite_horizon,
            **ppo_kwargs,
        )

    if at == 'marl':
        from agents.marl import CTDEAgent
        return CTDEAgent()

    if at in ('rollout', 'sequential_rollout'):
        _check_extra_keys(at, extra, _ROLLOUT_EXTRA_KEYS)
        if at == 'rollout':
            from agents.rollout import MonteCarloRolloutAgent as RolloutCls
        else:
            from agents.rollout import SequentialMCRolloutAgent as RolloutCls
        rollout_policy_cfg = extra.get('rollout_policy', {'agent_type': 'reactive', 'threshold': 0.7})
        rollout_policy = _build_agent(
            AgentConfig(
                agent_type=rollout_policy_cfg['agent_type'],
                extra={k: v for k, v in rollout_policy_cfg.items() if k != 'agent_type'},
            ),
            env, seed,
        )
        return RolloutCls(
            rollout_policy=rollout_policy,
            env=env,
            n_rollouts=extra.get('n_rollouts', 30),
            rollout_horizon=extra.get('rollout_horizon', None),
            seed=extra.get('rollout_seed', seed),
            action_threshold=extra.get('action_threshold', 0.5),
            initial_action=extra.get('initial_action', 'policy'),
            # Adaptive (sequential Wilcoxon) rollout budgeting is the default
            # (p=0.02, min=20, max=100 — the chosen Pareto operating point from
            # the instance_10p sweep: ~sub-1% mean cost regret at ~62% rollouts
            # saved). Set rollout_selection='fixed' to restore the legacy
            # fixed-n_rollouts behaviour. See docs/adaptive_rollout_literature.md.
            selection=extra.get('rollout_selection', 'adaptive'),
            p_threshold=extra.get('p_threshold', 0.02),
            min_rollouts=extra.get('min_rollouts', 20),
            max_rollouts=extra.get('max_rollouts', 100),
            rollout_batch=extra.get('rollout_batch', 5),
        )

    if at == 'dcl':
        _check_extra_keys(at, extra, _DCL_EXTRA_KEYS)
        from agents.dcl import DCLAgent, build_decomposition

        action_search = extra.get('action_search', 'sequential')
        decomposition = build_decomposition(
            action_search,
            env,
            estimator_kind=extra.get('policy_type', 'xgboost'),
            use_global_context=extra.get('use_global_context', True),
            hidden_dims=extra.get('hidden_dims', [256, 256]),
            lr=extra.get('policy_lr', 1e-3),
            epochs=extra.get('policy_epochs', 30),
            batch_size=extra.get('policy_batch_size', 256),
            clf_kwargs=extra.get('clf_kwargs', None),
            class_weight=extra.get('class_weight', None),
            finite_horizon=extra.get('finite_horizon', True),
        )
        heuristic_cfg = extra.get('heuristic_policy',
                                  {'agent_type': 'reactive', 'extra': {'threshold': 0.8}})
        heuristic = _build_agent(AgentConfig.from_dict(heuristic_cfg), env, seed)
        return DCLAgent(policy=decomposition, base_heuristic=heuristic,
                        env=env, action_search=action_search)

    if at == 'clairvoyant':
        # Perfect-information (wait-and-see) baseline. Solves a per-seed
        # deterministic plan from the episode's exact replayed noise; a lower
        # bound on any non-anticipative policy's cost. `warm_start` is an optional
        # {agent_type, extra} heuristic spec used as a local-search start.
        _check_extra_keys(at, extra, _CLAIRVOYANT_EXTRA_KEYS)
        from agents.clairvoyant import ClairvoyantAgent
        _tb = extra.get('time_budget_s', None)
        _ms = extra.get('max_sweeps', None)          # None ⇒ run BCD to its local optimum
        return ClairvoyantAgent(
            use_dp=bool(extra.get('use_dp', True)),
            warm_start_spec=extra.get('warm_start', None),
            max_sweeps=(int(_ms) if _ms is not None else None),
            time_budget_s=(float(_tb) if _tb is not None else None),
            n_grid=int(extra.get('n_grid', 128)),
            nf_max=int(extra.get('nf_max', 24)),
            seed=seed,
        )

    if at == 'donothing':
        _check_extra_keys(at, extra, set())
        from agents.heuristics import DoNothingAgent
        return DoNothingAgent(env.config)

    if at == 'explore_flip':
        # Behavior wrapper: base policy + the old warmstart "bit-flip" exploration.
        # Composes as a mixture mode (flip-wrap a reactive mode, leave do-nothing pure).
        _check_extra_keys(at, extra, _EXPLORE_FLIP_EXTRA_KEYS)
        base_spec = extra.get('base')
        if not base_spec:
            raise ValueError("agent_type='explore_flip' requires a 'base' policy spec in agent.extra.")
        from agents.heuristics import FlipWrapperAgent
        base = _build_agent(AgentConfig.from_dict(base_spec), env, seed)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), _FLIP_SEED_SALT]))
        return FlipWrapperAgent(
            base, env, rng,
            p_base=float(extra.get('p_base', 1.0 / 120)),
            p_high=float(extra.get('p_high', 0.50)),
            d_ref=float(extra.get('d_ref', 0.5)),
            act_bias=float(extra.get('act_bias', 0.9)),
            renovate_bias=float(extra.get('renovate_bias', 0.7)),
        )

    if at == 'mixture':
        # Per-episode weighted mixture of behavior policies (warmstart diversification).
        # Flips are now per-policy: to add acting-bias exploration to a mode, make that
        # mode an 'explore_flip' wrapping a base policy — and leave the 'donothing' mode
        # un-wrapped so failure-coverage episodes stay pure.
        _check_extra_keys(at, extra, _MIXTURE_EXTRA_KEYS)
        from agents.mixture import MixtureAgent
        policies = extra.get('policies', [])
        if not policies:
            raise ValueError("agent_type='mixture' requires a non-empty 'policies' list in agent.extra.")
        modes = [(float(p.get('weight', 1.0)), _mixture_policy_factory(p, env, seed))
                 for p in policies]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), _MIXTURE_SEED_SALT]))
        return MixtureAgent(modes, rng)

    raise ValueError(f"Unknown agent_type: {at!r}")


def _build_value_fn(name: str, finite_horizon: bool = True):
    if name == 'xgboost':
        from agents.fn.value_fn import XGBoostValueFn
        return XGBoostValueFn(finite_horizon=finite_horizon)
    if name == 'neural':
        from agents.fn.value_fn import NeuralValueFn
        return NeuralValueFn(finite_horizon=finite_horizon)
    if name == 'ranking':
        from agents.fn.ranking import RankingValueFn
        return RankingValueFn(finite_horizon=finite_horizon)
    raise ValueError(f"Unknown value_fn: {name!r}")


def _build_action_gen(name: str, log_q_breakdown: bool = False):
    if name == 'local_search':
        from agents.action_gen import LocalSearchGenerator
        return LocalSearchGenerator(log_q_breakdown=log_q_breakdown)
    if name == 'sequential':
        from agents.action_gen import SequentialGenerator
        return SequentialGenerator(log_q_breakdown=log_q_breakdown)
    if name == 'bdq':
        from agents.action_gen import BDQGenerator
        return BDQGenerator()
    raise ValueError(f"Unknown action_gen: {name!r}")
