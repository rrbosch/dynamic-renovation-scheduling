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

    # 2. Network — pass n_assets from instance
    if config.network == 'sioux_falls':
        network = load_sioux_falls(n_assets=inst['n_assets'])
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
        n_warmstart_episodes=int(tc_dict.get('n_warmstart_episodes', 0)),
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
                thr = np.column_stack([
                    [params[f'repair_threshold_{i}']   for i in range(_n)],
                    [params[f'restrict_threshold_{i}'] for i in range(_n)],
                    [params[f'renovate_threshold_{i}'] for i in range(_n)],
                ])  # shape (N, 3)
                return PerAssetReactiveAgent(thresholds=thr, env_config=_cfg)
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
_REACTIVE_EXTRA_KEYS = {'threshold', 'repair_threshold', 'restrict_threshold'}
_PACED_EXTRA_KEYS = {'threshold', 'pace_threshold'}


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

    if at in ('adp', 'dqn', 'actor_critic'):
        finite_horizon = extra.get('finite_horizon', True)
        vf = _build_value_fn(agent_config.value_fn, finite_horizon=finite_horizon)
        ag = _build_action_gen(agent_config.action_gen,
                               log_q_breakdown=extra.get('log_q_breakdown', False))

        from training.trainer import TrainingConfig  # reuse default

        if at == 'adp':
            from agents.dqn import ADPAgent
            return ADPAgent(vf, ag, env, TrainingConfig(), finite_horizon=finite_horizon)

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
        from agents.dcl import DCLAgent, XGBoostPolicy, NNPolicy

        finite_horizon = extra.get('finite_horizon', True)
        policy_type    = extra.get('policy_type', 'xgboost')

        value_fn = _build_value_fn(extra.get('value_fn', 'xgboost'), finite_horizon=finite_horizon)

        n_assets = env.config.n_assets
        if policy_type == 'nn':
            from agents.fn.policy import PolicyNetwork
            input_dim = 5 * n_assets + (1 if finite_horizon else 0)
            net    = PolicyNetwork(input_dim, n_assets,
                                   hidden_dims=extra.get('hidden_dims', [256, 256]),
                                   T=env.config.T, finite_horizon=finite_horizon)
            policy = NNPolicy(net, lr=extra.get('policy_lr', 1e-3))
        else:  # 'xgboost'
            policy = XGBoostPolicy(
                n_assets=n_assets,
                env_config=env.config,
                use_global_context=extra.get('use_global_context', True),
                clf_kwargs=extra.get('clf_kwargs', {}),
            )

        heuristic_cfg = extra.get('heuristic_policy',
                                  {'agent_type': 'reactive', 'extra': {'threshold': 0.8}})
        heuristic_policy = _build_agent(AgentConfig(**heuristic_cfg), env, seed)
        action_gen       = _build_action_gen(extra.get('action_gen', 'local_search'))

        return DCLAgent(
            policy=policy,
            value_fn=value_fn,
            env=env,
            heuristic_policy=heuristic_policy,
            action_gen=action_gen,
            rollout_horizon=extra.get('rollout_horizon', 10),
            n_rollouts=extra.get('n_rollouts', 5),
            min_samples_train=extra.get('min_samples_train', 5000),
            finite_horizon=finite_horizon,
            rng=np.random.default_rng(seed + 999),
        )

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
