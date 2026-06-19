"""Run logging utilities."""
from __future__ import annotations

import csv
import datetime
import json
import os
import sys
from pathlib import Path


class _TimestampedStream:
    """Line-buffering wrapper that prefixes each output line with a wall-clock
    timestamp. Wraps an underlying text stream (e.g. sys.stdout) so that *every*
    ``print()`` is timestamped without touching individual call sites.

    A prefix is emitted only at the start of a logical line, so a single print
    with embedded newlines is timestamped per line and partial writes (no
    trailing newline) are not split. Carriage returns are treated as line ends
    so this is intentionally NOT used for stderr (tqdm progress bars live there).
    """

    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        for line in data.splitlines(keepends=True):
            if self._at_line_start:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self._stream.write(f'[{ts}] ')
            self._stream.write(line)
            self._at_line_start = line.endswith('\n') or line.endswith('\r')
        return len(data)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        # Delegate everything else (isatty, fileno, encoding, ...) to the stream.
        return getattr(self._stream, name)


def enable_timestamped_stdout() -> None:
    """Prefix every stdout line with a timestamp. Idempotent; safe to call from
    multiple entry points. Leaves stderr untouched so tqdm bars stay intact."""
    if isinstance(sys.stdout, _TimestampedStream):
        return
    sys.stdout = _TimestampedStream(sys.stdout)


class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / 'training_log.csv'
        self._header_written = False
        self._eval_header_written = False
        self._metrics_header_written = False

    def write_config(self, config: dict) -> None:
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def log_step(self, episode: int, metrics: dict) -> None:
        """Append a row to training_log.csv."""
        row = {'episode': episode, **{k: v for k, v in metrics.items() if k != 'episodes'}}
        with open(self._log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def save_episodes(self, episodes: list) -> None:
        """Save episode data as a flat CSV (directly openable in Excel).

        Each row is one timestep. Columns: episode, t, cost,
        d_0..d_{N-1}, h_0..h_{N-1}, ell_0..ell_{N-1}, r_0..r_{N-1},
        a_0..a_{N-1}.
        """
        if not episodes or not episodes[0]:
            return
        first = episodes[0][0]
        state = first['state']
        n = len(state.d)
        n_actions = len(first['action'])
        fieldnames = (
            ['episode', 't', 'cost', 'c_travel', 'c_maint', 'c_risk']
            + [f'd_{i}' for i in range(n)]
            + [f'h_{i}' for i in range(n)]
            + [f'ell_{i}' for i in range(n)]
            + [f'r_{i}' for i in range(n)]
            + [f'n_fail_{i}' for i in range(n)]
            + [f'a_{i}' for i in range(n_actions)]
        )
        with open(self.run_dir / 'eval_episodes.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep_idx, ep in enumerate(episodes):
                for step in ep:
                    s = step['state']
                    row = {
                        'episode': ep_idx, 't': step['t'], 'cost': step['cost'],
                        'c_travel': step.get('c_travel', 0.0),
                        'c_maint':  step.get('c_maint',  0.0),
                        'c_risk':   step.get('c_risk',   0.0),
                    }
                    for i, v in enumerate(s.d):   row[f'd_{i}']   = v
                    for i, v in enumerate(s.h):   row[f'h_{i}']   = v
                    for i, v in enumerate(s.ell): row[f'ell_{i}'] = v
                    for i, v in enumerate(s.r):      row[f'r_{i}']      = v
                    for i, v in enumerate(s.n_fail): row[f'n_fail_{i}'] = v
                    for i, v in enumerate(step['action']): row[f'a_{i}'] = v
                    writer.writerow(row)

    def save_agent_metrics(self, episodes: list) -> None:
        """Save per-step agent metrics (e.g. n_candidates) to agent_metrics.csv.

        Columns: episode, t, <metric_cols...>
        Only written if at least one step has non-empty agent_metrics.
        """
        metric_keys: list[str] = []
        for ep in episodes:
            for step in ep:
                m = step.get('agent_metrics', {})
                if m:
                    metric_keys = list(m.keys())
                    break
            if metric_keys:
                break
        if not metric_keys:
            return
        fieldnames = ['episode', 't'] + metric_keys
        with open(self.run_dir / 'agent_metrics.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep_idx, ep in enumerate(episodes):
                for step in ep:
                    m = step.get('agent_metrics', {})
                    row = {'episode': ep_idx, 't': step['t']}
                    for k in metric_keys:
                        row[k] = m.get(k, '')
                    writer.writerow(row)

    # ------------------------------------------------------------------
    # Incremental evaluation saving (one episode at a time)
    # ------------------------------------------------------------------

    def start_eval(self, append: bool = False) -> None:
        """Reset incremental eval state. If append=True, keep existing file."""
        self._eval_header_written = append
        self._metrics_header_written = append

    def append_episode(self, ep_idx: int, ep_data: list) -> None:
        """Append one episode's rows to eval_episodes.csv."""
        if not ep_data:
            return
        state = ep_data[0]['state']
        n = len(state.d)
        n_actions = len(ep_data[0]['action'])
        fieldnames = (
            ['episode', 't', 'cost', 'c_travel', 'c_maint', 'c_risk']
            + [f'd_{i}' for i in range(n)]
            + [f'h_{i}' for i in range(n)]
            + [f'ell_{i}' for i in range(n)]
            + [f'r_{i}' for i in range(n)]
            + [f'n_fail_{i}' for i in range(n)]
            + [f'a_{i}' for i in range(n_actions)]
        )
        mode = 'a' if self._eval_header_written else 'w'
        with open(self.run_dir / 'eval_episodes.csv', mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._eval_header_written:
                writer.writeheader()
                self._eval_header_written = True
            for step in ep_data:
                s = step['state']
                row = {
                    'episode': ep_idx, 't': step['t'], 'cost': step['cost'],
                    'c_travel': step.get('c_travel', 0.0),
                    'c_maint':  step.get('c_maint',  0.0),
                    'c_risk':   step.get('c_risk',   0.0),
                }
                for i, v in enumerate(s.d):      row[f'd_{i}']      = v
                for i, v in enumerate(s.h):      row[f'h_{i}']      = v
                for i, v in enumerate(s.ell):    row[f'ell_{i}']    = v
                for i, v in enumerate(s.r):      row[f'r_{i}']      = v
                for i, v in enumerate(s.n_fail): row[f'n_fail_{i}'] = v
                for i, v in enumerate(step['action']): row[f'a_{i}'] = v
                writer.writerow(row)

    def append_agent_metrics(self, ep_idx: int, ep_data: list) -> None:
        """Append one episode's agent metrics to agent_metrics.csv."""
        metric_keys: list[str] = []
        for step in ep_data:
            m = step.get('agent_metrics', {})
            if m:
                metric_keys = list(m.keys())
                break
        if not metric_keys:
            return
        fieldnames = ['episode', 't'] + metric_keys
        mode = 'a' if self._metrics_header_written else 'w'
        with open(self.run_dir / 'agent_metrics.csv', mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._metrics_header_written:
                writer.writeheader()
                self._metrics_header_written = True
            for step in ep_data:
                m = step.get('agent_metrics', {})
                row = {'episode': ep_idx, 't': step['t']}
                for k in metric_keys:
                    row[k] = m.get(k, '')
                writer.writerow(row)

    def count_completed_eval_episodes(self) -> int:
        """Count distinct episode indices in eval_episodes.csv."""
        path = self.run_dir / 'eval_episodes.csv'
        if not path.exists():
            return 0
        seen = set()
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen.add(int(row['episode']))
        return len(seen)

    def load_eval_episode_costs(self, gamma: float) -> list[float]:
        """Load per-episode discounted total costs from eval_episodes.csv."""
        path = self.run_dir / 'eval_episodes.csv'
        if not path.exists():
            return []
        # Group costs by episode, ordered by t
        ep_costs: dict[int, list[tuple[int, float]]] = {}
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = int(row['episode'])
                t = int(row['t'])
                cost = float(row['cost'])
                ep_costs.setdefault(ep, []).append((t, cost))
        totals = []
        for ep in sorted(ep_costs.keys()):
            steps = sorted(ep_costs[ep], key=lambda x: x[0])
            total = sum((gamma ** t) * c for t, c in steps)
            totals.append(total)
        return totals

    def save_agent(self, agent) -> None:
        """Save the trained agent to results/<run>/agent/."""
        agent_dir = self.run_dir / 'agent'
        agent_dir.mkdir(exist_ok=True)
        agent.save(str(agent_dir))

    def save_buffer_predictions(self, buffer, agent) -> None:
        """V(s_post) predictions vs mc_return for all buffer transitions."""
        transitions = list(buffer._data)
        if not transitions:
            return
        post_states = [t.post_state for t in transitions]
        v_preds = agent.value_fn.predict(post_states)
        with open(self.run_dir / 'vf_buffer_predictions.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['mc_return', 'v_pred', 'residual'])
            writer.writeheader()
            for tr, vp in zip(transitions, v_preds):
                writer.writerow({'mc_return': tr.mc_return,
                                 'v_pred': float(vp),
                                 'residual': tr.mc_return - float(vp)})

    def save_eval_predictions(self, episodes, agent, env) -> None:
        """V(s_post) predictions vs realized returns for eval episodes."""
        gamma = env.config.gamma
        # V'(s_post) is trained with FUTURE-only targets (mc_return - cost), so the
        # apples-to-apples ground truth is the discounted return EXCLUDING the
        # current-step cost: realized_future_t = G_t - cost_t = gamma * G_{t+1}.
        # `realized_return` (incl. current cost) is kept for backward compatibility;
        # `residual` is now computed against `realized_future`.
        with open(self.run_dir / 'vf_eval_predictions.csv', 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['episode', 't', 'v_pred', 'realized_return',
                               'realized_future', 'residual'])
            writer.writeheader()
            for ep_idx, ep in enumerate(episodes):
                costs = [step['cost'] for step in ep]
                T = len(costs)
                realized = [0.0] * T
                G = 0.0
                for k in range(T - 1, -1, -1):
                    G = costs[k] + gamma * G
                    realized[k] = G
                for step, G_t, c_t in zip(ep, realized, costs):
                    post_state = env.post_decision_state(step['state'], step['action'])
                    vp = float(agent.value_fn.predict([post_state])[0])
                    realized_future = G_t - c_t
                    writer.writerow({'episode': ep_idx, 't': step['t'],
                                     'v_pred': vp,
                                     'realized_return': G_t,
                                     'realized_future': realized_future,
                                     'residual': realized_future - vp})
