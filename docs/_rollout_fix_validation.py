"""Validation: vanishing-horizon bug vs fix on instance_10p (shortened horizon).

Same per-asset economics as instance_10p; only the planning horizon T and tail
are shortened (T=40, tail=40 -> eval_length=80) so a full T+tail evaluation is
tractable locally. The vanishing-horizon pathology is purely about horizon
length, so this faithfully reproduces it.

Compares, CRN-paired across agents (shared 'evaluation' phase):
  - base reactive (the tuned degenerate base policy the rollout rolls out)
  - OLD rollout  : max_steps = T - t_next  (reproduces the bug: 0 in the tail)
  - FIXED rollout: fixed lookahead window (the shipped fix)  -- fixed + adaptive
"""
import json, numpy as np
from env.mdp import InfraEnv, EnvConfig, State
from env.network import load_sioux_falls
from env.tap import make_tap
from agents.heuristics import ReactiveAgent
from agents.rollout import MonteCarloRolloutAgent

INSTANCE="instances/instance_10p.json"
T=40; TAIL=40; EVAL_LEN=T+TAIL; NROLL=10
def _arr(v,n): return np.full(n,float(v)) if isinstance(v,(int,float)) else np.array(v,float)
def build_env(seed=0):
    inst=json.load(open(INSTANCE)); net=load_sioux_falls(n_assets=inst["n_assets"]); n=net.n_assets
    cfg=EnvConfig(n_assets=n,dt=inst["dt"],T=T,
        gamma=inst["gamma"]**inst["dt"],d_fail=inst["d_fail"],eta_ren=inst["eta_ren"],eta_load=inst["eta_load"],
        restrict_degrad_multiplier=float(inst.get("restrict_degrad_multiplier",0.5)),
        mu_h=_arr(inst["mu_h"],n),sigma_h=_arr(inst["sigma_h"],n),delta_repair=inst["delta_repair"],
        alpha0=_arr(inst["alpha0"],n),beta=_arr(inst["beta"],n),c_ren=_arr(inst["c_ren"],n),
        c_rep=_arr(inst["c_rep"],n),asset_lengths_m=_arr(inst["asset_lengths_m"],n),
        vot=float(inst.get("vot",10.76)),traffic_cost_factor=float(inst.get("traffic_cost_factor",1.0)),
        risk_base=float(inst.get("risk_base",10000.0)),
        d_init=np.array(inst["d_init"],float) if inst.get("d_init") is not None else None)
    return InfraEnv(net,make_tap(net,backend="fast"),cfg,rng_seed=seed)

class OldHorizonRollout(MonteCarloRolloutAgent):
    """Reproduce the pre-fix vanishing horizon: max_steps = T - t_next."""
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        self.rollout_horizon=None  # force the buggy branch below
    def _estimate_q(self,state,action):
        cfg=self.env.config
        s_post=self.env.post_decision_state(state,action)
        ic=self.env.immediate_cost(state,action,s_post)
        t_next=state.t+1
        max_steps=cfg.T - t_next      # <-- the bug
        if max_steps<=0: return ic
        fc=float(np.mean([self._single_rollout(s_post,t_next,max_steps,state,r) for r in range(self.n_rollouts)]))
        return ic+fc

env=build_env(); n=env.config.n_assets
base=ReactiveAgent(threshold=0.999974841735008,env_config=env.config,repair_threshold=0.9243,restrict_threshold=0.3342)

def run_eval(make_agent,label,n_eps=2):
    costs=[]; cr=cm=ct=0.0; acts={0:0,1:0,2:0,3:0}
    for ep in range(n_eps):
        agent=make_agent()
        env.begin_episode("evaluation",ep)
        st=env.reset(); disc=1.0; g=env.config.gamma; tot=0.0
        for t in range(EVAL_LEN):
            a=agent.act(st)
            for v in a: acts[int(v)]+=1
            st,cost,done=env.step(st,a)
            c_t,c_m,c_r=env.last_cost_breakdown
            tot+=disc*cost; cr+=disc*c_r; cm+=disc*c_m; ct+=disc*c_t; disc*=g
        costs.append(tot)
    s=cr+cm+ct; tota=sum(acts.values())
    print(f"{label:26s} mean_cost={np.mean(costs):.3e}  risk%={cr/s*100:4.1f} maint%={cm/s*100:4.1f} travel%={ct/s*100:4.1f}  none%={acts[0]/tota*100:4.1f} ren%={acts[2]/tota*100:.1f}")

print(f"# shortened eval: T={T} tail={TAIL} eval_len={EVAL_LEN} n_rollouts={NROLL}, instance_10p economics, 2 CRN episodes\n")
run_eval(lambda: base, "base reactive(degenerate)")
run_eval(lambda: OldHorizonRollout(rollout_policy=base,env=env,n_rollouts=NROLL,seed=0,action_threshold=0.5,
            initial_action='policy',selection='adaptive',p_threshold=0.02,min_rollouts=NROLL,max_rollouts=NROLL,rollout_batch=5),
         "OLD rollout (vanishing)")
run_eval(lambda: MonteCarloRolloutAgent(rollout_policy=base,env=env,n_rollouts=NROLL,seed=0,action_threshold=0.5,
            initial_action='policy',selection='fixed'),  # horizon=None -> fixed T window (the fix)
         "FIXED rollout (fixed-sel)")
run_eval(lambda: MonteCarloRolloutAgent(rollout_policy=base,env=env,n_rollouts=NROLL,seed=0,action_threshold=0.5,
            initial_action='policy',selection='adaptive',p_threshold=0.02,min_rollouts=NROLL,max_rollouts=NROLL,rollout_batch=5),
         "FIXED rollout (adaptive)")
