# LB-DESPOT

## Installation

```bash
git clone https://github.com/AutomobilePOMDP/UCT-DESPOT
cd UCT-DESPOT
julia
```
```julia
Pkg> add POMDPs
Pkg> registry add https://github.com/JuliaPOMDP/Registry
Pkg> activate .
Pkg> instantiate
Pkg> precompile
```
## Usage
```
using UCTDESPOT
solver = UCT_DESPOTSolver()
planner = solve(solver, pomdp)
```
Key parameter:
- K: The number of sampled scenarios.
- m: The minimal number of particles in a belief node. Belief nodes with number of particles less than m will not be expanded, whose value will be estimated by the average rollout value of all its state particles.
- T_max: The maximum online planning time per step.
- c: UCB exploration constant - specifies how much the solver should explore.
- rollout_policy: A policy used for rollout.
- initializer: A function used to initialize the action node, which take in a belief-action pair and then return its corresponding $(V_{init}, N_{init})$.