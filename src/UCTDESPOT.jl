module UCTDESPOT

using POMDPs
using BeliefUpdaters
using Parameters
using CPUTime
using ParticleFilters
using D3Trees
using Random
using Printf
using POMDPModelTools
using POMDPPolicies

using BasicPOMCP # for ExceptionRethrow and NoDecision
import BasicPOMCP.default_action

import Random.rand # Make it visible within this module

export # Make it visible to the public
    # Defined in this file (ARDESPOT.jl)
    UCT_DESPOTSolver,
    UCT_DESPOTPlanner,

    # Defined in random_2.jl
    DESPOTRandomSource,
    MemorizingSource,
    MemorizingRNG,

    # Defined in scenario_belief.jl
    ScenarioBelief,
    previous_obs,

    ReportWhenUsed # Defined in package MCTS

# include("random.jl")
include("random_2.jl")
include("scenario_belief.jl")
include("default_policy_sim.jl")

"""
    UCT_DESPOTSolver(<keyword arguments>)

Implementation of the ARDESPOT solver trying to closely match the pseudo code of:

http://bigbird.comp.nus.edu.sg/m2ap/wordpress/wp-content/uploads/2017/08/jair14.pdf

Each field may be set via keyword argument. The fields that correspond to algorithm
parameters match the definitions in the paper exactly.

# Fields
- `K`
- `D`
- `m`
- `T_max`
- `max_trials`
- `rollout_policy`
- `initializer`
- `rng`
- `random_source`
- `tree_in_info`
- `c`

Further information can be found in the field docstrings (e.g.
`?UCT_DESPOTSolver.xi`)
"""
@with_kw mutable struct UCT_DESPOTSolver <: Solver
    "The number of sampled scenarios."
    K::Int                                  = 500

    "The maximum depth of the DESPOT."
    D::Int                                  = 90

    "The minimal number of particles in a belief node"
    m::Int                                  = 20

    "Min reserve for random source"
    min_reserve::Int                        = 0

    "The maximum online planning time per step."
    T_max::Float64                          = 1.0

    "The maximum number of trials of the planner."
    max_trials::Int                         = typemax(Int)

    "A rollout policy "
    rollout_policy::Union{Policy, Nothing}  = nothing

    "Heuristic initializer for V and N"
    initializer::Function                    = (b,a)->(0.0,0)

    "A random number generator for the internal sampling processes."
    rng::MersenneTwister                    = MersenneTwister(rand(UInt32))

    "A source for random numbers in scenario rollout"
    random_source::DESPOTRandomSource       = MemorizingSource(K, D, rng, min_reserve=min_reserve)

    "If true, a reprenstation of the constructed DESPOT is returned by POMDPModelTools.action_info."
    tree_in_info::Bool                      = false

    "UCB exploration constant - specifies how much the solver should explore."
    c::Float64                              = 1.0
end

struct UCT_DESPOTPlanner{P<:POMDP, RS<:DESPOTRandomSource, RNG<:AbstractRNG} <: Policy
    sol::UCT_DESPOTSolver
    pomdp::P
    rollout_policy::Policy
    rs::RS
    rng::RNG
end

function UCT_DESPOTPlanner(sol::UCT_DESPOTSolver, pomdp::POMDP)
    rollout_policy = sol.rollout_policy === nothing ? RandomPolicy(pomdp) : sol.rollout_policy
    rng = deepcopy(sol.rng)
    rs = deepcopy(sol.random_source)
    Random.seed!(rs, rand(rng, UInt32))
    return UCT_DESPOTPlanner(deepcopy(sol), pomdp, rollout_policy, rs, rng)
end

include("tree.jl")
include("planner.jl")
include("pomdps_glue.jl")

include("visualization.jl")

end # module
