using UCTDESPOT
using Test

using POMDPs
using POMDPModels
using POMDPSimulators
using Random
using POMDPModelTools
using ParticleFilters

pomdp = BabyPOMDP()
pomdp.discount = 1.0

K = 10
rng = MersenneTwister(14)
rs = MemorizingSource(K, 90, rng)
Random.seed!(rs, 10)
b_0 = initialstate_distribution(pomdp)
scenarios = [i=>rand(rng, b_0) for i in 1:K]
o = false
b = ScenarioBelief(scenarios, rs, 0, o)

scenarios = [1=>rand(rng, b_0)]
b = ScenarioBelief(scenarios, rs, 0, false)

@testset "Rollout" begin
    pol = FeedWhenCrying()
    solver = UCT_DESPOTSolver(rollout_policy = pol)
    planner = UCT_DESPOTPlanner(solver, pomdp)
    r1 = UCTDESPOT.rollout(pomdp, pol, b, 10, (b,a)->(0.0,0))
    r2 = UCTDESPOT.rollout(pomdp, pol, b, 10, (b,a)->(0.0,0))
    @test r1 == r2
    tval = 7.0
    r3 = UCTDESPOT.rollout(pomdp, pol, b, 10, (b,a)->(tval,1))
    @test r3 == r2 + tval
end

# @testset "Random" begin
#     D,_ = UCTDESPOT.build_despot(p, b0)
#     rng = get_rng(p.rs, first(scen), D.Delta[1])
#     num1 = rand(rng)
#     rng = get_rng(p.rs, first(scen), D.Delta[1])
#     num2 = rand(rng)
#     @test num1 == num2
# end

pomdp = BabyPOMDP(-5, -10, 0.1, 0.8, 0.1, 0.95) 

# random rollout policy
solver = UCT_DESPOTSolver(c=100.0)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=100)
println("\nRandom rollout policy:")
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# FeedWhenCrying policy
solver = UCT_DESPOTSolver(c=100.0, rollout_policy=FeedWhenCrying())
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=100)
println("\nFeedWhenCrying as rollout policy:")
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")

# FeedWhenCrying policy with a bad initializer
initializer(b, a) = (reward(pomdp, true, false)/(1-discount(pomdp)), 1)
solver = UCT_DESPOTSolver(c=100.0, rollout_policy=FeedWhenCrying(), initializer=initializer)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=100)
println("\nFeedWhenCrying with a bad final value as rollout policy:")
@time hist = simulate(hr, pomdp, planner)
println("Discounted reward is $(discounted_reward(hist))")


# Type stability
pomdp = BabyPOMDP()
solver = UCT_DESPOTSolver(rollout_policy=FeedWhenCrying(),
                      rng=MersenneTwister(4)
                     )
p = solve(solver, pomdp)

b0 = initialstate_distribution(pomdp)
D,_ = @inferred UCTDESPOT.build_despot(p, b0)
@inferred UCTDESPOT.explore!(D, 1, p)
@inferred UCTDESPOT.next!(D, 1, p)
@inferred action(p, b0)

rng = MersenneTwister(4)
solver = UCT_DESPOTSolver(rollout_policy=FeedWhenCrying(),
                      rng=rng,
                      random_source=MemorizingSource(500, 90, rng),
                      tree_in_info=true
                     )
p = solve(solver, pomdp)
a = action(p, initialstate_distribution(pomdp))

# visualization
println("\nTree:\n")
show(stdout, MIME("text/plain"), D)
a, info = action_info(p, initialstate_distribution(pomdp))
show(stdout, MIME("text/plain"), info[:tree])

# from README:
println("\nTigerPOMDP in README:\n")
using POMDPs, POMDPModels, POMDPSimulators, UCTDESPOT

pomdp = TigerPOMDP()

solver = UCT_DESPOTSolver()
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
