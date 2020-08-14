# Branch until the leaf belief contains only one state particle or the steps count down to zero.
function branching_sim(pomdp::POMDP, policy::Policy, b::ScenarioBelief, steps::Integer, initializer)
    S = statetype(pomdp)
    O = obstype(pomdp)
    odict = Dict{O, Vector{Pair{Int, S}}}()

    a = action(policy, b)

    if steps <= 0
        return length(b.scenarios)*first(initializer(b, a))
    end

    r_sum = 0.0
    for (k, s) in b.scenarios
        if !isterminal(pomdp, s)
            rng = get_rng(b.random_source, k, b.depth)
            sp, o, r = gen(DDNOut(:sp, :o, :r), pomdp, s, a, rng)

            if haskey(odict, o)
                push!(odict[o], k=>sp)
            else
                odict[o] = [k=>sp]
            end

            r_sum += r
        end
    end

    next_r = 0.0
    for (o, scenarios) in odict
        bp = ScenarioBelief(scenarios, b.random_source, b.depth+1, o)
        if length(scenarios) == 1
            next_r += rollout(pomdp, policy, bp, steps-1, initializer)
        else
            next_r += branching_sim(pomdp, policy, bp, steps-1, initializer)
        end
    end

    return r_sum + discount(pomdp)*next_r
end

# once there is only one scenario left, just run a rollout
function rollout(pomdp::POMDP, policy::Policy, b0::ScenarioBelief, steps::Integer, initializer)
    @assert length(b0.scenarios) == 1
    disc = 1.0
    r_total = 0.0
    scenario_mem = copy(b0.scenarios)
    (k, s) = first(b0.scenarios)
    b = ScenarioBelief(scenario_mem, b0.random_source, b0.depth, b0._obs)

    while !isterminal(pomdp, s) 
        a = action(policy, b)
        if steps > 0
            rng = get_rng(b.random_source, k, b.depth)
            sp, o, r = gen(DDNOut(:sp, :o, :r), pomdp, s, a, rng)

            r_total += disc*r

            s = sp
            scenario_mem[1] = k=>s
            b = ScenarioBelief(scenario_mem, b.random_source, b.depth+1, o)

            disc *= discount(pomdp)
            steps -= 1
        else
            return r_total += disc*first(initializer(pomdp, b))
        end
    end
    return r_total
end
