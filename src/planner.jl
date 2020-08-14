function build_despot(p::UCT_DESPOTPlanner, b_0)
    D = UCT_DESPOT(p, b_0)
    b = 1
    trial = 1
    start = CPUtime_us()

    tree_depth = 0

    while CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        b, rewards = explore!(D, 1, p)
        if D.Delta[b] > tree_depth
            tree_depth = D.Delta[b]
        end
        backup!(D, b, p, rewards)
        trial += 1
    end

    return D, [trial, (CPUtime_us()-start)/1e6, tree_depth]
end

function explore!(D::UCT_DESPOT, b::Int, p::UCT_DESPOTPlanner)
    rewards = Float64[]
    while D.Delta[b] <= p.sol.D &&
            length(D.scenarios[b]) >= p.sol.m

        # If all scenarios in b are terminal, it will be meaningless to expand it.
        if all([isterminal(p.pomdp, s) for (k,s) in D.scenarios[b]])
            push!(rewards, 0.0)
            return b, rewards
        end

        if isempty(D.children[b]) # a leaf
            expand!(D, b, p)
        end
        last_b = b
        b, reward = next!(D, b, p) # it may expand new scenario in next belief node
        push!(rewards, reward)
        if b === nothing
            return last_b, rewards
        end
    end
    scenario_belief = get_belief(D, b, p.rs)
    push!(rewards, branching_sim(p.pomdp, p.rollout_policy, scenario_belief, p.sol.D-D.Delta[b], p.sol.initializer)/length(D.scenarios[b]))
    return b, rewards
end

function backup!(D::UCT_DESPOT, b::Int, p::UCT_DESPOTPlanner, rewards::Vector{Float64})
    discounted_return = 0.0 + discount(p.pomdp) * last(rewards)
    while b != 1
        depth = D.Delta[b]
        ba = D.parent[b]
        b = D.parent_b[b]
        discounted_return = rewards[depth] + discount(p.pomdp)*discounted_return

        D.N[b] += 1
        D.ba_N[ba] += 1
        D.ba_V[ba] += (discounted_return - D.ba_V[ba]) / D.ba_N[ba]
    end
end

function next!(D::UCT_DESPOT, b::Int, p::UCT_DESPOTPlanner)
    max = -Inf
    lnNb = log(D.N[b])
    best_ba = first(D.children[b])
    for ba in D.children[b]
        val = D.ba_V[ba] + p.sol.c * sqrt(lnNb/D.ba_N[ba])
        if val > max
            max = val
            best_ba = ba
        end
    end
    r = 0.0
    scen = rand(p.rng, D.scenarios[b])
    s = last(scen)
    rng = get_rng(p.rs, first(scen), D.Delta[b])
    if !isterminal(p.pomdp, s)
        sp, o, r = gen(DDNOut(:sp, :o, :r) , p.pomdp, s, D.ba_action[best_ba], rng)
        bp = get(D.ba_odict[best_ba], o, 0)
        return bp, r
    end
    return nothing, 0.0
end