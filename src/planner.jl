function build_despot(p::UCT_DESPOTPlanner, b_0)
    D = UCT_DESPOT(p, b_0)
    trial = 1
    start = CPUtime_us()

    tree_depth = 0

    while CPUtime_us()-start < p.sol.T_max*1e6 &&
          trial <= p.sol.max_trials
        explore!(D, 1, p)
        trial += 1
    end

    return D::UCT_DESPOT, [trial, (CPUtime_us()-start)/1e6]
end

function explore!(D::UCT_DESPOT, b::Int, p::UCT_DESPOTPlanner)
    if D.Delta[b] > p.sol.D || length(D.scenarios[b]) < p.sol.m
        scenario_belief = get_belief(D, b, p.rs)
        return (branching_sim(p.pomdp, p.rollout_policy, scenario_belief, p.sol.D-D.Delta[b], p.sol.initializer)/length(D.scenarios[b]))::Float64
    end

    # If all scenarios in b are terminal, it will be meaningless to expand it.
    if all([isterminal(p.pomdp, s) for (k,s) in D.scenarios[b]])
        return 0.0
    end

    if isempty(D.children[b]) # is a leaf
        expand!(D, b, p)
    end
    ba, r, bp = next!(D, b, p) # it may expand new scenario in next belief node
    R = r + (bp === nothing ? 0.0 : (discount(p.pomdp) * explore!(D, bp, p)))
    D.N[b] += 1
    D.ba_N[ba] += 1
    D.ba_V[ba] += (R - D.ba_V[ba]) / D.ba_N[ba]
    return R::Float64
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
    a = D.ba_action[best_ba]

    # expand unseen scenarios of best_ba
    len_scen = length(D.scenarios[b])
    new_expand = 0
    if D.last[best_ba] < len_scen
        Rsum = 0.0
        # Gsum = 0.0
        last_scen = D.last[best_ba] + p.sol.m < len_scen ? D.last[best_ba] + p.sol.m : len_scen
        new_expand = last_scen - D.last[best_ba]
        for scen in D.scenarios[b][D.last[best_ba]+1:last_scen]
            s = last(scen)
            rng = get_rng(p.rs, first(scen), D.Delta[b])
            if !isterminal(p.pomdp, s) # expand if s isn't a terminal state
                sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, rng)
                Rsum += r
                bp = get(D.ba_odict[best_ba], o, 0)
                if bp == 0
                    push!(D.scenarios, Vector{Pair{Int, statetype(p.pomdp)}}())
                    # store bp in ba_odict
                    bp = length(D.scenarios)
                    D.ba_odict[best_ba][o] = bp
                    push!(D.ba_children[best_ba], bp)
                    push!(D.obs, o)
                end
                # store scenario in bp
                push!(D.scenarios[bp], first(scen)=>sp)
                # Too much rollouts may cause the ba_V close to the value of rollout policy
                # Gsum += r + discount(p.pomdp) * rollout(p.pomdp,
                #                                         p.rollout_policy,
                #                                         ScenarioBelief([first(scen)=>sp,], p.rs, D.Delta[b] + 1, o),
                #                                         p.sol.D - D.Delta[b] - 1,
                #                                         p.sol.initializer)
            end
        end

        if length(D.obs) > length(D.children)
            bp_start = length(D.children) + 1
            resize!(D, length(D.obs))
            for bp in bp_start:length(D.obs) # Initialize bp
                # initialize bp related attributes
                D.children[bp] = Int[]
                D.parent_b[bp] = b
                D.parent[bp] = best_ba
                D.Delta[bp] = D.Delta[b]+1
                D.N[bp] = 0
            end
        end

        D.ba_R[best_ba] += (Rsum - D.ba_R[best_ba] * new_expand)/last_scen
        D.last[best_ba] = last_scen
        # D.ba_N[best_ba] += new_expand
        # D.ba_V[best_ba] += (Gsum - D.ba_V[best_ba] * new_expand) / D.ba_N[best_ba]
        # D.N[b] += new_expand
    end


    scen = rand(p.rng, D.scenarios[b][1:D.last[best_ba]])
    s = last(scen)
    rng = get_rng(p.rs, first(scen), D.Delta[b])
    bp = nothing
    if !isterminal(p.pomdp, s)
        o = @gen(:o)(p.pomdp, s, a, rng)
        bp = D.ba_odict[best_ba][o]
    end
    if new_expand == 0
        return best_ba::Int, D.ba_R[best_ba]::Float64, bp::Union{Int, Nothing}
    else
        return best_ba::Int, (Rsum/new_expand)::Float64, bp::Union{Int, Nothing}
    end
end