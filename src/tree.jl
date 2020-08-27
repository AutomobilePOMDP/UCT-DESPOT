struct UCT_DESPOT{S,A,O}
    scenarios::Vector{Vector{Pair{Int,S}}} # to scenarios (index-state pair) of every *belief node*
    last::Vector{Int} # denote the last expanded scenario of *ba node*
    children::Vector{Vector{Int}} # to children *ba nodes* of every *belief node*
    parent_b::Vector{Int} # maps to parent *belief node*
    parent::Vector{Int} # maps to the parent *ba node*
    Delta::Vector{Int}
    N::Vector{Int} # needed for action selection
    obs::Vector{O}

    ba_children::Vector{Vector{Int}} # to children *belief nodes* of every *ba node*
    ba_odict::Vector{Dict{O, Int}}
    ba_N::Vector{Int}
    ba_V::Vector{Float64}
    ba_action::Vector{A}
end

function UCT_DESPOT(p::UCT_DESPOTPlanner, b_0)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)

    root_scenarios = [i=>rand(p.rng, b_0) for i in 1:p.sol.K]
    if all([isterminal(p.pomdp, s) for (k,s) in root_scenarios])
        throw(NoTree())
    end

    return UCT_DESPOT{S,A,O}([root_scenarios],
                         Int[],
                         [Int[]],
                         [0],
                         [0],
                         [0],
                         [0],
                         Vector{O}(undef, 1),

                         Vector{Int}[],
                         Vector{Dict{O, Int}}[],
                         Int[],
                         Float64[],
                         A[]
                 )
end

function expand!(D::UCT_DESPOT, b::Int, p::UCT_DESPOTPlanner)
    S = statetype(p.pomdp)
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)

    belief = get_belief(D, b, p.rs)
    for a in actions(p.pomdp, belief)
        # initialize ba related attributes
        push!(D.ba_odict, Dict{O, Int}())
        push!(D.ba_action, a)
        ba_V, ba_N = p.sol.initializer(b, a)
        push!(D.ba_V, ba_V)
        push!(D.ba_N, ba_N)
        ba = length(D.ba_odict)
        push!(D.children[b], ba)
        push!(D.last, p.sol.m)

        Gsum = 0.0
        for scen in D.scenarios[b][1:p.sol.m]
            rng = get_rng(p.rs, first(scen), D.Delta[b])
            s = last(scen)
            if !isterminal(p.pomdp, s) # expand if s isn't a terminal state
                sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, rng)
                Gsum += r
                bp = get(D.ba_odict[ba], o, 0)
                if bp == 0
                    push!(D.scenarios, Vector{Pair{Int, S}}())
                    # store bp in ba_odict
                    bp = length(D.scenarios)
                    D.ba_odict[ba][o] = bp
                end
                # store scenario in bp
                push!(D.scenarios[bp], first(scen)=>sp)
            end
        end

        push!(D.ba_children, collect(values(D.ba_odict[ba])))
        nbps = length(D.ba_odict[ba])
        resize!(D, length(D.children) + nbps)
        for (o, bp) in D.ba_odict[ba] # Initialize bp
            # initialize bp related attributes
            D.obs[bp] = o
            D.children[bp] = Int[]
            D.parent_b[bp] = b
            D.parent[bp] = ba
            D.Delta[bp] = D.Delta[b]+1
            D.N[bp] = 0
            scenario_belief = get_belief(D, bp, p.rs)
            Gsum += discount(p.pomdp) * branching_sim(p.pomdp, p.rollout_policy, scenario_belief, p.sol.D-D.Delta[bp], p.sol.initializer)
        end
        # Update ba_V so that the best action can be choosen accordingly
        # D.ba_N[ba] += p.sol.m
        # D.ba_V[ba] += (Gsum - D.ba_V[ba] * p.sol.m) / D.ba_N[ba]
        D.ba_N[ba] += 1
        D.ba_V[ba] += (Gsum/p.sol.m - D.ba_V[ba])/D.ba_N[ba]
        D.N[b] += D.ba_N[ba]
    end
end

function get_belief(D::UCT_DESPOT, b::Int, rs::DESPOTRandomSource)
    if isassigned(D.obs, b)
        ScenarioBelief(D.scenarios[b], rs, D.Delta[b], D.obs[b])
    else
        ScenarioBelief(D.scenarios[b], rs, D.Delta[b], missing)
    end
end

function Base.resize!(D::UCT_DESPOT, n::Int)
    resize!(D.children, n)
    resize!(D.parent_b, n)
    resize!(D.parent, n)
    resize!(D.Delta, n)
    resize!(D.N, n)
    resize!(D.obs, n)
end
