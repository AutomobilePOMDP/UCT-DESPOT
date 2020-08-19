POMDPs.solve(sol::UCT_DESPOTSolver, p::POMDP) = UCT_DESPOTPlanner(sol, p)

function POMDPModelTools.action_info(p::UCT_DESPOTPlanner, b)
    try
        info = Dict{Symbol, Any}()
        Random.seed!(p.rs, rand(p.rng, UInt32))

        D, record = build_despot(p, b)

        if p.sol.tree_in_info
            info[:tree] = D
            info[:record] = record
        end

        check_consistency(p.rs)

        best_V = -Inf
        best_as = actiontype(p.pomdp)[]
        for ba in D.children[1]
            V = D.ba_V[ba]
            if V > best_V
                best_V = V
                best_as = [D.ba_action[ba]]
            elseif V == best_V
                push!(best_as, D.ba_action[ba])
            end
        end

        return rand(p.rng, best_as)::actiontype(p.pomdp), info # best_as will usually only have one entry, but we want to break the tie randomly
    catch ex
        return default_action(p.sol.default_action, p.pomdp, b, ex)::actiontype(p.pomdp), info
    end
end

POMDPs.action(p::UCT_DESPOTPlanner, b) = first(action_info(p, b))::actiontype(p.pomdp)

POMDPs.updater(p::UCT_DESPOTPlanner) = SIRParticleFilter(p.pomdp, p.sol.K, rng=p.rng)

function Random.seed!(p::UCT_DESPOTPlanner, seed)
    Random.seed!(p.rng, seed)
    return p
end
