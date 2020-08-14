function D3Trees.D3Tree(D::UCT_DESPOT; title="UCT-DESPOT Tree", kwargs...)
    lenb = length(D.children)
    lenba = length(D.ba_children)
    len = lenb + lenba
    K = length(D.scenarios[1])
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tt = fill("", len)
    link_style = fill("", len)
    for b in 1:lenb
        children[b] = D.children[b] .+ lenb
        text[b] = @sprintf("""
                           o:%s (|Φ|:%3d)
                           N:%d""",
                           b==1 ? "<root>" : string(D.obs[b]),
                           length(D.scenarios[b]),
                           D.N[b],
                          )
        tt[b] = """
                o: $(b==1 ? "<root>" : string(D.obs[b]))
                |Φ|: $(length(D.scenarios[b]))
                N: $(D.N[b])
                $(length(D.children[b])) children
                """
        link_width = 20.0*sqrt(length(D.scenarios[b])/K)
        link_style[b] = "stroke-width:$link_width"
        for ba in D.children[b]
            link_style[ba+lenb] = "stroke-width:$link_width"
        end

        for ba in D.children[b]
            children[ba+lenb] = D.ba_children[ba]
            text[ba+lenb] = @sprintf("""
                                     a:%s
                                     V:%6.2f, N:%d""",
                                     D.ba_action[ba],
                                     D.ba_V[ba], D.ba_N[ba])
            tt[ba+lenb] = """
                          a: $(D.ba_action[ba])
                          V: $(D.ba_V[ba])
                          N: $(D.ba_N[ba])
                          $(length(D.ba_children[ba])) children
                          """
        end
    end
    return D3Tree(children;
                  text=text,
                  tooltip=tt,
                  link_style=link_style,
                  title=title,
                  kwargs...
                 )
end

Base.show(io::IO, mime::MIME"text/html", D::UCT_DESPOT) = show(io, mime, D3Tree(D))
Base.show(io::IO, mime::MIME"text/plain", D::UCT_DESPOT) = show(io, mime, D3Tree(D))