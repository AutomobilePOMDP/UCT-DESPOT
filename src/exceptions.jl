struct NoTree <: NoDecision end

Base.show(io::IO, ::NoTree) = print(io, """
    No tree was created.

    Use the default_action solver parameter to specify behavior for this case.
    """)