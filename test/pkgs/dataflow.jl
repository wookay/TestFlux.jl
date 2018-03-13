module test_pkgs_dataflow

using Base.Test
using DataFlow # IVertex DataFlow.il
               # DVertex DataFlow.dl
using DataFlow: prewalk, walk

# https://github.com/MikeInnes/DataFlow.jl/blob/master/docs/vertices.md

# input-linked
iv = IVertex(2)
@test Vector{IVertex{Int}}() == iv.inputs
@test 2 == iv.value

@test prewalk(identity, iv) == iv
@test walk(iv, identity, identity) == iv

@test contains(iv, 2)

@test DataFlow.il(iv) == iv

# doubly-linked
dv = DVertex(2)
@test Vector{DVertex{Int}}() == dv.inputs
@test 2 == dv.value
@test DataFlow.dl(iv) == dv
@test DataFlow.il(dv) == iv

# Syntax → Graph
import DataFlow: syntax, graphm

ex = syntax(dv)
@test ex isa Expr
@test ex.args[1].args[1] == 2
@test graphm(Dict(), ex) == dv 

end # module test_pkgs_dataflow
