using Base.Test
using Flux # Dense
using BSON # BSON.@load

BSON.@load joinpath(@__DIR__, "mymodel.bson") model

@test model.layers[1] isa Dense
@test size(model.layers[1].W) == (5, 10)
