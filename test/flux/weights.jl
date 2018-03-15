using Base.Test
using Flux # Dense
using BSON # BSON.@load

BSON.@load joinpath(@__DIR__, "mymodel.bson") model
BSON.@load joinpath(@__DIR__, "myweights.bson") weights

Flux.loadparams!(model, weights)

@test model.layers[1] isa Dense
@test size(model.layers[1].W) == (5, 10)
