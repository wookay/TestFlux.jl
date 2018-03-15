# load

using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using BSON: @save, @load
using Base.Iterators: partition

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

@load "fashion_mnist_mlp_model.bson" m
@load "fashion_mnist_mlp_weights.bson" weights
Flux.loadparams!(m, weights)


using MLDatasets # FashionMNIST
using ColorTypes: N0f8, Gray

const Img = Matrix{Gray{N0f8}}

function prepare_test()
    # load full test set
    test_x,  test_y  = FashionMNIST.testdata() # 10_000

    testrange = 1:1_000 # 1:10_000
    test_imgs = Img.([test_x[:,:,i] for i in testrange])
    tX = hcat(float.(reshape.(test_imgs, :))...) |> gpu
    tY = onehotbatch(test_y[testrange], 0:9) |> gpu
    tX, tY
end

tX, tY = prepare_test()
@show(accuracy(tX, tY))
# accuracy(tX, tY) = 0.839
