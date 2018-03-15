# load

using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using BSON: @save, @load
using Base.Iterators: partition

m = Chain(
  Conv((2,2), 1=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax) |> gpu

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

evalcb = throttle(() -> @show(accuracy(tX, tY)), 1)
opt = ADAM(params(m))


@load "fashion_mnist_model.bson" m
@load "fashion_mnist_weights.bson" weights
Flux.loadparams!(m, weights)


using MLDatasets # FashionMNIST
using ColorTypes: N0f8, Gray

const Img = Matrix{Gray{N0f8}}

function prepare_test()
    # load full test set
    test_x,  test_y  = FashionMNIST.testdata() # 10_000
    test_imgs = Img.([test_x[:,:,i] for i in 1:10_000])
    
    # Prepare test set
    tX = cat(4, float.(test_imgs[1:10_000])...) |> gpu
    tY = onehotbatch(test_y[1:10_000], 0:9) |> gpu
    tX, tY
end

tX, tY = prepare_test()
@show(accuracy(tX, tY))
#  accuracy(tX, tY) = 0.3329
