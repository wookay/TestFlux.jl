# fashion mnist mlp 
#
# reference: https://github.com/FluxML/model-zoo/blob/master/mnist/mlp.jl

using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using BSON: @save, @load
using Base.Iterators: repeated

using MLDatasets # FashionMNIST
using ColorTypes: N0f8, Gray

const Img = Matrix{Gray{N0f8}}

function prepare_train()
    # load full training set
    train_x, train_y = FashionMNIST.traindata() # 60_000

    trainrange = 1:6_000 # 1:60_000
    imgs = Img.([train_x[:,:,i] for i in trainrange])
    # Stack images into one large batch
    X = hcat(float.(reshape.(imgs, :))...) |> gpu
    # One-hot-encode the labels
    Y = onehotbatch(train_y[trainrange], 0:9) |> gpu
    X, Y
end

function prepare_test()
    # load full test set
    test_x,  test_y  = FashionMNIST.testdata() # 10_000

    testrange = 1:1_000 # 1:10_000
    test_imgs = Img.([test_x[:,:,i] for i in testrange])
    tX = hcat(float.(reshape.(test_imgs, :))...) |> gpu
    tY = onehotbatch(test_y[testrange], 0:9) |> gpu
    tX, tY
end

X, Y = prepare_train()
tX, tY = prepare_test()

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

dataset = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))
opt = ADAM(params(m))

@epochs 5 Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 2))

accuracy(X, Y)
# 0.983

accuracy(tX, tY)
# 0.83

# @save "fashion_mnist_mlp_model.bson" m
# weights = Tracker.data.(params(m)) ;
# @save "fashion_mnist_mlp_weights.bson" weights
