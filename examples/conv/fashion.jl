# model for FashionMNIST
# 
# reference: https://github.com/FluxML/model-zoo/blob/master/mnist/conv.jl

using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using BSON: @save, @load
using Base.Iterators: partition

using MLDatasets # FashionMNIST
using ColorTypes: N0f8, Gray

const Img = Matrix{Gray{N0f8}}

function prepare_train()
    # load full training set
    train_x, train_y = FashionMNIST.traindata() # 6000
    imgs = Img.([train_x[:,:,i] for i in 1:6000])
    labels = onehotbatch(train_y, 0:9)
    imgs, labels
end

function prepare_test()
    # load full test set
    test_x,  test_y  = FashionMNIST.testdata() # 10_000
    test_imgs = Img.([test_x[:,:,i] for i in 1:10_000])

    # Prepare test set
    tX = cat(4, float.(test_imgs[1:10_000])...) |> gpu
    tY = onehotbatch(test_y[1:10_000], 0:9) |> gpu
    tX, tY
end

imgs, labels = prepare_train()
tX, tY = prepare_test()

# Partition into batches of size 1,000
train = [(cat(4, float.(imgs[i])...), labels[:,i])
         for i in partition(1:6000, 1000)]

train = gpu.(train)

m = Chain(
  Conv((2,2), 1=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax) |> gpu

# m(train[1][1])

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

evalcb = throttle(() -> @show(accuracy(tX, tY)), 1)
opt = ADAM(params(m))

@epochs 5 Flux.train!(loss, train, opt, cb = evalcb)

#=
INFO: Epoch 1
accuracy(tX, tY) = 0.1
accuracy(tX, tY) = 0.1
INFO: Epoch 2
accuracy(tX, tY) = 0.1
accuracy(tX, tY) = 0.1
INFO: Epoch 3
accuracy(tX, tY) = 0.1
accuracy(tX, tY) = 0.1001
INFO: Epoch 4
accuracy(tX, tY) = 0.1163
accuracy(tX, tY) = 0.1947
INFO: Epoch 5
accuracy(tX, tY) = 0.2864
accuracy(tX, tY) = 0.3105
=#


# @save "fashion_mnist_model.bson" m
# weights = Tracker.data.(params(m)) ;
# @save "fashion_mnist_weights.bson" weights
