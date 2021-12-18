using Flux
using KitBase.JLD2
using Flux: @epochs

cd(@__DIR__)
include("../common.jl")
@load "data/dataset.jld2" X Y
#@load "model/nn_pro.jld2" nn
@load "model/nn_layer.jld2" nn

data = Flux.Data.DataLoader((X, Y), shuffle = true)
ps = params(nn)
loss(x, y) = Flux.binarycrossentropy(nn(x), y)
cb = () -> println("loss: $(loss(X, Y))")
opt = ADAM()

@epochs 5 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

accuracy(nn, X, Y)

@save "model/nn_layer.jld2" nn
