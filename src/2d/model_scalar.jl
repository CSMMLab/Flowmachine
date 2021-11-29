using Kinetic, LinearAlgebra, JLD2, Flux
using Flux: onecold, @epochs

cd(@__DIR__)
@load "data.jld2" X1 Y1 X2 Y2

Z1 = zeros(Float32, 1, size(Y1, 2))
for i in axes(Z1, 2)
    Z1[1, i] = ifelse(Y1[2, i] == 1.0, 1.0, 0.0)
end
Z2 = zeros(Float32, 1, size(Y2, 2))
for i in axes(Z2, 2)
    Z2[1, i] = ifelse(Y2[2, i] == 1.0, 1.0, 0.0)
end

device = cpu#gpu

X1 = Float32.(X1) |> device
Z1 = Float32.(Z1) |> device
X2 = Float32.(X2) |> device
Z2 = Float32.(Z2) |> device

isNewStart = true
#isNewStart = false#true
if isNewStart
    nn = Chain(
        Dense(9, 36, sigmoid),
        Dense(36, 72, sigmoid),
        Dense(72, 36, sigmoid),
        Dense(36, 1, sigmoid),
    )
else
    @load "nn.jld2" nn
end
nn = nn |> device

data = Flux.Data.DataLoader((X1, Z1), shuffle = true) |> device
ps = params(nn)
sqnorm(x) = sum(abs2, x)
#loss(x, y) = sum(abs2, nn(x) - y) / size(x, 2) #+ 1e-6 * sum(sqnorm, ps)
loss(x, y) = Flux.binarycrossentropy(nn(x), y)
cb = () -> println("loss: $(loss(X1, Z1))")
opt = ADAM()

@epochs 2 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

# accuracy
function accuracy(X, Z)
    Z1 = nn(X)

    ZA1 = [round(Z1[1, i]) for i in axes(Z1, 2)]
    ZA = [round(Z[1, i]) for i in axes(Z, 2)]

    accuracy = 0.0
    for i in eachindex(ZA)
        if ZA[i] == ZA1[i]
            accuracy += 1.0
        end
    end
    accuracy /= length(ZA)

    return accuracy
end

accuracy(X1, Z1)
accuracy(X2, Z2)

nn = nn |> cpu

@save "nn_scalar.jld2" nn
