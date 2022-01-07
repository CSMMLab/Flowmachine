using Kinetic, Plots, Flux
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads
using Flux: @epochs
cd(@__DIR__)

X = Float32.([[1e-4, 0.0, 1e-4]; zeros(3); 1.0])
Y = Float32.(0.0)

###
# Kn = 1e-4
###

begin
    set = Setup(case = "sod", space = "1d1f3v", collision = "bgk", maxTime = 0.15, boundary = ["fix", "fix"], cfl = 0.5)
    ps = PSpace1D(0, 1, 100, 1)
    vs = VSpace3D(-6.0, 6.0, 64, -6.0, 6.0, 28, -6.0, 6.0, 28)
    gas = Gas(Kn = 1e-4, K = 0.0)
    ib = IB1F(ib_sod(set, ps, vs, gas)...)
    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, face = init_fvm(ks)
end

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt)
res = zero(ctr[1].w)

@showprogress for iter = 1:nt
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if iter % 3 == 0
        for i = 1:ks.ps.nx
            sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
            x, y = regime_data(ks, ctr[i].w, sw, zero(sw), zero(sw), ctr[i].f)
            global X = hcat(X, [x[1:2]; x[5:7]; x[10:11]])
            global Y = hcat(Y, y)
        end
    end
end

###
# Kn = 1e-3
###

begin
    gas = Gas(Kn = 1e-3, K = 0.0)
    ib = IB1F(ib_sod(set, ps, vs, gas)...)
    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, face = init_fvm(ks)
end

@showprogress for iter = 1:nt
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if iter % 3 == 0
        for i = 1:ks.ps.nx
            sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
            x, y = regime_data(ks, ctr[i].w, sw, zero(sw), zero(sw), ctr[i].f)
            global X = hcat(X, [x[1:2]; x[5:7]; x[10:11]])
            global Y = hcat(Y, y)
        end
    end
end

###
# Kn = 1e-2
###

begin
    gas = Gas(Kn = 1e-2, K = 0.0)
    ib = IB1F(ib_sod(set, ps, vs, gas)...)
    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, face = init_fvm(ks)
end

@showprogress for iter = 1:nt
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if iter % 3 == 0
        for i = 1:ks.ps.nx
            sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
            x, y = regime_data(ks, ctr[i].w, sw, zero(sw), zero(sw), ctr[i].f)
            global X = hcat(X, [x[1:2]; x[5:7]; x[10:11]])
            global Y = hcat(Y, y)
        end
    end
end

@save "data_sod.jld2" X Y

###
# model
###

#@load "../../nn_scalar.jld2" nn
#@load "nn_sod.jld2" nn
@load "data_sod.jld2" X Y

nn = Chain(
    Dense(7, 28, sigmoid),
    Dense(28, 56, sigmoid),
    Dense(56, 28, sigmoid),
    Dense(28, 1, sigmoid),
)

data = Flux.Data.DataLoader((X, Y), shuffle = true)
ps = params(nn)
loss(x, y) = Flux.binarycrossentropy(nn(x), y)
cb = () -> println("loss: $(loss(X, Y))")
opt = ADAM()

@epochs 5 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

#--- test ---#

function accuracy(nn, X, Z)
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

accuracy(nn, X, Y)

function split_regime!(regime, ks, ctr, nn)
    regime[0] = 0
    regime[end] = 0

    @inbounds @threads for i = 1:ks.ps.nx
        wR = [ctr[i+1].w[1:2]; ctr[i+1].w[end]]
        wL = [ctr[i-1].w[1:2]; ctr[i-1].w[end]]
        w = [ctr[i].w[1:2]; ctr[i].w[end]]
        sw = (wR .- wL) ./ ks.ps.dx[i] ./ 2
        τ = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
        regime[i] = Int(round(nn([w; abs.(sw); τ])[1]))
    end

    return nothing
end

regime = ones(Int, axes(ks.ps.x))
split_regime!(regime, ks, ctr, nn)

plot(ks, ctr)
plot!(ks.ps.x[1:ks.ps.nx], regime[1:ks.ps.nx])


@save "nn_sod.jld2" nn
