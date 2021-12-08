using KitBase, Flux, Plots
using KitBase.JLD2
using Flux: @epochs
pyplot()

cd(@__DIR__)
#@load "../nn.jld2" nn
@load "../sampler/nn_rif.jld2" nn

@load "kn3_ma4_aux.jld2" X Y
X1, Y1 = deepcopy(X), deepcopy(Y)
@load "kn3_ma5_aux.jld2" X Y
X2, Y2 = deepcopy(X), deepcopy(Y)
X = hcat(X1, X2)
Y = hcat(Y1, Y2)

device = cpu
data = Flux.Data.DataLoader((X, Y), shuffle = true) |> device
ps = params(nn)
#sqnorm(x) = sum(abs2, x)
#loss(x, y) = sum(abs2, nn(x) - y) / size(x, 2) #+ 1e-6 * sum(sqnorm, ps)
loss(x, y) = Flux.binarycrossentropy(nn(x), y)
cb = () -> println("loss: $(loss(X, Y))")
opt = ADAM()

@epochs 10 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))

include("../common.jl")
accuracy(nn, X, Y)

@save "nn_pro.jld2" nn

@load "kn3ref.jld2" ctr
#@load "kn2ref.jld2" ctr
include("tools.jl")
begin
    set = Setup(
        case = "cylinder",
        space = "2d2f2v",
        boundary = ["maxwell", "extra", "mirror", "mirror"],
        limiter = "minmod",
        cfl = 0.5,
        maxTime = 15.0, # time
    )
    ps = CSpace2D(1.0, 6.0, 60, 0.0, π, 50, 1, 1)
    vs = VSpace2D(-10.0, 10.0, 48, -10.0, 10.0, 48)
    gas = Gas(Kn = 1e-3, Ma = 5.0, K = 1.0)
    
    prim0 = [1.0, 0.0, 0.0, 1.0]
    prim1 = [1.0, gas.Ma * sound_speed(1.0, gas.γ), 0.0, 1.0]
    fw = (args...) -> prim_conserve(prim1, gas.γ)
    ff = function(args...)
        prim = conserve_prim(fw(args...), gas.γ)
        h = maxwellian(vs.u, vs.v, prim)
        b = h .* gas.K / 2 / prim[end]
        return h, b
    end
    bc = function(x, y)
        if abs(x^2 + y^2 - 1) < 1e-3
            return prim0
        else
            return prim1
        end
    end
    ib = IB2F(fw, ff, bc)

    ks = SolverSet(set, ps, vs, gas, ib)
end

rmap = zeros(ks.ps.nr, ks.ps.nθ)
@inbounds Threads.@threads for j = 1:ks.ps.nθ
    rmap[1, j] = 1
    for i = 2:ks.ps.nr
        swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
        swx = (swx1 + swx2) ./ 2
        swy = (swy1 + swy2) ./ 2
        #sw = sqrt.(sw1.^2 + sw2.^2)

        rmap[i, j] = judge_regime(ks, ctr[i, j].h, ctr[i, j].prim, swx, swy)
    end
end

@inbounds Threads.@threads for j = 1:ks.ps.nθ
    rmap[1, j] = 1
    for i = 2:ks.ps.nr
        swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
        swx = (swx1 + swx2) ./ 2
        swy = (swy1 + swy2) ./ 2
        sw = sqrt.(swx.^2 + swy.^2)
        tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)

        rmap[i, j] = nn([ctr[i, j].w; sw; tau])[1] |> round |> Int
    end
end

begin
    contourf(
        ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        rmap[:, :],
        ratio = 1,
    )
end

savefig("nn.png")
