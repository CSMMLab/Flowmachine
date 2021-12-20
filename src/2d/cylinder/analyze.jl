using Kinetic, Flux
using KitBase.JLD2, KitBase.WriteVTK
using Flux: @epochs
import PyPlot as plt

cd(@__DIR__)
include("../common.jl")

begin
    set = Setup(
        case = "cylinder",
        space = "2d2f2v",
        boundary = ["maxwell", "extra", "mirror", "mirror"],
        limiter = "minmod",
        cfl = 0.5,
        maxTime = 10.0, # time
    )
    ps = CSpace2D(1.0, 6.0, 60, 0.0, π, 50, 1, 1)
    vs = VSpace2D(-10.0, 10.0, 48, -10.0, 10.0, 48)
    #gas = Gas(Kn = 1e-3, Ma = 5.0, K = 1.0)
    gas = Gas(Kn = 1e-2, Ma = 5.0, K = 1.0)

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

#@load "kn3ref.jld2" ctr
@load "kn2ref.jld2" ctr

rg_ref = zeros(ks.ps.nr, ks.ps.nθ)
@inbounds Threads.@threads for j = 1:ks.ps.nθ
    rg_ref[1, j] = 1
    for i = 2:ks.ps.nr
        swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
        swx = (swx1 + swx2) ./ 2
        swy = (swy1 + swy2) ./ 2

        rg_ref[i, j] = judge_regime(ks, ctr[i, j].h, ctr[i, j].prim, swx, swy)
    end
end

rg_kngll = zeros(ks.ps.nr, ks.ps.nθ)
@inbounds Threads.@threads for j = 1:ks.ps.nθ
    rg_kngll[1, j] = 1
    for i = 2:ks.ps.nr
        swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
        swx = (swx1 + swx2) ./ 2
        swy = (swy1 + swy2) ./ 2
        sw = @. sqrt(swx^2 + swy^2)

        L = abs(ctr[i, j].w[1] / sw[1])
        ℓ = (1 / ctr[i, j].prim[end])^ks.gas.ω / ctr[i, j].prim[1] * sqrt(ctr[i, j].prim[end]) * ks.gas.Kn
        rg_kngll[i, j] = ifelse(ℓ / L > 0.05, 1, 0)
        #rg_kngll[i, j] = ifelse(ℓ / L > 0.01, 1, 0)
    end
end

begin
    plt.close("all")
    fig = plt.figure("contour", figsize=(8, 4))
    plt.contourf(
        ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        #rg_ref,
        rg_kngll,
        #rg_nn,
        levels = 20,
        cmap = plt.ColorMap("inferno"),
    )
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.xlim(-6, 6)
    plt.ylim(0, 6)
    plt.display(fig)
end
#fig.savefig("cylinder_rgref_kn3.pdf")
#fig.savefig("cylinder_rgkngll_kn2_c001.pdf")

@load "nn_pro.jld2" nn

global X = Float32.([[1.0, 0.0, 0.0, 1.0]; zeros(4); 1e-4])
global Y = Float32(0.0)
for j = 1:ks.ps.nθ, i = 2:ks.ps.nr
    swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
    swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
    swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
    swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
    swx = (swx1 + swx2) ./ 2
    swy = (swy1 + swy2) ./ 2

    x, y = regime_data(ks, ctr[i, j].w, swx, swy, ctr[i, j].h)
    global X = hcat(X, x)
    global Y = hcat(Y, y)
end

data = Flux.Data.DataLoader((X, Y), shuffle = true)
ps = params(nn)
loss(x, y) = Flux.binarycrossentropy(nn(x), y)
cb = () -> println("loss: $(loss(X, Y))")
opt = ADAM()

@epochs 20 Flux.train!(loss, ps, data, opt, cb = Flux.throttle(cb, 1))
accuracy(nn, X, Y)

rg_nn = zeros(ks.ps.nr, ks.ps.nθ)
@inbounds Threads.@threads for j = 1:ks.ps.nθ
    rg_nn[1, j] = 1
    for i = 2:ks.ps.nr
        swx1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.x[i+1, j] - ks.ps.x[i-1, j])
        swy1 = (ctr[i+1, j].w - ctr[i-1, j].w) / (1e-6 + ks.ps.y[i+1, j] - ks.ps.y[i-1, j])
        swx2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.x[i, j+1] - ks.ps.x[i, j-1])
        swy2 = (ctr[i, j+1].w - ctr[i, j-1].w) / (1e-6 + ks.ps.y[i, j+1] - ks.ps.y[i, j-1])
        swx = (swx1 + swx2) ./ 2
        swy = (swy1 + swy2) ./ 2
        sw = sqrt.(swx.^2 + swy.^2)
        tau = vhs_collision_time(ctr[i, j].prim, ks.gas.μᵣ, ks.gas.ω)

        rg_nn[i, j] = nn([ctr[i, j].w; sw; tau])[1] |> round |> Int
    end
end

begin
    plt.close("all")
    fig = plt.figure("contour", figsize=(8, 4))
    plt.contourf(
        ks.ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ks.ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        rg_nn,
        levels = 20,
        cmap = plt.ColorMap("inferno"),
    )
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.xlim(-6, 6)
    plt.ylim(0, 6)
    plt.display(fig)
end
#fig.savefig("cylinder_rgnn_kn3.pdf")
fig.savefig("cylinder_rgnn_kn2.pdf")
