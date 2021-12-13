using KitBase, KitBase.JLD2, KitBase.WriteVTK
#using Plots; pyplot()
using PyPlot

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

cd(@__DIR__)
#@load "kn3ref.jld2" ctr
@load "kn2ref.jld2" ctr

begin
    sol = zeros(ks.ps.nr, ks.ps.nθ, 4)
    for i in axes(sol, 1), j in axes(sol, 2)
        sol[i, j, :] .= ctr[i, j].prim
        sol[i, j, end] = 1 / sol[i, j, end]
    end
end

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
    end
end

begin
    close("all")
    fig = figure("contour", figsize=(8, 4))
    PyPlot.contourf(
        ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        #rg_ref,
        rg_kngll,
        levels = 20,
        cmap = ColorMap("inferno"),
    )
    colorbar()
    xlabel("x")
    ylabel("y")
    tight_layout()
    xlim(-6, 6)
    ylim(0, 6)
    display(fig)
end
