using KitBase, Plots, Flux
using KitBase.JLD2

cd(@__DIR__)
include("../common.jl")

begin
    set = Setup(case = "layer", space = "1d1f3v", maxTime = 0.2, boundary = ["fix", "fix"], cfl = 0.5)
    ps = PSpace1D(-0.5, 0.5, 500, 1)
    vs = VSpace3D(-6.0, 6.0, 28, -6.0, 6.0, 64, -6.0, 6.0, 28)
    gas = Gas(Kn = 5e-3, K = 0.0)
    fw = function(x)
        prim = zeros(5)
        if x <= 0
            prim .= [1.0, 0.0, 1.0, 0.0, 1.0]
        else
            prim .= [1.0, 0.0, -1.0, 0.0, 2.0]
        end

        return prim_conserve(prim, ks.gas.γ)
    end
    ib = IB1F(fw, vs, gas)
    ks = SolverSet(set, ps, vs, gas, ib)
end

begin
    @load "solkt_50t.jld2" ctr
    ctrkt = deepcopy(ctr)
    @load "solns_50t.jld2" ctr
    ctrns = deepcopy(ctr)
    @load "solad_50t.jld2" ctr
    ctrad = deepcopy(ctr)
end

# solution field
solkt = zeros(ks.ps.nx, 5)
solns = zero(solkt)
solad = zero(solkt)
for i in axes(solkt, 1)
    solkt[i, :] .= ctrkt[i].prim
    solns[i, :] .= ctrns[i].prim
    solad[i, :] .= ctrad[i].prim
    solkt[i, end] = 1 / solkt[i, end]
    solns[i, end] = 1 / solns[i, end]
    solad[i, end] = 1 / solad[i, end]
end

begin
    plot(ks.ps.x[1:ks.ps.nx], solkt[:, 1])
    plot!(ks.ps.x[1:ks.ps.nx], solns[:, 1])
    plot!(ks.ps.x[1:ks.ps.nx], solad[:, 1])
end

# distribution function
fc = (ctr[end÷2].f + ctr[end÷2+1].f) ./ 2
hc = reduce_distribution(fc, ks.vs.weights[:, 1, :], 2)
plot(ks.vs.v[1, :, 1], hc)



rg_ref = zeros(ks.ps.nx)
@inbounds Threads.@threads for i = 1:ks.ps.nx
    swx = (ctrkt[i+1].w - ctrkt[i-1].w) / (1e-6 + ks.ps.x[i+1] - ks.ps.x[i-1])
    rg_ref[i] = judge_regime(ks, ctrkt[i].f, ctrkt[i].prim, swx)
end

rg_kngll = zeros(ks.ps.nx)
@inbounds Threads.@threads for i = 1:ks.ps.nx
    swx = (ctrkt[i+1].w - ctrkt[i-1].w) / (1e-6 + ks.ps.x[i+1] - ks.ps.x[i-1])

    L = abs(ctrkt[i].w[1] / swx[1])
    ℓ = (1 / ctrkt[i].prim[end])^ks.gas.ω / ctrkt[i].prim[1] * sqrt(ctrkt[i].prim[end]) * ks.gas.Kn
    rg_kngll[i] = ifelse(ℓ / L > 0.05, 1, 0)
end

plot(ks.ps.x[1:ks.ps.nx], rg_ref)
plot!(ks.ps.x[1:ks.ps.nx], rg_kngll)
