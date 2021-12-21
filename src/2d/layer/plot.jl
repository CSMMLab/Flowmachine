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

function get_sol(ks, ctrkt, ctrns, ctrad)
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

    fc1 = (ctrkt[end÷2].f + ctrkt[end÷2+1].f) ./ 2
    hc1 = reduce_distribution(fc1, ks.vs.weights[:, 1, :], 2)
    fc3 = (ctrad[end÷2].f + ctrad[end÷2+1].f) ./ 2
    hc3 = reduce_distribution(fc3, ks.vs.weights[:, 1, :], 2)

    prim2 = (ctrns[end÷2].prim + ctrns[end÷2+1].prim) ./ 2
    sw2 = (ctrns[end÷2+1].w - ctrns[end÷2-1].w) ./ ks.ps.dx[end÷2] / 2
    τ = vhs_collision_time(prim2, ks.gas.μᵣ, ks.gas.ω)
    fc2 = chapman_enskog(ks.vs.u, ks.vs.v, ks.vs.w, prim2, sw2, zero(sw2), zero(sw2), ks.gas.K, τ)
    hc2 = reduce_distribution(fc2, ks.vs.weights[:, 1, :], 2)

    return solkt, solns, solad, hc1, hc2, hc3
end

###
# t = τ
###

begin
    @load "data/solkt_t.jld2" ctr
    ctrkt = deepcopy(ctr)
    @load "data/solns_t.jld2" ctr
    ctrns = deepcopy(ctr)
    @load "data/solkt_t.jld2" ctr
    ctrad = deepcopy(ctr)
end
solkt, solns, solad, hc1, hc2, hc3 = get_sol(ks, ctrkt, ctrns, ctrad)

begin
    begin
        plot(ks.ps.x[211:290], solkt[211:290, 1], lw=1.5, label="Kinetic", xlabel="x", ylabel="ρ")
        plot!(ks.ps.x[211:290], solns[211:290, 1], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[211:290], solad[211:290, 1], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_n_t.pdf")

    begin
        plot(ks.ps.x[211:290], solkt[211:290, 2], lw=1.5, label="Kinetic", xlabel="x", ylabel="U")
        plot!(ks.ps.x[211:290], solns[211:290, 2], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[211:290], solad[211:290, 2], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_u_t.pdf")

    begin
        plot(ks.ps.x[211:290], solkt[211:290, 3], lw=1.5, label="Kinetic", xlabel="x", ylabel="V")
        plot!(ks.ps.x[211:290], solns[211:290, 3], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[211:290], solad[211:290, 3], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_v_t.pdf")

    begin
        plot(ks.ps.x[211:290], solkt[211:290, 5], lw=1.5, label="Kinetic", xlabel="x", ylabel="T")
        plot!(ks.ps.x[211:290], solns[211:290, 5], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[211:290], solad[211:290, 5], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_t_t.pdf")
end

begin
    plot(ks.vs.v[1, :, 1], hc1, lw=1.5, label="Kinetic", xlabel="v", ylabel="f")
    plot!(ks.vs.v[1, :, 1], hc2, lw=1.5, line=:dash, label="NS")
    scatter!(ks.vs.v[1, :, 1], hc3, alpha=0.6, label="Adaptive")
end
savefig("figure/layer_f_t.pdf")

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

scatter(ks.ps.x[1:ks.ps.nx], rg_ref, alpha=0.7, label="NN", xlabel="x", ylabel="regime")
scatter!(ks.ps.x[1:ks.ps.nx], rg_kngll, alpha=0.7, label="KnGLL")
plot!(ks.ps.x[1:ks.ps.nx], rg_ref, lw=1.5, line=:dot, color=:gray27, label="True")
savefig("figure/layer_regime_t.pdf")

###
# t = 10τ
###

begin
    @load "data/solkt_10t.jld2" ctr
    ctrkt = deepcopy(ctr)
    @load "data/solns_10t.jld2" ctr
    ctrns = deepcopy(ctr)
    @load "data/solkt_10t.jld2" ctr
    ctrad = deepcopy(ctr)
end
solkt, solns, solad, hc1, hc2, hc3 = get_sol(ks, ctrkt, ctrns, ctrad)

begin
    begin
        plot(ks.ps.x[201:300], solkt[201:300, 1], lw=1.5, label="Kinetic", xlabel="x", ylabel="ρ")
        plot!(ks.ps.x[201:300], solns[201:300, 1], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[201:300], solad[201:300, 1], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_n_10t.pdf")

    begin
        plot(ks.ps.x[201:300], solkt[201:300, 2], lw=1.5, label="Kinetic", xlabel="x", ylabel="U")
        plot!(ks.ps.x[201:300], solns[201:300, 2], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[201:300], solad[201:300, 2], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_u_10t.pdf")

    begin
        plot(ks.ps.x[201:300], solkt[201:300, 3], lw=1.5, label="Kinetic", xlabel="x", ylabel="V")
        plot!(ks.ps.x[201:300], solns[201:300, 3], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[201:300], solad[201:300, 3], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_v_10t.pdf")

    begin
        plot(ks.ps.x[201:300], solkt[201:300, 5], lw=1.5, label="Kinetic", xlabel="x", ylabel="T")
        plot!(ks.ps.x[201:300], solns[201:300, 5], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[201:300], solad[201:300, 5], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_t_10t.pdf")
end

begin
    plot(ks.vs.v[1, :, 1], hc1, lw=1.5, label="Kinetic", xlabel="v", ylabel="f")
    plot!(ks.vs.v[1, :, 1], hc2, lw=1.5, line=:dash, label="NS")
    scatter!(ks.vs.v[1, :, 1], hc3, alpha=0.6, label="Adaptive")
end
savefig("figure/layer_f_10t.pdf")

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

begin
    scatter(ks.ps.x[1:ks.ps.nx], rg_ref, alpha=0.7, label="NN", xlabel="x", ylabel="regime")
    scatter!(ks.ps.x[1:ks.ps.nx], rg_kngll, alpha=0.7, label="KnGLL")
    plot!(ks.ps.x[1:ks.ps.nx], rg_ref, lw=1.5, line=:dot, color=:gray27, label="True")
end
savefig("figure/layer_regime_10t.pdf")

###
# t = 50τ
###

begin
    @load "data/solkt_50t.jld2" ctr
    ctrkt = deepcopy(ctr)
    @load "data/solns_50t.jld2" ctr
    ctrns = deepcopy(ctr)
    @load "data/solkt_50t.jld2" ctr
    ctrad = deepcopy(ctr)
end
solkt, solns, solad, hc1, hc2, hc3 = get_sol(ks, ctrkt, ctrns, ctrad)

begin
    begin
        plot(ks.ps.x[1:ks.ps.nx], solkt[:, 1], lw=1.5, label="Kinetic", xlabel="x", ylabel="ρ")
        plot!(ks.ps.x[1:ks.ps.nx], solns[:, 1], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[1:ks.ps.nx], solad[:, 1], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_n_50t.pdf")

    begin
        plot(ks.ps.x[1:ks.ps.nx], solkt[:, 2], lw=1.5, label="Kinetic", xlabel="x", ylabel="U")
        plot!(ks.ps.x[1:ks.ps.nx], solns[:, 2], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[1:ks.ps.nx], solad[:, 2], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_u_50t.pdf")

    begin
        plot(ks.ps.x[1:ks.ps.nx], solkt[:, 3], lw=1.5, label="Kinetic", xlabel="x", ylabel="V")
        plot!(ks.ps.x[1:ks.ps.nx], solns[:, 3], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[1:ks.ps.nx], solad[:, 3], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_v_50t.pdf")

    begin
        plot(ks.ps.x[1:ks.ps.nx], solkt[:, 5], lw=1.5, label="Kinetic", xlabel="x", ylabel="T")
        plot!(ks.ps.x[1:ks.ps.nx], solns[:, 5], lw=1.5, line=:dash, label="NS")
        scatter!(ks.ps.x[1:ks.ps.nx], solad[:, 5], alpha=0.6, label="Adaptive")
    end
    savefig("figure/layer_t_50t.pdf")
end

begin
    plot(ks.vs.v[1, :, 1], hc1, lw=1.5, label="Kinetic", xlabel="v", ylabel="f")
    plot!(ks.vs.v[1, :, 1], hc2, lw=1.5, line=:dash, label="NS")
    scatter!(ks.vs.v[1, :, 1], hc3, alpha=0.6, label="Adaptive")
end
savefig("figure/layer_f_50t.pdf")

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

begin
    scatter(ks.ps.x[1:ks.ps.nx], rg_ref, alpha=0.7, label="NN", xlabel="x", ylabel="regime")
    scatter!(ks.ps.x[1:ks.ps.nx], rg_kngll, alpha=0.7, label="KnGLL")
    plot!(ks.ps.x[1:ks.ps.nx], rg_ref, lw=1.5, line=:dot, color=:gray27, label="True")
end
savefig("figure/layer_regime_50t.pdf")
