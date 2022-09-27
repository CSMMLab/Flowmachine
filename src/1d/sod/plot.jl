using Plots
using JLD2
using KitBase
using KitML
using LinearAlgebra
using NPZ
#using Kinetic, Plots, LinearAlgebra
#using KitBase.JLD2
#using Flux: onecold
cd(@__DIR__)


begin
    set = Setup(case = "sod", space = "1d2f1v", maxTime = 0.15)
    ps = PSpace1D(0.0, 1.0, 200, 1)
    vs = VSpace1D(-5.0, 5.0, 100)
    #gas = Gas(Kn = 1e-4, K = 2, γ = 5/3)
    gas = Gas(Kn = 1e-3, K = 2, γ = 5/3)
    #gas = Gas(Kn = 1e-2, K = 2, γ = 5/3)
    ib = IB2F(KitBase.ib_sod(set, ps, vs, gas)...)
    ks = SolverSet(set, ps, vs, gas, ib)
end

# Kn = 1e-4
ctrs = []
regime_name = "1e-4"
begin
    @load "kn=1e-4/nn.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-4/kngll.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-4/pure_kinetic.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-4/pure_ns.jld2" ctr
    push!(ctrs, ctr)
end

# Kn = 1e-3
ctrs = []
regime_name = "1e-3"
begin
    @load "kn=1e-3/nn.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-3/kngll.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-3/pure_kinetic.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-3/pure_ns.jld2" ctr
    push!(ctrs, ctr)
end

# Kn = 1e-2
ctrs = []
regime_name = "1e-2"
begin
    @load "kn=1e-2/nn.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-2/kngll.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-2/pure_kinetic.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-3/pure_ns.jld2" ctr
    push!(ctrs, ctr)
    @load "kn=1e-2/pure_kinetic.jld2" ctr
end

sols = []
begin
    for idx in eachindex(ctrs)
        sol = zeros(ks.ps.nx, 3)
        for i in axes(sol, 1)
            sol[i, :] .= ctrs[idx][i].prim
            sol[i, end] = 1 / sol[i, end]
        end
        push!(sols, sol)
    end
end

function regime_data_1d(
    w,
    sw,
    f,
    u,
    K::Real,
    Kn::Real,
    μ = ref_vhs_vis(Kn, 1.0, 0.5),
    ω = 0.81,
)
    γ = heat_capacity_ratio(K, 1)
    prim = conserve_prim(w, γ)

    Mu, Mxi, _, _1 = KitBase.gauss_moments(prim, K)
    a = KitBase.pdf_slope(prim, sw, K)
    swt = -prim[1] .* KitBase.moments_conserve_slope(a, Mu, Mxi, 1)
    A = KitBase.pdf_slope(prim, swt, K)
    tau = KitBase.vhs_collision_time(prim, μ, ω)
    fr = KitBase.chapman_enskog(u, prim, a, A, tau)
    L = norm((f .- fr) ./ prim[1])

    x = [w; sw; tau]
    y = ifelse(L <= 0.005, 0.0, 1.0)
    return x, y
end

regime_data_1d(ks::SolverSet, w, sw, f) = regime_data_1d(w, sw, f, ks.vs.u, ks.gas.K, ks.gas.Kn)


begin
    ctr = ctrs[3]

    rg_ce = zeros(Int, ks.ps.nx)
    for i = 1:ks.ps.nx
        sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
        x, y = regime_data_1d(ks, ctr[i].w, sw, ctr[i].h)
        rg_ce[i] = floor(y) |> Int
    end

    rg_kn = zeros(Int, ks.ps.nx)
    KnGLL = zeros(ks.ps.nx)
    for i = 1:ks.ps.nx
        sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
        L = abs(ctr[i].w[1] / sw[1])
        ℓ = (1/ctr[i].prim[end])^ks.gas.ω / ctr[i].prim[1] * sqrt(ctr[i].prim[end]) * ks.gas.Kn

        KnGLL[i] = ℓ / L
        rg_kn[i] = ifelse(KnGLL[i] >= 0.05, 1, 0)
    end
end

scatter(ks.ps.x[1:ks.ps.nx], rg_ce, alpha=0.7, label="NN", xlabel="x", ylabel="regime")
scatter!(ks.ps.x[1:ks.ps.nx], rg_kn, alpha=0.7, label="KnGLL")
plot!(ks.ps.x[1:ks.ps.nx], rg_ce, lw=1.5, line=:dot, color=:gray27, label="True", xlabel="x", ylabel="regime")

npzwrite("sod_x.npy",ks.ps.x[1:ks.ps.nx])
npzwrite("sod_Kn_"*regime_name*"_NN.npy",rg_ce)
npzwrite("sod_Kn_"*regime_name*"_KnGLL.npy",rg_kn)
npzwrite("sod_Kn_"*regime_name*"_True.npy",rg_ce)

#savefig("figs/kn4_regime.pdf")
#savefig("figs/kn3_regime.pdf")
#savefig("figs/kn2_regime.pdf")

function curve(idx, s, lgd=:topright; ns=true)
    fig = plot(ks.ps.x[idx], sols[1][idx, s], lw=1.5, label="NN", legend=lgd)
    plot!(fig, ks.ps.x[idx], sols[2][idx, s], lw=1.5, line=:dash, label="KnGLL")
    plot!(fig, ks.ps.x[idx], sols[3][idx, s], lw=1.5, line=:dashdot, label="Kinetic")    
    ns && plot!(fig, ks.ps.x[idx], sols[4][idx, s], lw=2, line=:dot, color=:gray27, label="NS")

    npzwrite("sod_sol_x_"*string(s)*".npy",ks.ps.x[idx])
    
    npzwrite("sod_sol_Kn_"*regime_name*"_NN_"*string(s)*".npy",sols[1][idx, s])
    npzwrite("sod_sol_Kn_"*regime_name*"_KnGLL_"*string(s)*".npy",sols[2][idx, s])
    npzwrite("sod_sol_Kn_"*regime_name*"_Kinetic_"*string(s)*".npy",sols[3][idx, s])
    npzwrite("sod_sol_Kn_"*regime_name*"_NS_"*string(s)*".npy",sols[4][idx, s])

    xlabel!(fig, "x")
    if s == 1
        ylabel!(fig, "ρ")
    elseif s == 2
        ylabel!(fig, "U")
    elseif s == 3
        ylabel!(fig, "T")
    end

    return fig
end

idx = 1:ks.ps.nx
curve(idx, 1)
curve(idx, 2)
curve(idx, 3)

#savefig("kn4_n.pdf")
#savefig("kn3_n.pdf")
#curve(idx, 1; ns=false)
#savefig("kn2_n.pdf")

curve(idx, 3)
#savefig("kn4_t.pdf")
#savefig("kn3_t.pdf")
#curve(idx, 3; ns=false)
#savefig("kn2_t.pdf")

idx = ks.ps.nx÷5*3+1:ks.ps.nx÷5*4
curve(idx, 1)
#savefig("kn4_n_zoom.pdf")
#savefig("kn3_n_zoom.pdf")
#curve(idx, 1; ns=false)
#savefig("kn2_n_zoom.pdf")

curve(idx, 3, :topleft)
#savefig("kn4_t_zoom.pdf")
#savefig("kn3_t_zoom.pdf")
#curve(idx, 3, :topleft; ns=false)
#savefig("kn2_t_zoom.pdf")
