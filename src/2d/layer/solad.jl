"""
Note that ∇w takes its absolute value.
"""

using KitBase, Plots, Flux
using KitBase.JLD2
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress
cd(@__DIR__)
include("../common.jl")
@load "model/nn_layer.jld2" nn

function split_regime!(regime, regime0, ks, ctr, nn)
    regime[0] = 0
    regime[end] = 0

    @inbounds Threads.@threads for i = 1:ks.ps.nx
        wR = [ctr[i+1].w[1:3]; ctr[i+1].w[end]]
        wL = [ctr[i-1].w[1:3]; ctr[i-1].w[end]]
        w = [ctr[i].w[1:3]; ctr[i].w[end]]
        sw = (w .- wL) ./ ks.ps.dx[i]
        τ = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
        regime[i] = Int(round(nn([w; abs.(sw); τ])[1]))
        #=regime[i] = begin
            if abs(ctr[i].prim[2]) > 0.1
                1
            else
                Int(round(nn([w; sw; τ])[1]))
            end
        end=#

        if regime[i] == 1 && regime0[i] == 0
            Mu, Mv, Mw, t1, t2 = gauss_moments(ctr[i].prim, ks.gas.K)
            a = pdf_slope(ctr[i].prim, [sw[1:3]; 0.0; sw[end]], ks.gas.K)
            swt = -ctr[i].prim[1] .* moments_conserve_slope(a, Mu, Mv, Mw, 1, 0, 0)
            A = pdf_slope(ctr[i].prim, swt, ks.gas.K)
            ctr[i].f = chapman_enskog(ks.vs.u, ks.vs.v, ks.vs.w, ctr[i].prim, a, zero(a), zero(a), A, τ)
        end
    end

    regime0 .= regime

    return nothing
end

function up!(ks, ctr, face, dt, regime, p)
    kn_bzm, nm, phi, psi, phipsi = p

    res = zeros(5)
    avg = zeros(5)

    @inbounds Threads.@threads for i = 1:ks.ps.nx
        if regime[i] == 0
            KitBase.step!(
                face[i].fw,
                ctr[i].w,
                ctr[i].prim,
                face[i+1].fw,
                ks.gas.γ,
                ks.ps.dx[i],
                res,
                avg,
            )
        else
            KitBase.step!(
                face[i].fw,
                face[i].ff,
                ctr[i].w,
                ctr[i].prim,
                ctr[i].f,
                face[i+1].fw,
                face[i+1].ff,
                ks.gas.γ,
                kn_bzm,
                nm,
                phi,
                psi,
                phipsi,
                ks.ps.dx[i],
                dt,
                res,
                avg,
                :fsm,
            )
        end
    end
end

begin
    set = Setup(case = "layer", space = "1d1f3v", maxTime = 0.2, boundary = ["fix", "fix"])
    ps = PSpace1D(-0.5, 0.5, 500, 1)
    vs = VSpace3D(-6.0, 6.0, 28, -6.0, 6.0, 64, -6.0, 6.0, 28)
    #ps = PSpace1D(-0.5, 0.5, 300, 1)
    #vs = VSpace3D(-6.0, 6.0, 28, -6.0, 6.0, 48, -6.0, 6.0, 28)
    gas = begin
        Kn = 5e-3
        Gas(Kn = Kn, K = 0.0, fsm = fsm_kernel(vs, ref_vhs_vis(Kn, 1.0, 0.5)))
    end
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
    ctr, face = init_fvm(ks, ks.ps)
end

τ0 = vhs_collision_time(ctr[1].prim, ks.gas.μᵣ, ks.gas.ω)
tmax = 10τ0#50τ0
t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(tmax ÷ dt)
res = zero(ctr[1].w)
regime = ones(Int, axes(ks.ps.x))
regime0 = deepcopy(regime)

@time @showprogress for iter = 1:nt
    split_regime!(regime, regime0, ks, ctr, nn)

    @inbounds @threads for i = 1:ks.ps.nx+1
        if 1 ∉ (regime[i], regime[i-1])
            flux_gks!(
                face[i].fw,
                ctr[i-1].w .+ 0.5 .* ctr[i-1].sw .* ks.ps.dx[i-1],
                ctr[i].w .- 0.5 .* ctr[i].sw .* ks.ps.dx[i],
                ks.gas.K,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                dt,
                ks.ps.dx[i-1] / 2,
                ks.ps.dx[i] / 2,
                1.0,
                ctr[i-1].sw,
                ctr[i].sw,
            )
        else
            flux_kfvs!(
                face[i].fw,
                face[i].ff,
                ctr[i-1].f .+ 0.5 .* ctr[i-1].sf .* ks.ps.dx[i-1],
                ctr[i].f .- 0.5 .* ctr[i].sf .* ks.ps.dx[i],
                ks.vs.u,
                ks.vs.v,
                ks.vs.w,
                ks.vs.weights,
                dt,
                1.0,
                ctr[i-1].sf,
                ctr[i].sf,
            )
        end
    end

    up!(ks, ctr, face, dt, regime, ks.gas.fsm)

    #=global t += dt
    if abs(t - τ0) < dt
        @save "solad_t.jld2" ctr face
    elseif abs(t - 10 * τ0) < dt
        @save "solad_10t.jld2" ctr face
    end=#
end
#@save "solad_50t.jld2" ctr face
#=
sol = zeros(ks.ps.nx, 5)
for i in axes(sol, 1)
    sol[i, :] .= ctr[i].prim
    sol[i, end] = 1 / sol[i, end]
end

plot(ks.ps.x[1:ks.ps.nx], sol)
plot(ks.ps.x[1:ks.ps.nx], regime[1:ks.ps.nx])
=#

"""
adaptive: 623.054187 seconds (134.63 M allocations: 1.365 TiB, 1.29% gc time, 3.23% compilation time)
kinetic: 1985.342956 seconds (400.14 M allocations: 15.922 TiB, 1.82% gc time, 0.46% compilation time)
"""
