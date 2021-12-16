using KitBase, Plots, Flux
using KitBase.JLD2
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress
cd(@__DIR__)
include("../common.jl")
@load "nn_pro.jld2" nn

function split_regime!(regime, regime0, ks, ctr, nn)
    regime[0] = 0
    regime[end] = 0

    @inbounds Threads.@threads for i = 1:ks.ps.nx
        sw = (ctr[i+1].w .- ctr[i-1].w) / (2.0 * ks.ps.dx[i])
        τ = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
        regime[i] = Int(round(nn([ctr[i].w[1:3]; ctr[i].w[end]; sw[1:3]; sw[end]; τ])[1]))

        if regime[i] == 1 && regime0[i] == 0
            Mu, Mv, Mw, _, _1 = gauss_moments(ctr[i].prim, ks.gas.K)
            a = pdf_slope(ctr[i].prim, sw, ks.gas.K)
            swt = -ctr[i].prim[1] .* moments_conserve_slope(a, Mu, Mv, Mw, 1, 0, 0)
            A = pdf_slope(ctr[i].prim, swt, ks.gas.K)
            ctr[i].f = chapman_enskog(ks.vs.u, ctr[i].prim, a, A, τ)
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
    ps = PSpace1D(-1.0, 1.0, 1000, 1)
    vs = VSpace3D(-6.0, 6.0, 28, -6.0, 6.0, 72, -6.0, 6.0, 28)
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
    ctr, face = init_fvm(ks, ks.ps)
end

begin
    kn_bzm = hs_boltz_kn(ks.gas.μᵣ, 1.0)
    phi, psi, phipsi = kernel_mode(
        5,
        ks.vs.u1,
        ks.vs.v1,
        ks.vs.w1,
        ks.vs.du[1, 1, 1],
        ks.vs.dv[1, 1, 1],
        ks.vs.dw[1, 1, 1],
        ks.vs.nu,
        ks.vs.nv,
        ks.vs.nw,
        1.0,
    )
    p = (kn_bzm, 5, phi, psi, phipsi)
end

τ0 = vhs_collision_time(ctr[1].prim, ks.gas.μᵣ, ks.gas.ω)
tmax = 100 * τ0
t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(tmax ÷ dt)
res = zero(ctr[1].w)
regime = ones(Int, axes(ks.ps.x))
regime0 = deepcopy(regime)

@showprogress for iter = 1:nt
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

    up!(ks, ctr, face, dt, regime, p)

    global t += dt

    #=if abs(t - τ0) < dt
        @save "sol_t.jld2" ctr face
    elseif abs(t - 10 * τ0) < dt
        @save "sol_10t.jld2" ctr face
    end=#
end

