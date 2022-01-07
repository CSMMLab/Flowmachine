using Kinetic, Plots, Flux
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads
cd(@__DIR__)
@load "nn_sod.jld2" nn

function sbr!(
    fwL::T1,
    ffL::T2,
    w::T3,
    prim::T3,
    f::T4,
    fwR::T1,
    ffR::T2,
    uVelo::T5,
    vVelo::T5,
    wVelo::T5,
    γ,
    μᵣ,
    ω,
    Kn_bz,
    nm,
    phi,
    psi,
    phipsi,
    dx,
    dt,
    RES,
    AVG,
    collision = :fsm::Symbol,
) where {T1,T2,T3,T4,T5}

    w_old = deepcopy(w)
    @. w += (fwL - fwR) / dx
    prim .= conserve_prim(w, γ)

    M = maxwellian(uVelo, vVelo, wVelo, prim)
    τ = vhs_collision_time(prim, μᵣ, ω)

    @. RES += (w - w_old)^2
    @. AVG += abs(w)

    Q = zero(f[:, :, :])
    boltzmann_fft!(Q, f, Kn_bz, nm, phi, psi, phipsi)

    for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
        f[i, j, k] =
            (
                f[i, j, k] +
                (ffL[i, j, k] - ffR[i, j, k]) / dx +
                dt * (M[i, j, k] / τ * (1 - exp(-dt / τ)) + Q[i, j, k] * exp(-dt / τ))
            ) / (1.0 + dt / τ * (1 - exp(-dt / τ)))
    end

end

function split_regime!(regime, regime0, ks, ctr, nn)
    regime[0] = 0
    regime[end] = 0

    @inbounds @threads for i = 1:ks.ps.nx
        wR = [ctr[i+1].w[1:2]; ctr[i+1].w[end]]
        wL = [ctr[i-1].w[1:2]; ctr[i-1].w[end]]
        w = [ctr[i].w[1:2]; ctr[i].w[end]]
        sw = (w .- wL) ./ ks.ps.dx[i]
        τ = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
        regime[i] = Int(round(nn([w; abs.(sw); τ])[1]))

        if regime[i] == 1 && regime0[i] == 0
            Mu, Mv, Mw, t1, t2 = gauss_moments(ctr[i].prim, ks.gas.K)
            a = pdf_slope(ctr[i].prim, [sw[1:2]; zeros(2); sw[end]], ks.gas.K)
            swt = -ctr[i].prim[1] .* moments_conserve_slope(a, Mu, Mv, Mw, 1, 0, 0)
            A = pdf_slope(ctr[i].prim, swt, ks.gas.K)
            ctr[i].f = chapman_enskog(
                ks.vs.u,
                ks.vs.v,
                ks.vs.w,
                ctr[i].prim,
                a,
                zero(a),
                zero(a),
                A,
                τ,
            )
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
            #=sbr!(
                face[i].fw,
                face[i].ff,
                ctr[i].w,
                ctr[i].prim,
                ctr[i].f,
                face[i+1].fw,
                face[i+1].ff,
                ks.vs.u,
                ks.vs.v,
                ks.vs.w,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
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
            )=#
        end
    end
end

begin
    set = Setup(
        case = "sod",
        space = "1d1f3v",
        maxTime = 0.15,
        collision = "fsm",
        boundary = ["fix", "fix"],
    )
    ps = PSpace1D(0, 1, 200, 1)
    vs = VSpace3D(-6.0, 6.0, 64, -6.0, 6.0, 28, -6.0, 6.0, 28)
    gas = begin
        Kn = 1e-4
        Gas(Kn = Kn, K = 0.0, fsm = fsm_kernel(vs, ref_vhs_vis(Kn, 1.0, 0.5)))
    end
    ib = IB1F(ib_sod(set, ps, vs, gas)...)
    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, face = init_fvm(ks)
end

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt)
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
end

plot(ks, ctr)
plot(regime[1:ks.ps.nx])

"""
Kn = 1e-4
7.856452 seconds (6.88 M allocations: 9.526 GiB, 3.15% gc time, 7.59% compilation time) ~ 100
81.253323 seconds (35.72 M allocations: 100.876 GiB, 2.06% gc time, 4.02% compilation time) ~ 200

Kn = 1e-3
171.219305 seconds (20.46 M allocations: 235.016 GiB, 3.12% gc time, 1.61% compilation time) ~ 100
514.904113 seconds (36.01 M allocations: 713.923 GiB, 2.51% gc time, 0.07% compilation time) ~ 200

Kn = 1e-2
378.002585 seconds (24.71 M allocations: 971.264 GiB, 2.83% gc time, 0.11% compilation time) ~ 100

"""