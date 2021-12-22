using Kinetic, Plots, Flux
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress
using Base.Threads: @threads
cd(@__DIR__)

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

function up!(ks, ctr, face, dt, regime, p)
    kn_bzm, nm, phi, psi, phipsi = p

    res = zeros(5)
    avg = zeros(5)

    @inbounds Threads.@threads for i = 1:ks.ps.nx
        #=KitBase.step!(
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
        )=#
        sbr!(
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
        )
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
    ps = PSpace1D(0, 1, 100, 1)
    vs = VSpace3D(-6.0, 6.0, 64, -6.0, 6.0, 28, -6.0, 6.0, 28)
    gas = begin
        Kn = 1e-2
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
    @inbounds @threads for i = 1:ks.ps.nx+1
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

    up!(ks, ctr, face, dt, regime, ks.gas.fsm)
end

"""
Kn = 1e-4
422.120548 seconds (41.79 M allocations: 1.842 TiB, 2.94% gc time, 0.11% compilation time)

Kn = 1e-3
420.792971 seconds (41.64 M allocations: 1.842 TiB, 2.93% gc time, 0.09% compilation time)

Kn = 1e-2
421.556153 seconds (41.96 M allocations: 1.842 TiB, 2.45% gc time, 0.08% compilation time)
"""