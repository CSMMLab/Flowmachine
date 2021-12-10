using Kinetic, Plots, Flux
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

function split_regime!(regime, regime0, ks, ctr, nn)
    @inbounds Threads.@threads for i in eachindex(regime)
        if i == 0
            sw = (ctr[i+1].w .- ctr[i].w) / ks.ps.dx[i]
        else
            sw = (ctr[i].w .- ctr[i-1].w) / ks.ps.dx[i]
        end
        tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
        regime[i] = Int(round(nn([ctr[i].w[1:2]; ctr[i].w[end]; sw[1:2]; sw[end]; tau])[1]))

        if regime[i] == 1 && regime0[i] == 0
            Mu, Mv, Mw, _, _1 = gauss_moments(ctr[i].prim, ks.gas.K)
            a = pdf_slope(ctr[i].prim, sw, ks.gas.K)
            swt = -ctr[i].prim[1] .* moments_conserve_slope(a, Mu, Mv, Mw, 1, 0, 0)
            A = pdf_slope(ctr[i].prim, swt, ks.gas.K)
            tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
            ctr[i].f .= chapman_enskog(ks.vs.u, ks.vs.v, ks.vs.w, ctr[i].prim, a, zero(a), zero(a), A, tau)
            #ctr[i].f .= maxwellian(ks.vs.u, ks.vs.v, ks.vs.w, ctr[i].prim)
        end
    end

    regime0 .= regime

    return nothing
end

function kngll_regime!(regime, regime0, ks, ctr, nn)
    @inbounds Threads.@threads for i in eachindex(regime)
        if i == 0
            sw = (ctr[i+1].w .- ctr[i].w) / ks.ps.dx[i]
        else
            sw = (ctr[i].w .- ctr[i-1].w) / ks.ps.dx[i]
        end

        L = abs(ctr[i].w[1] / sw[1])
        ℓ = (1/ctr[i].prim[end])^ks.gas.ω / ctr[i].prim[1] * sqrt(ctr[i].prim[end]) * ks.gas.Kn
        regime[i] = ifelse(ℓ / L >= 0.01, 1, 0)

        if regime[i] == 1 && regime0[i] == 0
            Mu, Mv, Mw, _, _1 = gauss_moments(ctr[i].prim, ks.gas.K)
            a = pdf_slope(ctr[i].prim, sw, ks.gas.K)
            swt = -ctr[i].prim[1] .* moments_conserve_slope(a, Mu, Mv, Mw, 1, 0, 0)
            A = pdf_slope(ctr[i].prim, swt, ks.gas.K)
            tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
            ctr[i].f .= chapman_enskog(ks.vs.u, ks.vs.v, ks.vs.w, ctr[i].prim, a, zero(a), zero(a), A, tau)
        end
    end

    regime0 .= regime

    return nothing
end

function up!(KS, ctr, regime, p)
    kn_bz, nm, phi, psi, phipsi = p

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
                KS.gas.γ,
                kn_bz,
                nm,
                phi,
                psi,
                phipsi,
                KS.ps.dx[i],
                dt,
                res,
                avg,
                :fsm,
            )
        end
    end
end

cd(@__DIR__)
#@load "../nn_scalar.jld2" nn
@load "/home2/vavrines/Coding/Flowmachine/src/1d/sampler/nn_rif.jld2" nn

set = Setup(case = "sod", space = "1d1f3v", collision = "boltzmann", maxTime = 0.15, cfl = 0.5)
ps = PSpace1D(0.0, 1.0, 100, 1)
vs = VSpace3D(-8.0, 8.0, 48, -8.0, 8.0, 28, -8.0, 8.0, 28)
gas = Gas(Kn = 1e-3, Pr = 2/3, K = 0.0)
ib = IB1F(ib_sod(set, ps, vs, gas)...)
ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks, ks.ps, :dynamic_array; structarray = false)

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

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ctr[1].w)
regime = ones(Int, axes(ks.ps.x))
regime0 = deepcopy(regime)

@showprogress for iter = 1:20#nt
    #split_regime!(regime, regime0, ks, ctr, nn)
    kngll_regime!(regime, regime0, ks, ctr, nn)

    for i in eachindex(face)
        if 1 ∉ (regime[i], regime[i-1])
            flux_gks!(
                face[i].fw,
                ctr[i-1].w .+ ctr[i-1].sw .* ks.ps.dx[i-1] / 2,
                ctr[i].w .- ctr[i].sw .* ks.ps.dx[i] / 2,
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
                ctr[i-1].f .+ ctr[i-1].sf .* ks.ps.dx[i-1] / 2,
                ctr[i].f .- ctr[i].sf .* ks.ps.dx[i] / 2,
                ks.vs.u,
                ks.vs.v,
                ks.vs.w,
                ks.vs.weights,
                dt,
                ctr[i-1].sf,
                ctr[i].sf,
            )
        end
    end
    
    #KitBase.update!(ks, ctr, face, dt, res; coll = Symbol(ks.set.collision), bc = Symbol(ks.set.boundary))
    up!(ks, ctr, regime, p)

    t += dt
end

plot(ks, ctr)
plot!(ks.ps.x[1:ks.ps.nx], regime[1:ks.ps.nx])

kngll_regime!(regime, regime0, ks, ctr, nn)
