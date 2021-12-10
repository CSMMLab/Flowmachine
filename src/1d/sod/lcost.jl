using Kinetic, Plots, Flux
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress
using BenchmarkTools

function split_regime!(regime, regime0, ks, ctr, nn)
    @inbounds Threads.@threads for i in eachindex(regime)
        if i == 0
            sw = (ctr[i+1].w .- ctr[i].w) / ks.ps.dx[i]
        else
            sw = (ctr[i].w .- ctr[i-1].w) / ks.ps.dx[i]
        end
        tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
        regime[i] = Int(round(nn([ctr[i].w; sw; tau])[1]))

        if regime[i] == 1 && regime0[i] == 0
            Mu, Mxi, _, _1 = gauss_moments(ctr[i].prim, ks.gas.K)
            a = pdf_slope(ctr[i].prim, sw, ks.gas.K)
            swt = -ctr[i].prim[1] .* moments_conserve_slope(a, Mu, Mxi, 1)
            A = pdf_slope(ctr[i].prim, swt, ks.gas.K)
            tau = vhs_collision_time(ctr[i].prim, ks.gas.μᵣ, ks.gas.ω)
            ctr[i].f .= chapman_enskog(ks.vs.u, ctr[i].prim, a, A, tau)
        end
    end

    regime0 .= regime

    return nothing
end

function up!(ks, ctr, dt, regime)
    res = zeros(3)
    avg = zeros(3)

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
                ks.vSpace.u,
                ks.vSpace.weights,
                ks.gas.γ,
                ks.gas.μᵣ,
                ks.gas.ω,
                ks.gas.Pr,
                ks.ps.dx[i],
                dt,
                res,
                avg,
                :bgk,
            )
        end
    end
end

cd(@__DIR__)
#@load "../nn_scalar.jld2" nn
@load "/home2/vavrines/Coding/Flowmachine/src/1d/sampler/nn_rif.jld2" nn

set = Setup(case = "sod", space = "1d1f1v", maxTime = 0.15)
ps = PSpace1D(0.0, 1.0, 200, 1)
vs = VSpace1D(-5.0, 5.0, 100)
gas = Gas(Kn = 1e-4, K = 0.0, γ = 3.0)
ib = IB1F(ib_sod(set, ps, vs, gas)...)
ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks, ks.ps, :dynamic_array; structarray = true)
for i in eachindex(face)
    face[i].fw .= 0.0
    face[i].ff .= 0.0
end

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ctr[1].w)
avg = zero(res)
regime = ones(Int, axes(ks.ps.x))
regime0 = deepcopy(regime)

@time for iter = 1:nt
    #reconstruct!(ks, ctr)
    split_regime!(regime, regime0, ks, ctr, nn)

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
                ks.vs.weights,
                dt,
                ctr[i-1].sf,
                ctr[i].sf,
            )
        end
    end
    
    up!(ks, ctr, dt, regime)

    global t += dt
end

plot(ks, ctr)
