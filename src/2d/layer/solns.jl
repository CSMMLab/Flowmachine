using KitBase, Plots
using KitBase.JLD2
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress
cd(@__DIR__)

begin
    set = Setup(case = "layer", space = "1d0f0v", maxTime = 0.2, boundary = ["fix", "fix"], cfl = 0.3)
    ps = PSpace1D(-0.5, 0.5, 500, 1)
    vs = nothing
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
    ib = IB(fw, gas)
    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, face = init_fvm(ks, ks.ps)
end

τ0 = vhs_collision_time(ctr[1].prim, ks.gas.μᵣ, ks.gas.ω)
tmax = 50 * τ0
t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(tmax ÷ dt)
res = zero(ctr[1].w)

@time @showprogress for iter = 1:nt
    #reconstruct!(ks, ctr)

    @inbounds @threads for i = 1:ks.ps.nx+1
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
    end

    update!(ks, ctr, face, dt, res)

    #=global t += dt
    if abs(t - τ0) < dt
        @save "solns_t.jld2" ctr face
    elseif abs(t - 10 * τ0) < dt
        @save "solns_10t.jld2" ctr face
    end=#
end
#@save "solns_50t.jld2" ctr face

#=sol = zeros(ks.ps.nx, 5)
for i in axes(sol, 1)
    sol[i, :] .= ctr[i].prim
    sol[i, end] = 1 / sol[i, end]
end

plot(ks.ps.x[1:ks.ps.nx], sol)
=#