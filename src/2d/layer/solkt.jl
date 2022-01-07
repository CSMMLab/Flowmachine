using KitBase, Plots
using KitBase.JLD2
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress
cd(@__DIR__)

function up!(ks, ctr, face, dt, p)
    kn_bzm, nm, phi, psi, phipsi = p

    res = zeros(5)
    avg = zeros(5)

    @inbounds @threads for i = 1:ks.ps.nx
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

        #=KitBase.step!(
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
            ks.vs.weights,
            ks.gas.γ,
            ks.gas.μᵣ,
            ks.gas.ω,
            ks.gas.Pr,
            ks.ps.dx[i],
            dt,
            res,
            avg,
            :bgk,
        )=#
    end

    return nothing
end

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
    ctr, face = init_fvm(ks, ks.ps)
end

fsm = fsm_kernel(ks.vs, ks.gas.μᵣ)
τ0 = vhs_collision_time(ctr[1].prim, ks.gas.μᵣ, ks.gas.ω)
tmax = 10τ0#50τ0
t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(tmax ÷ dt)
res = zero(ctr[1].w)

@time @showprogress for iter = 1:nt
    #reconstruct!(ks, ctr)

    @inbounds @threads for i = 1:ks.ps.nx+1
        flux_kfvs!(
            face[i].fw,
            face[i].ff,
            ctr[i-1].f, #.+ 0.5 .* ctr[i-1].sf .* ks.ps.dx[i-1],
            ctr[i].f, #.- 0.5 .* ctr[i].sf .* ks.ps.dx[i],
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

    up!(ks, ctr, face, dt, fsm)

    #=global t += dt
    if abs(t - τ0) < dt
        @save "sol_t.jld2" ctr face
    elseif abs(t - 10 * τ0) < dt
        @save "sol_10t.jld2" ctr face
    end=#
end
#@save "sol_50t.jld2" ctr face

# field
sol = zeros(ks.ps.nx, 5)
for i in axes(sol, 1)
    sol[i, :] .= ctr[i].prim
    sol[i, end] = 1 / sol[i, end]
end

plot(ks.ps.x[1:ks.ps.nx], sol[:, 1])

# distribution function
fc = (ctr[end÷2].f + ctr[end÷2+1].f) ./ 2
hc = reduce_distribution(fc, vs.weights[:, 1, :], 2)
plot(ks.vs.v[1, :, 1], hc)
