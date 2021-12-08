using Kinetic, Plots, Flux
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)

set = Setup(case = "sod", space = "1d2f1v", maxTime = 0.15)
ps = PSpace1D(0.0, 1.0, 200, 1)
vs = VSpace1D(-5.0, 5.0, 100)
gas = Gas(Kn = 1e-4, K = 2.0, γ = 5/3)
ib = IB1F(ib_sod(set, ps, vs, gas)...)
ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks, ks.ps, :dynamic_array; structarray = true)

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ctr[1].w)
avg = zero(res)
regime = ones(Int, axes(ks.ps.x))
regime0 = deepcopy(regime)

@showprogress for iter = 1:nt
    #reconstruct!(ks, ctr)
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    t += dt
end

plot(ks, ctr)
