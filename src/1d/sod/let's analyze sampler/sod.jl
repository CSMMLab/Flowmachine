"""
Please add the latest Kinetic.jl package as it fixes the issue with the regime classifier.
Backend requirement: 
- KitBase: v0.7.6
- KitML: v0.4.4
"""

using Kinetic, NPZ, Plots
using KitBase.ProgressMeter: @showprogress
cd(@__DIR__)

set = Setup(case = "sod", space = "1d1f1v", maxTime = 0.15)
ps = PSpace1D(0.0, 1.0, 100, 1)
vs = VSpace1D(-5.0, 5.0, 64)
gas = Gas(Kn = 1e-3, K = 0, γ = 3)
ib = IB1F(ib_sod(set, ps, vs, gas)...)
ks = SolverSet(set, ps, vs, gas, ib)
ctr, face = init_fvm(ks)

t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(ks.set.maxTime ÷ dt) + 1
res = zero(ctr[1].w)

X, Y, Z, I = begin
    prim = [1.0, 0.0, 1.0]
    w = prim_conserve(prim, ks.gas.γ)
    sw = zeros(3)
    M = maxwellian(ks.vs.u, prim)
    [[1e-4, 0.0, 1e-4]; zeros(3); 1.0], 0.0, M, 0
end

@showprogress for iter = 1:nt
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if iter % 3 == 0
        for i = 1:ks.ps.nx
            sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2.0
            x, y = regime_data(ks, ctr[i].w, sw, ctr[i].f)
            global X = hcat(X, x)
            global Y = hcat(Y, y)
            global Z = hcat(Z, ctr[i].f)
            global I = hcat(I, iter)
        end
    end
end

#plot(ks, ctr)

npzwrite("data.npz", Dict("I" => I, "X" => X, "Y" => Y, "Z" => Z))
