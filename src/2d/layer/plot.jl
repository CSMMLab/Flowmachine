using KitBase, Plots, Flux
using KitBase.JLD2
cd(@__DIR__)
@load "sol_t.jld2" ctr

begin
    set = Setup(case = "layer", space = "1d1f3v", maxTime = 0.2, boundary = ["fix", "fix"])
    ps = PSpace1D(-0.6, 0.6, 600, 1)
    vs = VSpace3D(-6.0, 6.0, 24, -6.0, 6.0, 64, -6.0, 6.0, 24)
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
end

# solution field
sol = zeros(ks.ps.nx, 5)
for i in axes(sol, 1)
    sol[i, :] .= ctr[i].prim
    sol[i, end] = 1 / sol[i, end]
end

plot(ks.ps.x[1:ks.ps.nx], sol)

# distribution function
fc = (ctr[end÷2].f + ctr[end÷2+1].f) ./ 2
vs2d = VSpace2D(ks.vs.u0, ks.vs.u1, ks.vs.nu, ks.vs.w0, ks.vs.w1, ks.vs.nw)
hc = reduce_distribution(fc, vs2d.weights, 2)
plot(ks.vs.v[1, :, 1], hc)
