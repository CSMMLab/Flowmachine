using Kinetic, Plots
using KitBase.JLD2
using Base.Threads: @threads
using KitBase.ProgressMeter: @showprogress
cd(@__DIR__)

X = Float32.([[1e-4, 0.0, 0.0, 1e-4]; zeros(4); 1.0])
Y = Int32(0)

begin
    set = Setup(space = "1d1f3v", boundary = ["fix", "fix"], collision = "fsm")
    ps = PSpace1D(-0.5, 0.5, 250, 1)
    vs = VSpace3D(-6.0, 6.0, 28, -6.0, 6.0, 28, -6.0, 6.0, 28)
    gas = begin
        Kn = 5e-3
        Gas(Kn = Kn, K = 0.0, fsm = fsm_kernel(vs, ref_vhs_vis(Kn, 1.0, 0.5)))
    end
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

τ0 = vhs_collision_time(ctr[1].prim, ks.gas.μᵣ, ks.gas.ω)
tmax = 50 * τ0
t = 0.0
dt = timestep(ks, ctr, t)
nt = Int(tmax ÷ dt)
res = zero(ctr[1].w)

@showprogress for iter = 1:nt
    evolve!(ks, ctr, face, dt)
    update!(ks, ctr, face, dt, res)

    if iter%5 == 0
        for i = 1:ks.ps.nx
            if abs(ctr[i].prim[2]) > 1e-6
                sw = (ctr[i+1].w .- ctr[i-1].w) / ks.ps.dx[i] / 2
                x, y = regime_data(ks, ctr[i].w, sw, zero(sw), zero(sw), ctr[i].f)
                rx = [x[1:3]; x[5:8]; x[10:11]]
                global X = hcat(X, rx)
                global Y = hcat(Y, y)
            end
        end
    end

    global t += dt
end
@save "data/dataset.jld2" X Y
