using KitBase, KitBase.JLD2, KitBase.WriteVTK
#using Plots; pyplot()
using PyPlot

begin
    set = Setup(
        case = "cylinder",
        space = "2d2f2v",
        boundary = ["maxwell", "extra", "mirror", "mirror"],
        limiter = "minmod",
        cfl = 0.5,
        maxTime = 10.0, # time
    )
    ps = CSpace2D(1.0, 6.0, 60, 0.0, π, 50, 1, 1)
    vs = VSpace2D(-10.0, 10.0, 48, -10.0, 10.0, 48)
    gas = Gas(Kn = 1e-3, Ma = 5.0, K = 1.0)
    
    prim0 = [1.0, 0.0, 0.0, 1.0]
    prim1 = [1.0, gas.Ma * sound_speed(1.0, gas.γ), 0.0, 1.0]
    fw = (args...) -> prim_conserve(prim1, gas.γ)
    ff = function(args...)
        prim = conserve_prim(fw(args...), gas.γ)
        h = maxwellian(vs.u, vs.v, prim)
        b = h .* gas.K / 2 / prim[end]
        return h, b
    end
    bc = function(x, y)
        if abs(x^2 + y^2 - 1) < 1e-3
            return prim0
        else
            return prim1
        end
    end
    ib = IB2F(fw, ff, bc)

    ks = SolverSet(set, ps, vs, gas, ib)
end

cd(@__DIR__)
@load "kn3ref.jld2" ctr

begin
    sol = zeros(ks.ps.nr, ks.ps.nθ, 4)
    for i in axes(sol, 1), j in axes(sol, 2)
        sol[i, j, :] .= ctr[i, j].prim
        sol[i, j, end] = 1 / sol[i, j, end]
    end
end

#--- Plots ---#
contourf(
    ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
    ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
    sol[:, :, 4],
    #ratio = 1,
    xlabel = "x",
    ylabel = "y",
)

#--- PyPlot ---#
begin
    close("all")
    fig = figure("contour", figsize=(8, 4))
    PyPlot.contourf(
        ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        sol[:, :, 4],
        aspect=1,
        levels=20,
        cmap=ColorMap("inferno"),
    )
    #axis("off")
    colorbar()
    xlabel("x")
    ylabel("y")
    #axis("equal")
    tight_layout()
    #PyPlot.title("U-velocity")
    xlim(-6, 6)
    ylim(0, 6)
    #PyPlot.grid("on")
    #display(gcf())
    #display(fig)
end

fig.savefig("cylinder_t_kn3.pdf")
