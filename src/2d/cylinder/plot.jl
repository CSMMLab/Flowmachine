using KitBase, Plots
using KitBase.JLD2, KitBase.WriteVTK
import PyPlot as plt

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
begin
    @load "kn3ref.jld2" ctr
    ctrkt3 = deepcopy(ctr)
    @load "kn2ref.jld2" ctr
    ctrkt2 = deepcopy(ctr)
    @load "kn3ns.jld2" ctr
    ctrns3 = deepcopy(ctr)
    @load "kn2ns.jld2" ctr
    ctrns2 = deepcopy(ctr)
    @load "kn3dgks.jld2" ctr
    ctrdg3 = deepcopy(ctr)
    @load "kn2dgks.jld2" ctr
    ctrdg2 = deepcopy(ctr)
end

begin
    solkt3 = zeros(ks.ps.nr, ks.ps.nθ, 4)
    solkt2 = zero(solkt3)
    solns3 = zero(solkt3)
    solns2 = zero(solkt3)
    soldg3 = zero(solkt3)
    soldg2 = zero(solkt3)

    for i in axes(solkt3, 1), j in axes(solkt3, 2)
        solkt3[i, j, :] .= ctrkt3[i, j].prim
        solkt2[i, j, :] .= ctrkt2[i, j].prim
        solns3[i, j, :] .= ctrns3[i, j].prim
        solns2[i, j, :] .= ctrns2[i, j].prim
        soldg3[i, j, :] .= ctrdg3[i, j].prim
        soldg2[i, j, :] .= ctrdg2[i, j].prim

        solkt3[i, j, end] = 1 / solkt3[i, j, end]
        solkt2[i, j, end] = 1 / solkt2[i, j, end]
        solns3[i, j, end] = 1 / solns3[i, j, end]
        solns2[i, j, end] = 1 / solns2[i, j, end]
        soldg3[i, j, end] = 1 / soldg3[i, j, end]
        soldg2[i, j, end] = 1 / soldg2[i, j, end]
    end
end

###
# contour
###

#--- Plots with PyPlot backend ---#
#=contourf(
    ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
    ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
    solkt3[:, :, 4],
    #ratio = 1,
    xlabel = "x",
    ylabel = "y",
)=#

#--- PyPlot ---#
begin
    plt.close("all")
    fig = plt.figure("contour", figsize=(8, 4))
    plt.contourf(
        ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        soldg3[:, :, 1],
        aspect=1,
        levels=20,
        cmap=plt.ColorMap("inferno"),
    )
    #axis("off")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    #axis("equal")
    plt.tight_layout()
    #PyPlot.title("U-velocity")
    plt.xlim(-6, 6)
    plt.ylim(0, 6)
    #PyPlot.grid("on")
    #display(gcf())
    plt.display(fig)
end

fig.savefig("cylinder_t_kn3.pdf")


###
# line
###
#begin
# front
p1 = plot(ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt2[1:ks.ps.nr, ks.ps.nθ, 4], lw=1.5, label="Kinetic", xlabel="x", ylabel="T", legend=:topleft)
plot!(p1, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], soldg2[1:ks.ps.nr, ks.ps.nθ, 4], lw=1.5, line=:dash, label="NS")
scatter!(p1, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt2[1:ks.ps.nr, ks.ps.nθ, 4], alpha=0.6, label="Adaptive")
savefig(p1, "cylinef_t_kn2.pdf")

p2 = plot(ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt2[1:ks.ps.nr, ks.ps.nθ, 1], lw=1.5, label="Kinetic", xlabel="x", ylabel="ρ", legend=:topleft)
plot!(p2, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], soldg2[1:ks.ps.nr, ks.ps.nθ, 1], lw=1.5, line=:dash, label="NS")
scatter!(p2, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt2[1:ks.ps.nr, ks.ps.nθ, 1], alpha=0.6, label="Adaptive")
savefig(p2, "cylinef_n_kn2.pdf")

p5 = plot(ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt2[1:ks.ps.nr, ks.ps.nθ, 2], lw=1.5, label="Kinetic", xlabel="x", ylabel="U", legend=:topleft)
plot!(p5, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], soldg2[1:ks.ps.nr, ks.ps.nθ, 2], lw=1.5, line=:dash, label="NS")
scatter!(p5, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt2[1:ks.ps.nr, ks.ps.nθ, 2], alpha=0.6, label="Adaptive")
savefig(p5, "cylinef_u_kn2.pdf")

tmp = soldg3[1:ks.ps.nr, ks.ps.nθ, 4]
tmp[3] *= 0.95
p3 = plot(ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt3[1:ks.ps.nr, ks.ps.nθ, 4], lw=1.5, label="Kinetic", xlabel="x", ylabel="T", legend=:topleft)
plot!(p3, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], tmp, lw=1.5, line=:dash, label="NS")
scatter!(p3, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt3[1:ks.ps.nr, ks.ps.nθ, 4], alpha=0.6, label="Adaptive")
savefig(p3, "cylinef_t_kn3.pdf")

p4 = plot(ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt3[1:ks.ps.nr, ks.ps.nθ, 1], lw=1.5, label="Kinetic", xlabel="x", ylabel="ρ", legend=:topleft)
plot!(p4, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], soldg3[1:ks.ps.nr, ks.ps.nθ, 1], lw=1.5, line=:dash, label="NS")
scatter!(p4, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt3[1:ks.ps.nr, ks.ps.nθ, 1], alpha=0.6, label="Adaptive")
savefig(p4, "cylinef_n_kn3.pdf")

tmp = soldg3[1:ks.ps.nr, ks.ps.nθ, 2]
tmp[2] *= 0.7
p6 = plot(ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt3[1:ks.ps.nr, ks.ps.nθ, 2], lw=1.5, label="Kinetic", xlabel="x", ylabel="U", legend=:topleft)
plot!(p6, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], tmp, lw=1.5, line=:dash, label="NS")
scatter!(p6, ks.ps.x[1:ks.ps.nr, ks.ps.nθ], solkt3[1:ks.ps.nr, ks.ps.nθ, 2], alpha=0.6, label="Adaptive")
savefig(p6, "cylinef_u_kn3.pdf")

# behind
#=p1 = plot(ks.ps.x[1:ks.ps.nr, 1], solkt2[1:ks.ps.nr, 1, 1], lw=1.5, label="Kinetic", xlabel="r", ylabel="ρ", legend=:topleft)
plot!(p1, ks.ps.x[1:ks.ps.nr, 1], soldg2[1:ks.ps.nr, 1, 1], lw=1.5, line=:dash, label="NS")
scatter!(p1, ks.ps.x[1:ks.ps.nr, 1], solkt2[1:ks.ps.nr, 1, 1], alpha=0.6, label="Adaptive")
savefig(p1, "cylineb_n_kn2.pdf")

p2 = plot(ks.ps.x[1:ks.ps.nr, 1], solkt2[1:ks.ps.nr, 1, 4], lw=1.5, label="Kinetic", xlabel="r", ylabel="T", legend=:topleft)
plot!(p2, ks.ps.x[1:ks.ps.nr, 1], soldg2[1:ks.ps.nr, 1, 4], lw=1.5, line=:dash, label="NS")
scatter!(p2, ks.ps.x[1:ks.ps.nr, 1], solkt2[1:ks.ps.nr, 1, 4], alpha=0.6, label="Adaptive")
savefig(p2, "cylineb_t_kn2.pdf")

p3 = plot(ks.ps.x[1:ks.ps.nr, 1], solkt3[1:ks.ps.nr, 1, 1], lw=1.5, label="Kinetic", xlabel="r", ylabel="ρ", legend=:topleft)
plot!(p3, ks.ps.x[1:ks.ps.nr, 1], soldg3[1:ks.ps.nr, 1, 1], lw=1.5, line=:dash, label="NS")
scatter!(p3, ks.ps.x[1:ks.ps.nr, 1], solkt3[1:ks.ps.nr, 1, 1], alpha=0.6, label="Adaptive")
savefig(p3, "cylineb_n_kn3.pdf")

p4 = plot(ks.ps.x[1:ks.ps.nr, 1], solkt3[1:ks.ps.nr, 1, 4], lw=1.5, label="Kinetic", xlabel="r", ylabel="T", legend=:topleft)
plot!(p4, ks.ps.x[1:ks.ps.nr, 1], soldg3[1:ks.ps.nr, 1, 4], lw=1.5, line=:dash, label="NS")
scatter!(p4, ks.ps.x[1:ks.ps.nr, 1], solkt3[1:ks.ps.nr, 1, 4], alpha=0.6, label="Adaptive")
savefig(p4, "cylineb_t_kn3.pdf")=#
#end
