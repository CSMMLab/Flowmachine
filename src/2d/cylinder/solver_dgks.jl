using KitBase, KitBase.JLD2
using KitBase.ProgressMeter: @showprogress

cd(@__DIR__)
include("tools.jl")

function ev!(
    KS::SolverSet,
    ctr::T1,
    a1face::T2,
    a2face::T2,
    dt;
    mode = KitBase.symbolize(KS.set.flux)::Symbol,
    bc = KitBase.symbolize(KS.set.boundary),
) where {T1<:AbstractArray{ControlVolume2D2F,2},T2<:AbstractArray{Interface2D2F,2}}

    nx, ny, dx, dy = begin
        if KS.ps isa CSpace2D
            KS.ps.nr, KS.ps.nθ, KS.ps.dr, KS.ps.darc
        else
            KS.ps.nx, KS.ps.ny, KS.ps.dx, KS.ps.dy
        end
    end

    if firstindex(KS.pSpace.x[:, 1]) < 1
        idx0 = 1
        idx1 = nx + 1
    else
        idx0 = 2
        idx1 = nx
    end
    if firstindex(KS.pSpace.y[1, :]) < 1
        idy0 = 1
        idy1 = ny + 1
    else
        idy0 = 2
        idy1 = ny
    end

    # x direction
    @inbounds Threads.@threads for j = 1:ny
        for i = idx0:idx1
            vn = KS.vSpace.u .* a1face[i, j].n[1] .+ KS.vSpace.v .* a1face[i, j].n[2]
            vt = KS.vSpace.v .* a1face[i, j].n[1] .- KS.vSpace.u .* a1face[i, j].n[2]
            wL = local_frame(ctr[i-1, j].w, a1face[i, j].n[1], a1face[i, j].n[2])
            wR = local_frame(ctr[i, j].w, a1face[i, j].n[1], a1face[i, j].n[2])

            flux_gks!(
                a1face[i, j].fw,
                a1face[i, j].fh,
                a1face[i, j].fb,
                wL,
                wR,
                vn,
                vt,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.μᵣ,
                KS.gas.ω,
                dt,
                ks.ps.dr[i-1, j] / 2,
                ks.ps.dr[i, j] / 2,
                a1face[i, j].len,
                zeros(4),
                zeros(4),
            )

            a1face[i, j].fw .=
                global_frame(a1face[i, j].fw, a1face[i, j].n[1], a1face[i, j].n[2])
        end
    end

    # y direction
    @inbounds Threads.@threads for j = idy0:idy1
        for i = 1:nx
            vn = KS.vSpace.u .* a2face[i, j].n[1] .+ KS.vSpace.v .* a2face[i, j].n[2]
            vt = KS.vSpace.v .* a2face[i, j].n[1] .- KS.vSpace.u .* a2face[i, j].n[2]
            wL = local_frame(ctr[i, j-1].w, a2face[i, j].n[1], a2face[i, j].n[2])
            wR = local_frame(ctr[i, j].w, a2face[i, j].n[1], a2face[i, j].n[2])

            flux_gks!(
                a2face[i, j].fw,
                a2face[i, j].fh,
                a2face[i, j].fb,
                wL,
                wR,
                vn,
                vt,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.μᵣ,
                KS.gas.ω,
                dt,
                ks.ps.darc[i-1, j] / 2,
                ks.ps.darc[i, j] / 2,
                a2face[i, j].len,
                zeros(4),
                zeros(4),
            )

            a2face[i, j].fw .=
                global_frame(a2face[i, j].fw, a2face[i, j].n[1], a2face[i, j].n[2])
        end
    end

    KitBase.evolve_boundary!(KS, ctr, a1face, a2face, dt, mode, bc)

    return nothing

end

begin
    set = Setup(
        case = "cylinder",
        space = "2d2f2v",
        boundary = ["maxwell", "extra", "mirror", "mirror"],
        limiter = "minmod",
        cfl = 0.5,
        maxTime = 15.0, # time
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

isNewRun = true
#isNewRun = false
if isNewRun
    ctr, a1face, a2face = init_fvm(ks, ks.ps)
else
    @load "kn3dgks.jld2" ctr a1face a2face
end

t = 0.0
dt = timestep(ks, ctr, 0.0)
nt = ks.set.maxTime ÷ dt |> Int
res = zeros(4)
@showprogress for iter = 1:nt
    ev!(ks, ctr, a1face, a2face, dt)
    update!(ks, ctr, a1face, a2face, dt, res)

    for j = ks.ps.nθ÷2+1:ks.ps.nθ
        ctr[ks.ps.nr+1, j].w .= ks.ib.fw(6, 0)
        ctr[ks.ps.nr+1, j].prim .= conserve_prim(ctr[ks.ps.nr+1, j].w, ks.gas.γ)
        ctr[ks.ps.nr+1, j].sw .= 0.0
        ctr[ks.ps.nr+1, j].h .= maxwellian(ks.vs.u, ks.vs.v, ctr[ks.ps.nr+1, j].prim)
        ctr[ks.ps.nr+1, j].b .= ctr[ks.ps.nr+1, j].h .* ks.gas.K ./ 2 ./ ctr[ks.ps.nr+1, j].prim[end]
    end

    global t += dt
    if iter % 199 == 0
        println("residuals: $(res)")
        @save "kn3dgks.jld2" ctr a1face a2face
    end

    if maximum(res) < 1e-6
        break
    end
end

@save "kn3dgks.jld2" ctr a1face a2face
#=
begin
    sol = zeros(ks.ps.nr, ks.ps.nθ, 4)
    for i in axes(sol, 1), j in axes(sol, 2)
        sol[i, j, :] .= ctr[i, j].prim
        sol[i, j, end] = 1 / sol[i, j, end]
    end
    contourf(
        ps.x[1:ks.ps.nr, 1:ks.ps.nθ],
        ps.y[1:ks.ps.nr, 1:ks.ps.nθ],
        sol[:, :, 4],
        ratio = 1,
    )
end
=#