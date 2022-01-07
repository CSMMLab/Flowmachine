using KitBase, Plots
using KitBase.JLD2
using KitBase.ProgressMeter: @showprogress
pyplot()
cd(@__DIR__)

function flux_sgks!(
    fw::X,
    wL::Y,
    wR::Y,
    inK::Real,
    γ::Real,
    μᵣ::Real,
    ω::Real,
    dt::Real,
    dxL::Real,
    dxR::Real,
    dy::Real,
    swL = zero(wL)::Y,
    swR = zero(wR)::Y,
) where {X,Y}

    primL = conserve_prim(wL, γ)
    primR = conserve_prim(wR, γ)

    Mu1, Mv1, Mw1, MuL1, MuR1 = gauss_moments(primL, inK)
    Mu2, Mv2, Mw2, MuL2, MuR2 = gauss_moments(primR, inK)

    w =
        primL[1] .* moments_conserve(MuL1, Mv1, Mw1, 0, 0, 0) .+
        primR[1] .* moments_conserve(MuR2, Mv2, Mw2, 0, 0, 0)
    prim = conserve_prim(w, γ)
    tau =
        vhs_collision_time(prim, μᵣ, ω) +
        2 * dt * abs(primL[1] / primL[end] - primR[1] / primR[end]) /
        (primL[1] / primL[end] + primR[1] / primR[end])

    Mu, Mv, Mw, MuL, MuR = gauss_moments(prim, inK)
    sw0L = (w .- (wL .- swL .* dxL)) ./ dxL
    sw0R = ((wR .+ swR .* dxR) .- w) ./ dxR
    gaL = pdf_slope(prim, sw0L, inK)
    gaR = pdf_slope(prim, sw0R, inK)
    sw =
        -prim[1] .* (
            moments_conserve_slope(gaL, MuL, Mv, Mw, 1, 0, 0) .+
            moments_conserve_slope(gaR, MuR, Mv, Mw, 1, 0, 0)
        )
    gaT = pdf_slope(prim, sw, inK)

    # time-integration constants
    Mt = zeros(5)
    Mt[4] = tau * (1.0 - exp(-dt / tau))
    Mt[5] = -tau * dt * exp(-dt / tau) + tau * Mt[4]
    Mt[1] = dt - Mt[4]
    Mt[2] = -tau * Mt[1] + Mt[5]
    Mt[3] = 0.5 * dt^2 - tau * Mt[1]

    # flux related to central distribution
    Muv = moments_conserve(Mu, Mv, Mw, 1, 0, 0)
    MauL = moments_conserve_slope(gaL, MuL, Mv, Mw, 2, 0, 0)
    MauR = moments_conserve_slope(gaR, MuR, Mv, Mw, 2, 0, 0)
    MauT = moments_conserve_slope(gaT, Mu, Mv, Mw, 1, 0, 0)

    fw .= Mt[1] .* prim[1] .* Muv

    # flux related to upwind distribution
    MuvL = moments_conserve(MuL1, Mv1, Mw1, 1, 0, 0)
    #MauL = moments_conserve_slope(faL, MuL1, Mv1, Mw1, 2, 0, 0)
    #MauLT = moments_conserve_slope(faTL, MuL1, Mv1, Mw1, 1, 0, 0)

    MuvR = moments_conserve(MuR2, Mv2, Mw2, 1, 0, 0)
    #MauR = moments_conserve_slope(faR, MuR2, Mv2, Mw2, 2, 0, 0)
    #MauRT = moments_conserve_slope(faTR, MuR2, Mv2, Mw2, 1, 0, 0)

    #@. fw +=
    #    Mt[4] * primL[1] * MuvL - (Mt[5] + tau * Mt[4]) * primL[1] * MauL -
    #    tau * Mt[4] * primL[1] * MauLT + Mt[4] * primR[1] * MuvR -
    #    (Mt[5] + tau * Mt[4]) * primR[1] * MauR - tau * Mt[4] * primR[1] * MauRT
    @. fw += Mt[4] * primL[1] * MuvL + Mt[4] * primR[1] * MuvR

    fw .*= dy

    return nothing

end

function ev!(KS, ctr, a1face, a2face, dt)
    nx, ny, dx, dy = KS.ps.nr, KS.ps.nθ, KS.ps.dr, KS.ps.darc
    idx0, idx1 = 1, nx+1
    idy0, idy1 = 1, ny+1

    # r direction
    @inbounds Threads.@threads for j = 1:ny
        w = local_frame(ctr[1, j].w, a1face[1, j].n[1], a1face[1, j].n[2])
        flux_boundary_maxwell!(
            a1face[1, j].fw,
            [1.0, 0.0, 0.0, 1.0],
            w,
            KS.gas.K,
            KS.gas.γ,
            dt,
            a1face[1, j].len,
            1,
        )
        a1face[1, j].fw .=
                global_frame(a1face[1, j].fw, a1face[1, j].n[1], a1face[1, j].n[2])
    end
    @inbounds Threads.@threads for j = 2:ny
        for i = idx0:idx1
            wL = local_frame(ctr[i-1, j].w, a1face[i, j].n[1], a1face[i, j].n[2])
            wR = local_frame(ctr[i, j].w, a1face[i, j].n[1], a1face[i, j].n[2])

            flux_sgks!(
                a1face[i, j].fw,
                wL,
                wR,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.μᵣ,
                KS.gas.ω,
                dt,
                ks.ps.dr[i-1, j] / 2,
                ks.ps.dr[i, j] / 2,
                a1face[i, j].len,
            )

            a1face[i, j].fw .=
                global_frame(a1face[i, j].fw, a1face[i, j].n[1], a1face[i, j].n[2])
        end
    end

    # θ direction
    @inbounds Threads.@threads for j = idy0:idy1
        for i = 1:nx
            wL = local_frame(ctr[i, j-1].w, a2face[i, j].n[1], a2face[i, j].n[2])
            wR = local_frame(ctr[i, j].w, a2face[i, j].n[1], a2face[i, j].n[2])

            flux_sgks!(
                a2face[i, j].fw,
                wL,
                wR,
                KS.gas.K,
                KS.gas.γ,
                KS.gas.μᵣ,
                KS.gas.ω,
                dt,
                ks.ps.darc[i, j-1] / 2,
                ks.ps.darc[i, j] / 2,
                a2face[i, j].len,
            )

            a2face[i, j].fw .=
                global_frame(a2face[i, j].fw, a2face[i, j].n[1], a2face[i, j].n[2])
        end
    end

    return nothing
end

begin
    set = Setup(
        case = "cylinder",
        space = "2d0f0v",
        boundary = ["isothermal", "extra", "mirror", "mirror"],
        limiter = "minmod",
        cfl = 0.1,
        maxTime = 15.0, # time
    )
    ps = CSpace2D(1.0, 6.0, 60, 0.0, π, 50, 1, 1)
    vs = nothing
    gas = Gas(Kn = 1e-2, Ma = 5.0, K = 1.0)

    prim0 = [1.0, 0.0, 0.0, 1.0]
    prim1 = [1.0, gas.Ma * sound_speed(1.0, gas.γ), 0.0, 1.0]
    fw = (args...) -> prim_conserve(prim1, gas.γ)
    bc = function(x, y)
        if abs(x^2 + y^2 - 1) < 1e-3
            return prim0
        else
            return prim1
        end
    end
    ib = IB(fw, bc)

    ks = SolverSet(set, ps, vs, gas, ib)
    ctr, a1face, a2face = init_fvm(ks, ks.ps)
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
    end
    for j = 0:ks.ps.nθ
        KitBase.bc_isothermal!(ctr[0, j], ctr[1, j], ks.gas.γ)
    end

    global t += dt
    if iter % 1000 == 0
        println("residuals: $(res)")
    end

    if maximum(res) < 1e-6
        break
    end
end
@save "kn2ns.jld2" ctr a1face a2face

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
        xlabel = "x",
        ylabel = "y",
    )
end
