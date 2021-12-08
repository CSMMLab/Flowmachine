using KitBase, LinearAlgebra

function regime_data(w, swx, swy, f, u, v, K, Kn, μ=ref_vhs_vis(Kn, 1.0, 0.5), ω=0.81)
    γ = heat_capacity_ratio(K, 2)
    prim = conserve_prim(w, γ)
    
    Mu, Mv, Mxi, _, _1 = gauss_moments(prim, K)
    a = pdf_slope(prim, swx, K)
    b = pdf_slope(prim, swy, K)
    swt = -prim[1] .* (moments_conserve_slope(a, Mu, Mv, Mxi, 1, 0) .+ moments_conserve_slope(b, Mu, Mv, Mxi, 0, 1))
    A = pdf_slope(prim, swt, K)
    tau = vhs_collision_time(prim, μ, ω)
    
    fr = chapman_enskog(u, v, prim, a, b, A, tau)
    L = norm((f .- fr) ./ prim[1])

    sw = (swx.^2 + swy.^2).^0.5
    x = [w; sw; tau]
    y = ifelse(L <= 0.005, 0.0, 1.0)

    return x, y
end

function regime_data(ks::SolverSet, w, swx, swy, f)
    regime_data(w, swx, swy, f, ks.vs.u, ks.vs.v, ks.gas.K, ks.gas.Kn)
end

function regime_number(Y, rg=0)
    idx = 0
     for i in axes(Y, 2)
        if Y[1, i] == rg
             idx += 1
         end
     end

     RG = ifelse(rg == 0, "NS", "BZ")
     println("$(RG) regime: $(idx) of $(size(Y, 2))")

     return nothing
 end

function accuracy(nn, X, Z)
    Z1 = nn(X)

    ZA1 = [round(Z1[1, i]) for i in axes(Z1, 2)]
    ZA = [round(Z[1, i]) for i in axes(Z, 2)]

    accuracy = 0.0
    for i in eachindex(ZA)
        if ZA[i] == ZA1[i]
            accuracy += 1.0
        end
    end
    accuracy /= length(ZA)

    return accuracy
end
