using KitBase, LinearAlgebra

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

# cylinder
function judge_regime(ks, f::AbstractMatrix, prim, swx, swy)
    τ = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
    fr = chapman_enskog(ks.vs.u, ks.vs.v, prim, swx, swy, ks.gas.K, τ)

    return judge_regime(f, fr, prim)
end

# layer
function judge_regime(ks, f, prim, swx, swy = zero(swx), swz = zero(swx))
    τ = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
    fr = chapman_enskog(ks.vs.u, ks.vs.v, ks.vs.w, prim, swx, swy, swz, ks.gas.K, τ)

    return judge_regime(f, fr, prim)
end

function judge_regime(f, fr, prim)
    L = norm((f .- fr) ./ prim[1])
    return ifelse(L <= 0.005, 0, 1)
end
