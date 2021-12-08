function cir(nom, uv_max = 5.0)
    x = zeros(nom * nom)  # exclude r=0
    y = zeros(nom * nom)
    w = zeros(nom * nom)
    k = 1
    for i = 1:nom # r=0 excluded
        for j = 0:(nom-1) # phi=2*pi excluded
            x[k] = i * uv_max / nom * cos(j * 2 * π / nom)
            y[k] = i * uv_max / nom * sin(j * 2 * π / nom)
            w[k] = (2 * i - 1) * (uv_max / nom)^2 * π / nom
            k += 1
        end
    end

    return x, y, w
end

function get_moments(f, u, v, ω)
    w = zeros(eltype(f), 4)
    w[1] = discrete_moments(f, u, ω, 0)
    w[2] = discrete_moments(f, u, ω, 1)
    w[3] = discrete_moments(f, v, ω, 1)
    w[4] = 0.5 * (discrete_moments(f, u, ω, 2) + discrete_moments(f, v, ω, 2))
    return w
end

u, v, weights = cir(160)

using KitBase, Test

prim = [1.0, 0.0, 0.0, 1.0]
γ = 2.0
w = prim_conserve(prim, γ)

M = maxwellian(u, v, prim)
w1 = get_moments(M, u, v, weights)

@test w ≈ w1 atol = 0.05
