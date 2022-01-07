using KitBase, DataFrames
using KitBase.CSV
cd(@__DIR__)

###
# parameters
###
file = "data/pdfs.csv"
kn = [1e-3, 1e-3]
dx = [1e-2, 1e-2]

###
# workflow
###
f = open(file)
data = []
for line in eachline(f)
    a = split(line, ",")
    b = [parse(Float64, a[i]) for i = 1:length(a)]
    push!(data, b)
end
pdfs = data[3:end];

vs = VSpace1D(minimum(data[1]), maximum(data[1]), length(data[1]), data[1], data[1][2:end] .- data[1][1:end - 1], data[2])
println(vs.u)
#println(vs.weights)
ws = [moments_conserve(pdfs[i], vs.u, vs.weights) for i in axes(pdfs, 1)]
wa = zeros(9, 3)
for i in 1:9
    wa[i, :] .= ws[i]
end

w = [ws[3], ws[6]]
sw = [(ws[2] - ws[1]) / dx[1], (ws[5] - ws[4]) / dx[2]]

function f_ns(vs, w, sw, kn, γ = 3, K = 0, μ = ref_vhs_vis(kn, 1, 0.5), ω = 0.81)
    μ = ref_vhs_vis(kn, 1, 0.5)
    prim = conserve_prim(w, γ)
    Mu, Mxi, _, _1 = gauss_moments(prim, K)
    a = pdf_slope(prim, sw, K)
    swt = -prim[1] .* moments_conserve_slope(a, Mu, Mxi, 1)
    A = pdf_slope(prim, swt, K)
    τ = vhs_collision_time(prim, μ, ω)
    fr = chapman_enskog(vs.u, prim, a, A, τ)

    return fr, τ
end

fr1, τ1 = f_ns(vs, w[1], sw[1], kn[1])
fr2, τ2 = f_ns(vs, w[2], sw[2], kn[2])

###
# Write
###
df0 = DataFrame(density = wa[:, 1], velocity = wa[:, 2], energy = wa[:, 3])
CSV.write("data/w.csv", df0)

df1 = DataFrame(gradient1 = sw[1], gradient2 = sw[2])
CSV.write("data/dw.csv", df1)

df2 = DataFrame(fns1 = fr1, fns2 = fr2)
CSV.write("data/fns.csv", df2)

df3 = DataFrame(tau = [τ1, τ2], kn = kn, dx = dx)
CSV.write("data/paras.csv", df3)
