using KitBase

prim = [1., 0.1, 0., 0., 1.]
w = prim_conserve(prim, 5/3)
w1 = [1.1, 0., 0., 0., 1.0]
sw = (w1 .- w) / 1e-1
μ = 1e-3
ω = 0.81

tau = vhs_collision_time(prim, μ, ω)
Mu, Mv, Mw, _, _1 = gauss_moments(prim, 0)
a = pdf_slope(prim, sw, 0)
swt = -prim[1] .* moments_conserve_slope(a, Mu, Mv, Mw, 1, 0, 0)
A = pdf_slope(prim, swt, 0)
tau = vhs_collision_time(prim, μ, ω)

vs = VSpace3D(-8.0, 8.0, 48, -8.0, 8.0, 28, -8.0, 8.0, 28)
fr = chapman_enskog(vs.u, vs.v, vs.w, prim, a, zero(a), zero(a), A, tau)
M = maxwellian(vs.u, vs.v, vs.w, prim)

moments_conserve(fr, vs.u, vs.v, vs.w, vs.weights)
moments_conserve(M, vs.u, vs.v, vs.w, vs.weights)