using NPZ, Plots, NipponColors
cd(@__DIR__)

D = dict_color()

# sampler
gmv = npzread("generated_macroscopic_variables_normalized.npy")
∇gmv = npzread("generated_macroscopic_variables_normalized_gradients.npy")

# solver generator
smv = npzread("sod_macroscopic_variables_normalized.npy")
∇smv = npzread("sod_macroscopic_variables_gradients.npy")

# plot
p1 = scatter(gmv[:, 1], gmv[:, 2], marker = (:circle, 3, 0.6, D["ukon"], stroke(1, 0.1, :gray27, :dot)), legend=:none, xlims=(-1, 1), ylims=(0, 1), xlabel="U", ylabel="T")
savefig(p1, "illu_generator_u.pdf")

p2 = scatter(smv[:, 1], smv[:, 2], marker = (:circle, 3, 0.6, D["ukon"], stroke(1, 0.1, :gray27, :dot)), legend=:none, xlims=(-1, 1), ylims=(0, 1), xlabel="U", ylabel="T")
savefig(p2, "illu_solver_u.pdf")

p3 = scatter(∇gmv[:, 1], ∇gmv[:, 2], marker = (:circle, 3, 0.6, D["ukon"], stroke(1, 0.1, :gray27, :dot)), legend=:none, xlims=(-10, 10), ylims=(-10, 10), xlabel="∇U", ylabel="∇T")
savefig(p3, "illu_generator_du.pdf")

p4 = scatter(∇smv[:, 1], ∇smv[:, 2], marker = (:circle, 3, 0.6, D["ukon"], stroke(1, 0.1, :gray27, :dot)), legend=:none, xlims=(-10, 10), ylims=(-10, 10), xlabel="∇U", ylabel="∇T")
savefig(p4, "illu_solver_du.pdf")
