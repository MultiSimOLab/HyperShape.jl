# The settings of the reaction–diffusion method in papers, as named tuples of keywords of ReactionDiffusion (not part
# of HyperShape: the validation scripts splat them, `ReactionDiffusion(; τ, KIM2020...)`, and any keyword written after
# them replaces the one of the tuple).

# Kim, Kim, Choi, Yoon (2020), a density (the script sets the rest: φ₀ = 0.5 with bounds = (0.0, 1.0), SIMP p = 3,
# Volume(…; scale = |Ω|), and τ: 5e-4 for their cantilever, 1e-3 for their half MBB beam). Every constraint goes to the
# augmented Lagrangian (volume = nothing), with the penalty rule of the paper.
const KIM2020 = (Δt = 0.1, normalize = L2Norm(), mass = :consistent, volume = nothing,
                 lagrangian = AugmentedLagrangian(Bertsekas(μ0 = 0.2, growth = 1.025, window = 1, progress = Inf, μmax = 2.0)))

# The normalisation of Takezawa et al. (2010), C = 1 / (Σᵢ gᵢ² / mᵢ)^½: the L² norm without |D| (L2Norm() divides
# |D| by it). A reduction over a lazy broadcast, without allocations.
takezawa_normalization(g, m, area) = 1 / sqrt(sum(Broadcast.instantiate(Broadcast.broadcasted((a, b) -> a^2 / b, g, m))))

# Takezawa, Nishiwaki and Kitamura (2010), the double well: their explicit scheme with 20 substeps, no settling, their
# normalisation and the volume lowered 1 % of |D| per iteration (the script sets κ, their κ = 1e-5 on the bridge 2 × 1,
# that is τ = 5e-6, the start with holes, k = max(φ³, 1e-5) and 100 iterations with stop = nothing). Differences with
# the paper: the multiplier of the volume is exact (instead of their rule for ζ) and the discretisation is P1 finite
# elements with lumped mass (instead of finite differences).
const TAKEZAWA2010 = (reaction = DoubleWell(W = 0.25, η = 20.0, scheme = :explicit, substeps = 20, settle = false),
                      normalize = takezawa_normalization, volume = ExactVolume(max_step = 0.01))
