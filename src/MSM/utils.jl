using Distributions, Statistics
# Compute simulated moments
function simmoments(
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int, θ::AbstractVector{T}
) where {T<:AbstractFloat}
    X, _ = generate(θ, dgp, S)
    vec(mean(make_moments(mn, mn.data_transform(X)), dims=2))
end

# Compute simulated moments and their covariance matrix
function simmomentscov(
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int, θ::AbstractVector{T}
) where {T<:AbstractFloat}
    X = generate(θ, dgp, S)
    m = make_moments(mn, mn.data_transform(X))
    mean(m, dims=2), cov(permutedims(m))
end

# MVN random walk proposal
function proposal(
    x::AbstractVector{T}, δ::T, Σ::AbstractMatrix{T}
) where {T<:AbstractFloat}
rand(MvNormal(x, δ * Σ))
end
