# CUE objective, in form of likelihood, to MAXIMIZE
function bmsm_obj_cue(
    θ̂ₓ::AbstractVector{T}, θ⁺::AbstractVector{T}, 
    mn::MomentNetwork, dgp::AbstractDGP{T}, S::Int 
) where {T<:AbstractFloat}
    # Ensure solution is in parameter space
    insupport(dgp, θ⁺) || return Inf

    # Compute simulated moments and their covariance matrix
    θ̂ₛ, Σ̂ₛ = simmomentscov(mn, dgp, S, θ⁺)
    Σ̂ₛ *= dgp.N*(1+1/S) # scale for accuracy
    isposdef(Σ̂ₛ) || return Inf
    # Compute weighting matrix, return bmsm objective
    W = inv(Σ̂ₛ)
    err = √dgp.N * (θ̂ₛ - θ̂ₓ)
    -.5dot(err, W, err)
end


