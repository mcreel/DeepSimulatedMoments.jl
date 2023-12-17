# CUE objective, in form of likelihood, to MAXIMIZE
function msm_obj(
    θ::AbstractVector{T}, # trial value
    θ̂ₓ::AbstractVector{T}, # real data net fit
    mn::MomentNetwork,
    dgp::AbstractDGP{T}, 
    S::Int 
) where {T<:AbstractFloat}
    !insupport(dgp,θ̂ₓ) ? error("data moment is not in prior suppor") : nothing  
    # Ensure solution is in parameter space
    insupport(dgp, θ) || return -Inf

    # Compute simulated moments and their covariance matrix
    θ̂ₛ, Σ̂ₛ = simmomentscov(mn, dgp, S, θ)
    Σ̂ₛ *= dgp.N*(1+1/S) # scale for accuracy
    isposdef(Σ̂ₛ) || return -Inf
    # Compute weighting matrix, return bmsm objective
    W = inv(Σ̂ₛ)
    err = √dgp.N * (θ̂ₛ - θ̂ₓ)
    -.5dot(err, W, err)
end


