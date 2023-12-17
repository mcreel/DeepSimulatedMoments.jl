# TODO: prior? not needed at present, as priors are uniform
@views function mcmc(
    θ, # start value
    θ⁺, # the NN fitted value
    δ, # tuning
    Σₚ, # proposal covariance
    S,  # simulations per evaluation of likelihood
    mn, # MomentNetwork
    dgp;
    burnin::Int=100,
    chainlength::Int=1_000,
    verbosity::Int=10
)
    # set likelihood and proposal
    Lₙ = θ -> msm_obj(θ, θ⁺, mn, dgp, S) 
    proposal2 = θ -> proposal(θ, Float32(δ), Σₚ)

    Lₙθ = Lₙ(θ)
    naccept = 0 # Number of acceptance / rejections
    accept = false
    acceptance_rate = 1f0
    chain = zeros(chainlength, size(θ, 1) + 2)
    for i ∈ 1:burnin+chainlength
        θᵗ = proposal2(θ) # new trial value
        Lₙθᵗ = Lₙ(θᵗ) # Objective at trial value
        # Accept / reject trial value
        accept = rand() < exp(Lₙθᵗ - Lₙθ)
        if accept
            # Replace values
            θ = θᵗ
            Lₙθ = Lₙθᵗ
            # Increment number of accepted values
            naccept += 1
        end
        # Add to chain if burnin is passed
        # @info "current log-L" Lₙθ
        if i > burnin
            chain[i-burnin,:] = vcat(θ, accept, Lₙθ)
        end
        # Report
        if verbosity > 0 && mod(i, verbosity) == 0
            acceptance_rate = naccept / verbosity
            @info "Current parameters (iteration i=$i)" round.(θ, digits=3)' acceptance_rate
            naccept = 0
        end
    end
    return chain
end
