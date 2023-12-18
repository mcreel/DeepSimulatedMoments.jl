using DeepSimulatedMoments, Flux
#=
#function main()
dgp = MA2(100) # Create an MA(2) DGP with 100 observations
tcn = build_tcn(dgp) # Build a TCN for this DGP

# Set up the hyperparameters
hp = HyperParameters(
    validation_size=1_000, loss=rmse_conv, 
    print_every=10, nsamples=20, epochs=5,
)

# Create the moment network
net = MomentNetwork(
    tcn |> hp.dev, ADAMW(), hp
)

# Train the moment network
iterations, losses = train_network(net, dgp)


# draw  a true parameter, and some corresponding data 
data, θtrue  = generate(dgp,1)
θtrue = vec(θtrue)

# Do Bayesian MSM
θnn = make_moments(net, net.data_transform(data))[:] # get the net fit for real data
@show θnn
@show θtrue
=#
δ = 1.0  # proposal tuning
covreps = 1000 # sims for proposal covariance
S = 50 # sims per likelihood eval
# this next line not working, as param out of support
#_,Σₚ = simmomentscov(net, dgp, covreps, θnn) # get the proposal covariance
_,Σₚ = simmomentscov(net, dgp, covreps, θtrue) # get the proposal covariance

# do the MCMC sampling
# CHANGE true to nn, when working
mcmc(θnn, θnn, δ, Σₚ, S, net, dgp) # θnn is the start value for chain

#return net
#end

