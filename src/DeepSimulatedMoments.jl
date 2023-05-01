module DeepSimulatedMoments

using DifferentialEquations
using LinearAlgebra
using Flux
using StatsBase

include("DGPs/AbstractDGP.jl")
include("DGPs/MA2.jl")
include("DGPs/Logit.jl")
include("DGPs/GARCH.jl")
include("DGPs/JumpDiffusion.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/TCN.jl")

end