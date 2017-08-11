using Base.Test, SDDiP, JuMP, GLPKMathProgInterface, Ipopt

@testset "Lagrangian" begin
    include("Lagrangian.jl")
end

@testset "SDDiP" begin
    include("SDDiP.jl")
end

@testset "@binarystate" begin
    include("binarystate.jl")
end
