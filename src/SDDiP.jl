module SDDiP

export setSDDiPsolver!,
Pattern,
@binarystate, @binarystates

# Our Lagrangian solver
include(joinpath(dirname(@__FILE__),"Lagrangian","Lagrangian.jl"))

using SDDP, JuMP, Compat, Reexport
@reexport using .Lagrangian

include("solver.jl")
include("binary_expansion.jl")
include("binarystate.jl")
include("strengthening.jl")
end
