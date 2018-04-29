#
# Subgradient descent types and methods.
#
"""
    BinaryMethod

The parameters for solving a Lagrangian dual using subgradients.

initialbound   Initial upper or lower bound for the Lagrangian dual
tol            Tolerance for terminating
"""
mutable struct BinaryMethod{T<:Tolerance} <: AbstractLagrangianMethod
    initialbound::Float64
    tol::T
end
function BinaryMethod(; initialbound=0.0, tol=Unit(1e-6))
    BinaryMethod(initialbound, tol)
end

"""
    lagrangian_method!(lp::LinearProgramData{BinaryMethod}, m::JuMP.Model, π::Vector{Float64})

# Arguments
* lp        Information about the primal problem
* m         The primal problem
* π         Initial iterate

# Returns
* status, objective, and modifies π
"""
function lagrangian_method!{T}(lp::LinearProgramData{BinaryMethod{T}}, m::JuMP.Model, π::Vector{Float64})

    # We know what the objective is
    mipobj = lp.method.initialbound

    tol = lp.method.tol

    # To make things easier, make primal always min, dual always max
    reversed = false
    if getobjectivesense(m) == :Max
        lp.obj *= -1
        mipobj *= -1
        setobjectivesense(m, :Min)
        reversed = true
    end

    n = length(π)
    π .= 0.0
    # @show π
    # we will sample the Lagrangian at pi = 0
    bound, direction = solve_primal(m, lp, π)
    # this setup makes it possible to cycle when 0 is one of many
    # subgradients. need to test not if abs(direction[i]) < 1e-6,
    # but if meeting relaxed i^th constraint doesn't change objective
    # iter = 0
    # lockin = Int[]
    while !isclose(mipobj, bound, tol)
        # iter += 1
        # @show direction
        # @show iter
        # Santity check
        @assert bound <= mipobj + 1e-6
        @inbounds for i = 1:n
            # (i in lockin) && continue
            # If dual=0 is feasible, let it be TODO: improve on this.
            if abs(direction[i]) < 1e-6
                # push!(lockin, i)
                continue
            end
            π[i] += 1 / direction[i] * (mipobj - bound)
        end
        bound, direction = solve_primal(m, lp, π)
    end
    # Undo transformation if needed
    if reversed
        π .*= -1
        lp.obj *= -1
        setobjectivesense(m, :Max)
        return :Optimal, -bound
    else
        return :Optimal, bound
    end
end
