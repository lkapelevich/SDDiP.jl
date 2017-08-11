module Lagrangian
using Compat, JuMP

export
# Solve for the Lagrangian duals
lagrangian_method!,
# Relax, solve, then store
lagrangiansolve!,
# Structs to implement methods
LevelMethod,
SubgradientMethod,
# Structs to solve the primal
LinearProgram,
# Each method needs to know about the problem being solved
LinearProgramData,
# Utilities
getslack,
isclose, closetozero,
Absolute, Relative, Unit

@compat abstract type AbstractLagrangianMethod end
@compat abstract type AbstractProblemClass end
immutable LinearProgram <: AbstractProblemClass end

type LinearProgramData{M<:AbstractLagrangianMethod, C<:AbstractProblemClass}
    obj::QuadExpr                                # objective
    @compat constraints::Vector{<:ConstraintRef} # constraints being relaxed
    relaxed_bounds::Vector{Float64}             # bounds on constraints when we relax them
    senses::Vector{Symbol}                      # we will cache the sense of constraints
    slacks::Vector{AffExpr}                     # also cache Ax-b
    old_bound::Vector{Float64}                  # cache before relaxing constraints
    method::M                                   # method parameters
    pc::C                                       # problem class
end

include("utils.jl")
include("levelmethod.jl")
include("subgradient.jl")

"""
    LinearProgramData(obj::QuadExpr, constraints::Vector{<:ConstraintRef}, relaxed_bounds::Vector{Float64}; method=LevelMethod(), problem_class=LinearProgram())

Creates a `LinearProgramData` object for calling `lagrangiansolve!`.

# Arguments
* obj:               Objective of the linear program without relaxation.
* constraints:       A vector of type `JuMP.ConstraintRef` with the contraints to be relaxed.
* relaxed_bounds:    A RHS that will replace the RHS of the constraint being relaxed (choose a big number for ≤ or = constraints, and something small for ≥).
* method:            Solving parameters, of type `AbstractLagrangianMethod`.
* problem_class:     To overload how the primal problem is solved.
"""
@compat function LinearProgramData(obj::QuadExpr, constraints::Vector{<:ConstraintRef}, relaxed_bounds::Vector{Float64}; method=LevelMethod(), problem_class=LinearProgram())
    @assert length(constraints) == length(relaxed_bounds)
    LinearProgramData(obj, constraints,
        relaxed_bounds,
        sense.(constraints),
        getslack.(constraints),
        zeros(Float64, length(constraints)),
        method,
        problem_class
    )
end

function relaxandcache!(l::LinearProgramData, m::JuMP.Model)
    for (i, c) in enumerate(l.constraints)
        con = m.linconstr[c.idx]
        if l.senses[i] == :le
            l.old_bound[i] = con.ub
            con.ub = l.relaxed_bounds[i]
        elseif l.senses[i] == :ge
            l.old_bound[i] = con.lb
            con.lb = l.relaxed_bounds[i]
        else
            l.old_bound[i] = con.lb
            con.ub =  l.relaxed_bounds[i]
            con.lb = -Inf
        end
    end
end
function recover!(l::LinearProgramData, m::JuMP.Model, π::Vector{Float64})
    for (i, c) in enumerate(l.constraints)
        con = m.linconstr[c.idx]
        m.linconstrDuals[c.idx] = -π[i]
        if l.senses[i] == :le
            con.ub = l.old_bound[i]
        elseif l.senses[i] == :ge
            con.lb = l.old_bound[i]
        else
            con.lb = con.ub = l.old_bound[i]
        end
    end
end

"""
    lagrangian_method!{M<:AbstractLagrangianMethod}(l::LinearProgramData{M}, m::JuMP.Model, π::Vector{Float64})

Solve the Lagrangian dual problem.
"""
lagrangian_method!{M<:AbstractLagrangianMethod}(l::LinearProgramData{M}, m::JuMP.Model, π::Vector{Float64}) = error("No solve method defined.")

"""
    lagrangiansolve!{M<:AbstractLagrangeinMethod}(l::LinearProgramData{M}, m::JuMP.Model, π::Vector{Float64})

Relax the primal, solve the Lagrangian dual, reset the primal.
"""
function lagrangiansolve!(l::LinearProgramData, m::JuMP.Model, π::Vector{Float64})

    @assert length(π) == length(l.constraints)
    @assert !isempty(l.constraints)

    # Relax bounds, cache old bounds
    relaxandcache!(l, m)

    # Solve
    status, obj = lagrangian_method!(l, m, π)
    @assert status == :Optimal

    # Set objective
    m.objVal = obj
    # JuMP.setobjective(m, getobjectivesense(m), l.obj) # bypass quad terms prep
    m.obj = l.obj

    # Set duals and reset constraint bounds
    recover!(l, m, π)

    return status, obj
end

end # module
