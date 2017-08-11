function sense(c::ConstraintRef)
    constr = c.m.linconstr[c.idx]
    if constr.lb == constr.ub
        return :eq
    elseif constr.lb == -Inf
        return :le
    elseif constr.ub == Inf
        return :ge
    end
    error("Range constraint not supported.")
end

function getslack(c::ConstraintRef)
    constr = c.m.linconstr[c.idx]
    if constr.ub == Inf
        return constr.terms - constr.lb
    else
        return constr.terms - constr.ub
    end
end

getdualsense(m::JuMP.Model) = getobjectivesense(m)==:Min? :Max : :Min

function issatisfied(slack::Float64, sense::Symbol, tol=1e-9)
    if sense == :eq
        return abs(slack) < tol
    elseif sense == :le
        return slack < tol
    elseif sense == :ge
        return slack > -tol
    end
    error("Unknown sense $(sense).")
end

@compat abstract type Tolerance end
immutable Absolute <: Tolerance
    val::Float64
end
immutable Relative <: Tolerance
    val::Float64
end
immutable Unit <: Tolerance
    val::Float64
end
function isclose(f1::Float64, f2::Float64, tol::Tolerance)::Bool
    closetozero(f1 - f2, f1, f2, tol)
end
function closetozero(gap::Float64, ::Float64, ::Float64, tol::Absolute)::Bool
    return abs(gap) < tol.val
end
function closetozero(gap::Float64, f1::Float64, f2::Float64, tol::Relative)::Bool
    if min(abs(f1), abs(f2)) < 1e-6
        warn("Switching to unit tolerance test.")
        return closetozero(gap, f1, f2, Unit(tol.val))
    end
    return abs(gap) / min(abs(f1), abs(f2)) < tol.val
end
function closetozero(gap::Float64, f1::Float64, f2::Float64, tol::Unit)::Bool
    return abs(gap) / (1 + min(abs(f1), abs(f2))) < tol.val
end

const PrimalOutput = Tuple{Float64, Vector{Float64}}

# For a fixed π, solve minₓ{L = cᵀx + πᵀ(Ax-b)} or maxₓ{L = cᵀx - πᵀ(Ax-b)}
function solve_primal{M<:AbstractLagrangianMethod, C<:LinearProgram}(m::JuMP.Model, d::LinearProgramData{M, C}, π::Vector{Float64})
    # Set the Lagrangian the objective in the primal model
    if getobjectivesense(m) == :Min
        @objective(m, :Min, d.obj + dot(π, d.slacks))
        subgradient = d.slacks
    else
        @objective(m, :Max, d.obj - dot(π, d.slacks))
        subgradient = -d.slacks
    end
    @assert solve(m, ignore_solve_hook=true) == :Optimal
    getobjectivevalue(m), getvalue(subgradient)
end

immutable UnsetSolver <: JuMP.MathProgBase.AbstractMathProgSolver end
