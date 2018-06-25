# Prepare a solvehook to solve via a Lagrangian relaxation in SDDP

# Overload how stage problems are solved in the backward pass
function SDDP.JuMPsolve(::Type{SDDP.BackwardPass}, m::SDDPModel, sp::JuMP.Model)
    if sp.solvehook == nothing
        JuMP.solve(sp)
    else
        JuMP.solve(sp, require_duals=true, iteration=length(m.log))
    end
end

lagrangian(m::JuMP.Model) = m.ext[:Lagrangian]

struct MixedSolvers
    LP::JuMP.MathProgBase.AbstractMathProgSolver
    MIP::JuMP.MathProgBase.AbstractMathProgSolver
end

# Just handy caching
struct IncreasingPattern
    benders::Int
    strengthenedbenders::Int
    lagrangian::Int
end
"""
    Pattern(;benders=0, strengthened_benders=0, lagrangian=1)

Construct a cut pattern.

# Example:
Pattern(benders=0, strengthened_benders=1, lagrangian=4) means: in every cycle
of 5 iterations, add 1 strengthened benders cut and 4 lagrangian cuts.
"""
function Pattern(;benders=0, strengthened_benders=0, lagrangian=1)
    @assert benders >= 0 && strengthened_benders >= 0 && lagrangian >= 0
    strengthened_benders += benders
    lagrangian += strengthened_benders
    @assert lagrangian > 0
    IncreasingPattern(benders, strengthened_benders, lagrangian)
end

function getcuttype(iteration::Int, p::IncreasingPattern)
    if mod(iteration, p.lagrangian) + 1 <= p.benders
        return :benders
    elseif mod(iteration, p.lagrangian) + 1 <= p.strengthenedbenders
        return :strengthened_benders
    else
        return :lagrangian
    end
end

# The solvehook
function SDDiPsolve!(sp::JuMP.Model; require_duals::Bool=false, iteration::Int=-1, kwargs...)
    @assert !(require_duals && iteration == -1)
    solvers = sp.ext[:solvers]
    if require_duals && SDDP.ext(sp).stage > 1
        # Update the objective we cache in case the objective has noises
        l = lagrangian(sp)
        l.obj = getobjective(sp)
        cuttype = getcuttype(iteration, sp.ext[:pattern])
        if cuttype == :benders
            # Solve linear relaxation
            setsolver(sp, solvers.LP)
            status = JuMP.solve(sp, ignore_solve_hook=true, relaxation=true)
        elseif cuttype == :strengthened_benders
            # Get the LP duals
            setsolver(sp, solvers.LP)
            @assert JuMP.solve(sp, ignore_solve_hook=true, relaxation=true) == :Optimal
            π = -SDDP.getdual.(SDDP.states(sp))
            # Update slacks because RHSs change each iteration
            l.slacks = getslack.(l.constraints)
            # Relax bounds to formulate Lagrangian
            Lagrangian.relaxandcache!(l, sp)
            # Change the MIP objective
            Lagrangian.setlagrangianobjective!(sp, l, π)
            # Solve the Lagrangian, with LP πs chosen and fixed
            setsolver(sp, solvers.MIP)
            status = solve(sp, ignore_solve_hook=true)
            # Undo changes
            sp.obj = l.obj
            Lagrangian.recover!(l, sp, π)
        else
            # Update initial bound of the dual problem
            @assert solve(sp) == :Optimal
            l.method.initialbound = getobjectivevalue(sp)
            # Slacks have a new RHS each iteration, so update them
            l.slacks = getslack.(l.constraints)
            # Somehow choose duals to start with
            π0 = zeros(length(l.constraints)) # or rand, or ones
            # Lagrangian objective and duals
            setsolver(sp, solvers.MIP)
            status, _ = lagrangiansolve!(l, sp, π0)
        end
        sp.obj = l.obj
    else
        # We are in the forward pass, or we are in stage 1
        setsolver(sp, solvers.MIP)
        status = JuMP.solve(sp, ignore_solve_hook=true)
    end
    status
end

"""
    setSDDiPsolver!(sp::JuMP.Model; method=Subgradient(0.0), pattern=Pattern(), MIPsolver=sp.solver, LPsolver=MIPsolver)

Sets a JuMP solvehook for integer SDDP to stage problem `sp` that will call a
a Lagrangian solver of type `method.` Argument `pattern` can be used to specify
a pattern of different cut types.

You should specify an LP/MIP solver if you are using different cut types in a cut
pattern, and you are not using a solver that can solve both MIPs and LPs.
"""
function setSDDiPsolver!(sp::JuMP.Model; method=Subgradient(0.0), pattern=Pattern(), MIPsolver=sp.solver, LPsolver=MIPsolver)

    constraints = SDDP.LinearConstraint[]
    for s in SDDP.states(sp)
        # xinₜ = xoutₜ₋₁ is being relaxed
        push!(constraints, s.constraint)
        # xout has to be binary
        setcategory(s.variable, :Bin)
        # Explicitly add bounds in case we relax integrality on some iterations
        setlowerbound(s.variable, 0.0)
        setupperbound(s.variable, 1.0)
        # Set upper/lower bounds on the incoming state
        x0 = sp.linconstr[s.constraint.idx].terms.vars[1]
        setlowerbound(x0, 0.0)
        setupperbound(x0, 1.0)
    end

    relaxed_bounds = ones(length(constraints))
    sp.ext[:Lagrangian] = LinearProgramData(QuadExpr(),         # objective
                                           constraints,         # relaxed constraints
                                           relaxed_bounds,      # RHS of relaxed constraints
                                           method=method)       # method to solve
    # Store pattern and solvers
    sp.ext[:pattern] = pattern
    sp.ext[:solvers] = MixedSolvers(LPsolver, MIPsolver)

    JuMP.setsolvehook(sp, SDDiPsolve!)
end
