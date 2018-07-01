# Prepare a solvehook to solve via a Lagrangian relaxation in SDDP

function make_dual_infeasible(sense::Symbol, pi::Float64, x::Float64)
    if sense == :Max
        y = 1.0 - x
    else
        y = x
    end
    if y ≈ 0.0
        perturbup(pi)
    else
        perturbdown(pi)
    end
end

function perturbup(pi::Float64)
    if pi > 1e-6
        pi * 1.5
    elseif pi < -1e-6
        0.0
    else
        1.0
    end
end
function perturbdown(pi::Float64)
    if pi < -1e-6
        pi * 1.5
    elseif pi > 1e-6
        0.0
    else
        -1.0
    end
end

function initialize_dual_one!(sp::JuMP.Model, π0::Vector{Float64})
    # Actual function evaluated at given state
    actual_bound = getobjectivevalue(sp)
    l = lagrangian(sp)
    sense = getobjectivesense(sp)
    @inbounds for i = 1:length(l.constraints)
        # Flip the RHS
        idx = l.constraints[i].idx
        flip = 1.0 - sp.linconstr[idx].lb
        sp.linconstr[idx].lb = sp.linconstr[idx].ub = flip
        # Get one numerical gradient (doesn't say anything about function elsewhere)
        @assert solve(sp) == :Optimal
        diff = getobjectivevalue(sp) - actual_bound
        numerical_gradient = diff * (2.0 * flip - 1.0)
        # Change RHS back
        sp.linconstr[idx].lb = sp.linconstr[idx].ub = 1.0 - flip
        # Make dual infeasible by considering our one other evaluation
        constr = sp.linconstr[idx]
        # The Lagrangian multiplier is the negative of the LP dual = rate of change
        π0[i] = -make_dual_infeasible(sense, numerical_gradient, constr.lb)
    end
end
function initialize_dual_all!(sp::JuMP.Model, π0::Vector{Float64})
    # Actual function evaluated at given state
    actual_bound = getobjectivevalue(sp)
    # Flip the RHS of all constraints corresponding to all state variables
    l = lagrangian(sp)
    C = length(l.constraints)
    flip = zeros(C)
    @inbounds for i = 1:C
        idx = l.constraints[i].idx
        flip[i] = 1.0 - sp.linconstr[idx].lb
        sp.linconstr[idx].lb = sp.linconstr[idx].ub = flip[i]
    end
    # Get one numerical gradient (doesn't say anything about function elsewhere)
    @assert solve(sp) == :Optimal
    sense = getobjectivesense(sp)
    diff = getobjectivevalue(sp) - actual_bound
    @inbounds for i = 1:length(l.constraints)
        idx = l.constraints[i].idx
        numerical_gradient = diff * (2.0 * flip[i] - 1.0)
        # Change RHS back
        sp.linconstr[idx].lb = sp.linconstr[idx].ub = 1.0 - flip[i]
        # Make dual infeasible by considering our one other evaluation
        constr = sp.linconstr[idx]
        # The Lagrangian multiplier is the negative of the LP dual = rate of change
        π0[i] = -make_dual_infeasible(sense, numerical_gradient, constr.lb)
    end
end

"""
    initial_dual(sp::JuMP.Model, π0::Vector{Float64})

Finds an infeasible dual for `sp`.
"""
function initialize_dual!(sp::JuMP.Model, π0::Vector{Float64}, use_one_neighbors::Bool=true)
    if use_one_neighbors
        initialize_dual_one!(sp, π0)
    else
        initialize_dual_all!(sp, π0)
    end
end

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
    integeroptimality::Int
    lagrangian::Int
end
"""
    Pattern(;benders=0, strengthened_benders=0, integer_optimality=0, lagrangian=1)

Construct a cut pattern.

# Example:
Pattern(benders=0, strengthened_benders=1, integer_optimality=0, lagrangian=4)
means: in every cycle of 5 iterations, add 1 strengthened benders cut and 4
lagrangian cuts.
"""
function Pattern(;benders=0, strengthened_benders=0, integer_optimality=0, lagrangian=1)
    @assert benders >= 0 && strengthened_benders >= 0 && integer_optimality >= 0 && lagrangian >= 0
    strengthened_benders += benders
    integer_optimality += strengthened_benders
    lagrangian += integer_optimality
    @assert lagrangian > 0
    IncreasingPattern(benders, strengthened_benders, integer_optimality, lagrangian)
end

function getcuttype(iteration::Int, p::IncreasingPattern)
    if mod(iteration, p.lagrangian) + 1 <= p.benders
        return :benders
    elseif mod(iteration, p.lagrangian) + 1 <= p.strengthenedbenders
        return :strengthened_benders
    elseif mod(iteration, p.lagrangian) + 1 <= p.integeroptimality
        return :integer_opt
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
        elseif cuttype == :integer_opt
            status = JuMP.solve(sp, ignore_solve_hook=true)
            Q = getobjectivevalue(sp)
            # Need a bound on the future cost at every point other than the
            # current state. The cached problem bound will do.
            L = SDDP.ext(sp).problembound
            for s in SDDP.states(sp)
                x0 = sp.linconstr[s.constraint.idx].terms.vars[1]
                idx = s.constraint.idx
                if getvalue(x0) > 0.5
                    sp.linconstrDuals[idx] = Q - L
                else
                    sp.linconstrDuals[idx] = L - Q
                end
            end
        else
            # Update initial bound of the dual problem
            @assert solve(sp) == :Optimal
            l.method.initialbound = getobjectivevalue(sp)
            # Slacks have a new RHS each iteration, so update them
            l.slacks = getslack.(l.constraints)
            # Somehow choose duals to start with
            π0 = zeros(length(l.constraints)) # or rand, or ones
            initialize_dual!(sp, π0)
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
