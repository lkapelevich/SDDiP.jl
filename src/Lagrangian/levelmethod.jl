#
# Level Method types and methods.
#
"""
    LevelMethod

The parameters for solving a Lagrangian dual using the Level Method.

initialbound   Initial upper or lower bound for the Lagrangian dual
level          Level method parameter 0 ≥ λ ≥ 1
tol            Tolerance: we stop when the gap between our approximation of the function and the actual function value is less than `tol`
solver         A quadratic solver
maxit          To terminate the method
"""
immutable LevelMethod <: AbstractLagrangianMethod
    initialbound::Float64                           # starting bound for the Lagrangian dual problem
    level::Float64                                  # parameter between 0 and 1
    tol::Tolerance                                  # tolerance for terminating
    solver::JuMP.MathProgBase.AbstractMathProgSolver # should be a quadratic solver
    maxit::Int                                      # a cap on iterations
end
function LevelMethod(initialbound::Float64; level=0.5, tol=Unit(1e-6), quadsolver=UnsetSolver(), maxit=1e4)
    if quadsolver == UnsetSolver()
        error("You must specify a MathProgBase solver that can handle quadratic objective functions.")
    end
    if 0. <= level <= 1.
        return LevelMethod(initialbound, level, tol, quadsolver, maxit)
    else
        error("Level parameter must be between 0 and 1.")
    end
end

"""
    lagrangian_method!(lp::LinearProgramData{LevelMethod}, m::JuMP.Model, π::Vector{Float64})

The Level Method (Lemarechal, Nemirovskii, Nesterov, 1992).

# Arguments
* lp        Information about the primal problem
* m         The primal problem
* π         Initial iterate

# Returns
* status, objective, and modifies π
"""
function lagrangian_method!(lp::LinearProgramData{LevelMethod}, m::JuMP.Model, π::Vector{Float64})

    levelmethod = lp.method
    N = length(π)
    tol = levelmethod.tol

    # Gap between approximate model and true function each iteration
    gap = Inf
    # Dual problem has the opposite sense to the primal
    dualsense = getdualsense(m)
    # Let's make new storage for the best multiplier found so far
    bestmult = copy(π)

    # The approximate model will be a made from linear hyperplanes
    approx_model = Model(solver=levelmethod.solver)

    @variables approx_model begin
        θ                   # The objective of the approximate model
        x[i=1:N]            # The x variables in the approximate model are the Lagrangian duals.
    end
    # There are sign restrictions on some duals
    for (i, sense) in enumerate(lp.senses)
        if sense == :ge
            setupperbound(x[i], 0)
        elseif sense == :le
            setlowerbound(x[i], 0)
        end
    end
    # Let's not be unbounded from the beginning
    if dualsense == :Min
        setlowerbound(θ, levelmethod.initialbound)
        best_actual = Inf
    else
        setupperbound(θ, levelmethod.initialbound)
        best_actual = -Inf
    end

    iteration = 0

    while iteration < levelmethod.maxit
        iteration += 1
        # Evaluate the real function and a subgradient
        m.internalModelLoaded = false
        f_actual, fdash = solve_primal(m, lp, π)

        # Improve the model, undo level bounds on θ, and update best function value so far
        if dualsense == :Min
            @constraint(approx_model, θ >= f_actual - dot(fdash, π) + dot(fdash, x))
            setupperbound(θ, Inf)
            if f_actual < best_actual
                best_actual = f_actual
                bestmult .= π
            end
        else
            @constraint(approx_model, θ <= f_actual - dot(fdash, π) + dot(fdash, x))
            setlowerbound(θ, -Inf)
            if f_actual > best_actual
                best_actual = f_actual
                bestmult .= π
            end
        end
        # Get a bound from the approximate model
        approx_model.internalModelLoaded = false
        @objective(approx_model, dualsense, θ)
        @assert solve(approx_model) == :Optimal
        f_approx = getobjectivevalue(approx_model)
        # Check the gap
        gap = abs(best_actual - f_approx)
        # Stop if best_actual ≈ f_approx
        # Note: if fdash ≈ ̃0, then we expect that best_actual ≈ f_approx anyway.
        # We still check for this condition in case the solver allows a component
        # of ̃x to get very large, so that dot(fdash, x) is nonzero and f_approx is incorrect.
        if closetozero(gap, best_actual, f_approx, tol) || isclose(norm(fdash), 0.0, tol)
            if dualsense == :Min
                π .= -1 * bestmult # bestmult not the same as getvalue(x), approx_model may have just gotten lucky
            else
                π .= bestmult
            end
            return :Optimal, best_actual
        end
        # Form a level
        if dualsense == :Min
            level = f_approx + gap * lp.method.level + tol.val/10
            setupperbound(θ, level)
        else
            level = f_approx - gap * lp.method.level - tol.val/10
            setlowerbound(θ, level)
        end
        # Get the next iterate
        approx_model.internalModelLoaded = false
        @objective(approx_model, Min, sum((π[i]-x[i])^2 for i=1:N))
        @assert solve(approx_model) == :Optimal
        # Update π for this iteration
        π .= getvalue(x)
    end
    warn("Lagrangian relaxation did not solve properly.")
    return :IterationLimit, f_approx

end
