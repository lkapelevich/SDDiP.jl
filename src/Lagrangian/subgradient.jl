#
# Subgradient descent types and methods.
#
"""
    SubgradientMethod

The parameters for solving a Lagrangian dual using subgradients.

initialbound   Initial upper or lower bound for the Lagrangian dual
tol            Tolerance for terminating
wait           A parameter for reducing step sizes
maxit          A cap on iterations
"""
immutable SubgradientMethod <: AbstractLagrangianMethod
    initialbound::Float64
    tol::Tolerance
    wait::Int
    maxit::Int
end
function SubgradientMethod(initialbound::Float64; tol=Unit(1e-6), wait=30, maxit=1e4)
    SubgradientMethod(initialbound, tol, wait, maxit)
end

function correctsign!(π::Array{Float64}, senses::Array{Symbol})
    for (i, sense) in enumerate(senses)
        if sense == :le
            π[i] = abs(π[i])
        elseif sense == :ge
            π[i] = -abs(π[i])
        end
    end
end

"""
    lagrangian_method!(lp::LinearProgramData{SubgradientMethod}, m::JuMP.Model, π::Vector{Float64})

Solve the Lagrangian dual of a linear program using subgradient descent.

# Arguments
* lp        Information about the primal problem
* m         The primal prbolem
* π         Initial iterate

# Returns
* status, objective, and modifies π
"""
function lagrangian_method!(lp::LinearProgramData{SubgradientMethod}, m::JuMP.Model, π::Vector{Float64})

    # old_solvehook = m.solvehook
    # JuMP.setsolvehook(m, specialised_solve)
    # then JuMP.setsolvehook(m, old_solvehook)

    subgradient = lp.method
    tol = subgradient.tol

    # Stepping parameter
    μ = 2.
    incumbent = subgradient.initialbound
    # A bound, the strongest bound so far, and a bound we can check against every lp.wait iterations
    bound = best_bound = cached_bound = -Inf

    iteration = 0
    timesunchanged = 0

    reversed = false
    if getobjectivesense(m) == :Max
        incumbent *= -1
        lp.obj *= -1
        setobjectivesense(m, :Min)
        π .= -π
        reversed = true
    end

    correctsign!(π, lp.senses) # if user started with the wrong signs

    while iteration < subgradient.maxit

        iteration += 1

        # Wait for some number of iterations before checking if we improved
        if mod(iteration, subgradient.wait) == 0
            # If we found a stronger lower bound
            if cached_bound < best_bound
                cached_bound = best_bound
            else
                # If we didn't improve, take smaller steps
                μ /= 2
                timesunchanged += 1
            end
        end

        # Let L = f + πᵀ(Ax-b), set this as the objective in the original model
        m.internalModelLoaded = false
        bound, direction = solve_primal(m, lp, π)
        # Update if we found a stronger bound
        (bound > best_bound) && (best_bound = bound)

        # Check if we have a new incumbent
        feasible = true
        for (slack, sense) in zip(direction, lp.senses)
            if !issatisfied(slack, sense)
                feasible = false
                break
            end
        end

        if feasible
            # The value of cᵀx, the primal objective, is an upper bound
            candidate = getvalue(lp.obj)
            # Check πᵀ(Ax-b) = candidate - bound
            if isclose(candidate, bound, tol) || timesunchanged > 30 # need to think about how to terminate with duality gap
                # Complimentary slackness => we are done
                if reversed
                    # Undo transformation
                    π .*= -1
                    lp.obj *= -1
                    setobjectivesense(m, :Max)
                    return :Optimal, -bound
                else
                    return :Optimal, bound
                end
            end
            # Update incumbent if we strictly improved
            (candidate < incumbent) && (incumbent = candidate)
        end

        # Calculate the stepsize
        step = μ * (incumbent - bound) / sum(direction.^2)
        # Update multipliers
        for i = eachindex(π)
            π[i] += step * direction[i]
        end
        # Set correct sign for all multipliers
        for (i, s) in enumerate(lp.senses)
            if s == :le
                π[i] = max(0, π[i])
            elseif s == :ge
                π[i] = min(0, π[i])
            end
        end
    end
    return :IterationLimit, bound
    warn("Lagrangian relaxation did not solve properly.")
end
