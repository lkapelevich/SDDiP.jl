#
# Kelley's method with lazy hyperplanes.
#
"""
    KelleyMethod

The parameters for solving a Lagrangian dual with Kelley's method.

# Arguments
* initialbound   Initial upper or lower bound for the Lagrangian dual
* tol            Tolerance: we stop when the gap between our approximation of the function and the actual function value is less than `tol`
* maxit          To terminate the method
"""
mutable struct KelleyMethod{T<:Tolerance} <: AbstractLagrangianMethod
    initialbound::Float64   # starting bound for the Lagrangian dual problem
    tol::T                  # tolerance for terminating
    maxit::Int              # a cap on iterations
end
function KelleyMethod(; initialbound=0.0, tol=Unit(1e-6), maxit=10_000)
    KelleyMethod(initialbound, tol, maxit)
end

"""
    lagrangian_method!(lp::LinearProgramData{KelleyMethod}, m::JuMP.Model, π::Vector{Float64})

Kelley's method with a lazy callback approach.

# Arguments
* lp        Information about the primal problem
* m         The primal problem
* π         Initial iterate

# Returns
* status, objective, and modifies π
"""
function lagrangian_method!(lp::LinearProgramData{KelleyMethod{T}}, m::JuMP.Model, π::Vector{Float64}) where T

    kelleys = lp.method
    N      = length(π)
    tol    = kelleys.tol

    # Gap between approximate model and true function each iteration
    gap = Inf
    # Let's make new storage for the best multiplier found so far
    bestmult = copy(π)
    # Dual problem has the opposite sense to the primal
    dualsense = getdualsense(m)
    fdash = zeros(N)

    # The approximate model will be a made from linear hyperplanes
    approx_model = Model(solver=m.solver)

    @variables approx_model begin
        θ                   # The objective of the approximate model
        x[i=1:N]            # The x variables in the approximate model are the Lagrangian duals.
    end
    # There are sign restrictions on some duals
    for (i, sense) in enumerate(lp.senses)
        if sense == :ge
            setupperbound(x[i], 0.0)
        elseif sense == :le
            setlowerbound(x[i], 0.0)
        end
    end
    # Let's not be unbounded from the beginning
    if dualsense == :Min
        setlowerbound(θ, kelleys.initialbound)
        best_actual, f_actual, f_approx = Inf, Inf, -Inf
    else
        setupperbound(θ, kelleys.initialbound)
        best_actual, f_actual, f_approx = -Inf, -Inf, Inf
    end

    iteration = 0

    while iteration < kelleys.maxit
        iteration += 1
        # Evaluate the real function and a subgradient
        f_actual, fdash = solve_primal(m, lp, π)

        # Improve the model and update best function value so far
        if dualsense == :Min
            @constraint(approx_model, θ >= f_actual + dot(fdash, x - π))
            if f_actual < best_actual
                best_actual = f_actual
                bestmult .= π
            end
        else
            @constraint(approx_model, θ <= f_actual + dot(fdash, x - π))
            # println(f_actual + dot(fdash, x - π))
            if f_actual > best_actual
                best_actual = f_actual
                bestmult .= π
            end
        end
        # Get a bound from the approximate model
        @objective(approx_model, dualsense, θ)
        @assert solve(approx_model) == :Optimal
        f_approx = getobjectivevalue(approx_model)::Float64
        # Check the gap
        gap = abs(best_actual - f_approx)
        #=
            Stop if best_actual ≈ f_approx
            Note: if fdash ≈ ̃0, then we expect that best_actual ≈ f_approx
            anyway. We still check for this condition in case the solver allows
            a component of ̃x to get very large, so that dot(fdash, x) is nonzero
            and f_approx is incorrect.
        =#
        if closetozero(gap, best_actual, f_approx, tol) || isclose(norm(fdash), 0.0, tol)
            π .= bestmult
            if dualsense == :Min
                π .*= -1 # bestmult not the same as getvalue(x), approx_model may have just gotten lucky
            end
            # @show bestmult
            return :Optimal, best_actual::Float64, approx_model
        end

        # Get the next iterate
        π .= getvalue(x)
    end
    warn("Lagrangian relaxation did not solve properly.")
    return :IterationLimit, f_approx::Float64, approx_model

end
