
function strengthen_kelley!(lp::LinearProgramData{KelleyMethod{T}},
    m::JuMP.Model, π::Vector{Float64}, approx_model::JuMP.Model, i::Int) where T

    kelleys = lp.method
    N      = length(π)
    tol    = kelleys.tol

    # Gap between approximate model and true function each iteration
    gap = Inf
    # Let's make new storage for the best multiplier found so far
    bestmult = copy(π)
    # Dual problem has the opposite sense to the primal
    dualsense = Lagrangian.getdualsense(m)
    fdash = zeros(N)
    if dualsense == :Min
        best_actual, f_actual, f_approx = Inf, Inf, -Inf
    else
        best_actual, f_actual, f_approx = -Inf, -Inf, Inf
    end

    iteration = 0
    θ, x = approx_model[:θ], approx_model[:x]

    while iteration < kelleys.maxit
        iteration += 1
        # Evaluate the real function and a subgradient
        f_actual, fdash = Lagrangian.solve_primal(m, lp, π)
        # Make the dual strong in all but one dimension
        if i != length(fdash) && abs(fdash[i]) > 1e-6
            if iteration == 1
                # c1 = @constraint(approx_model, x[i] == π[i] + (kelleys.initialbound - f_actual) / fdash[i])
            else
                # JuMP.setRHS(c1, π[i] - (f_actual - kelleys.initialbound) / fdash[i])
            end
        end
        # println(m)
        # @show f_actual, fdash
        # println(approx_model)

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
        if solve(approx_model) != :Optimal
            println(approx_model)
            @show kelleys.initialbound, f_actual, π, fdash
            error()
        end
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
        if Lagrangian.closetozero(gap, best_actual, f_approx, tol) || Lagrangian.isclose(norm(fdash), 0.0, tol)
            π .= bestmult
            if dualsense == :Min
                π .*= -1 # bestmult not the same as getvalue(x), approx_model may have just gotten lucky
            end
            # @show bestmult
            return :Optimal, best_actual::Float64
        end

        # Get the next iterate
        π .= getvalue(x)
    end
    warn("Lagrangian relaxation did not solve properly.")
    return :IterationLimit, f_approx::Float64

end
