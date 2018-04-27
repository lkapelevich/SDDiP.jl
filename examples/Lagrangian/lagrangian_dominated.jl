# Example from Zhou et al., 2016, Stochastic Dual Integer Programming
#

using JuMP, Gurobi, SDDiP

# ===================================================
# P1
m1 = Model(solver=GurobiSolver(OutputFlag=0))
@variables(m1, begin
    x1, Bin
    x2, Bin
end)

@objective(m1, Min, x1 + x2)

# ===================================================
# P2
function Q(z1, z2)
    m2 = Model(solver=GurobiSolver(OutputFlag=0))
    @variables(m2, begin
        y, Int
        0 <= x1 <= 1
        0 <= x2 <= 1
    end)
    @constraints(m2, begin
        y <= 4
        y >= 2.6 - 0.5x2 - 0.25x1
    end)

    c1 = @constraint(m2, x1 == z1)
    c2 = @constraint(m2, x2 == z2)

    @objective(m2, Min, 4y)
    m2, c1, c2

end
# L(π)
function L(π1, π2, z1, z2)
    m3 = Model(solver=GurobiSolver(OutputFlag=0))
    @variables(m3, begin
        y, Int
        0 <= x1 <= 1
        0 <= x2 <= 1
    end)
    @constraints(m3, begin
        y <= 4
        y >= 2.6 - 0.5x2 - 0.25x1
    end)
    @objective(m3, Min, 4y - π1 * x1 - π2 * x2)
    m3
end

# ===================================================
# Solve the first problem
solve(m1)
z1, z2 = getvalue(x1), getvalue(x2)
# ===================================================
# Pass values to the second problem
m1, c1, c2 = Q(z1, z2)
# ===================================================
# Solve the second problem using a Lagrangian solver
method = LevelMethod(initialbound = 100.0, quadsolver=GurobiSolver(OutputFlag=0))
# method = SubgradientMethod(100.0) # gap
lp     = LinearProgramData(m1.obj, [c1; c2], [100.0; 100.0], method=method)
π = [0.0; 0.0]
val, status = lagrangiansolve!(lp, m1, π)

println("With Lagrangian solver")
println("Objective at (0,0): ", getobjectivevalue(m1))
println("Duals are:")
println(m1.linconstrDuals[c1.idx])
println(m1.linconstrDuals[c2.idx])

# Note the primal is degenerate
# ===================================================
# With Benders
println("With Benders")
m2, c1, c2 = Q(z1, z2)
solve(m2, relaxation=true)
println("Objective at (0,0): ", getobjectivevalue(m2))
println("Duals are:")
println(m2.linconstrDuals[c1.idx])
println(m2.linconstrDuals[c2.idx])

# ===================================================
# With strengthened Benders
# z1, z2 = 1, 1
π1, π2 = m2.linconstrDuals[c1.idx], m2.linconstrDuals[c2.idx]
# π1, π2 = 0, -4
m3 = L(π1, π2, z1, z2)
solve(m3)
# println(getobjectivevalue(m3) - (π1 * z1 + π2 * z2))
println("Strengthened benders")
println("Objective at (0,0): ", getobjectivevalue(m3))
