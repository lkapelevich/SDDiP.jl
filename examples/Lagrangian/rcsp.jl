# Resource constrained shortest path problem

using SDDiP, JuMP, Ipopt, Base.Test, GLPKMathProgInterface

const JuMPversion = Pkg.installed("JuMP")

immutable RCSPData
    A::Array{Int,2}
    costs::Vector{Float64}
    resources::Vector{Float64}
    capacity::Float64
end

function getdata(arcs::Vector{Tuple{Int, Int}}, costs::Vector{Float64}, resources::Vector{Float64}, capacity::Float64)
    narcs = length(arcs)
    @assert narcs == length(costs) == length(resources)
    nnodes = maximum(maximum.(arcs))
    A = zeros(nnodes, narcs)
    for (i, a) in enumerate(arcs)
        A[a[1], i] = 1
        A[a[2], i] = -1
    end
    RCSPData(A, costs, resources, capacity)
end

function getmodel(arcs::Vector{Tuple{Int, Int}}, costs::Vector{Float64}, resources::Vector{Float64}, capacity::Float64)
    data = getdata(arcs, costs, resources, capacity)

    nnodes, narcs = size(data.A)

    m = Model(solver=GLPKSolverMIP())
    # Include arc or don't include arc
    @variable(m, 0 <= x[i=1:narcs] <= 1, Int)
    # Flow in = flow out
    @constraint(m, data.A * x .== 0)
    # Something has to flow
    @constraint(m, x[end] == 1)
    # Resource constraint
    complicating = @constraint(m, dot(data.resources, x) <= data.capacity)
    # Minimise costs
    @objective(m, :Min, dot(data.costs, x))

    return m, complicating
end

arcs = [
(1, 2)
(1, 3)
(2, 5)
(1, 4)
(1, 6)
(2, 3)
(2, 4)
(3, 5)
(3, 6)
(3, 7)
(4, 5)
(4, 7)
(5, 7)
(6, 7)
(7, 1) # last arc is a dummy arc
]

costs = [12; 12; 20; 18; 11; 12; 8; 1; 6; 10; 3; 9; 4; 5; 0]
resources = [5; 18; 3; 12; 24; 3; 4; 6; 1; 10; 7; 16; 14; 16; 0]
capacity = 32.0

# Sove MIP
model, complicating = getmodel(arcs, float(costs), float(resources), capacity)
@assert solve(model, relaxation=false) == :Optimal
println("The MIP objective is: ", getobjectivevalue(model))

# Solve using a linear relaxation
@assert solve(model, relaxation=true) == :Optimal
lpbound = getobjectivevalue(model)
println("The LP relaxation objective is: ", lpbound)

# ========================================================
# Solve using a Lagrangian relaxation

# It doesn't matter whether we relax integrality as well
if JuMPversion < v"0.17"
    for xx in getvariable(model, :x)
        setcategory(xx, :Cont)
    end
else
    for xx in model[:x]
        setcategory(xx, :Cont)
    end
end
setsolver(model, GLPKSolverLP())

println("\n**** Solved using the subgradient method. **** ")

π0 = [0.2]
dualbound = 50.0
relaxed_bound = 100.0;

subgradient = SubgradientMethod(initialbound=dualbound, wait=20)
RCSPdata    = LinearProgramData(model.obj,
                                [complicating],
                                [relaxed_bound],
                                method=subgradient)

status, bestvalue = lagrangiansolve!(RCSPdata, model, π0)
# (Although we could have solved the relaxed problem much more efficiently, it's a shortest path)
println("The optimal value is $bestvalue")
println("and the best multiplier is $π0.")

# In this problem the Lagrangian and LP relaxation bounds are the same
@test isapprox(bestvalue, lpbound, atol=1e-6)
@test isapprox(π0[1], 0.5, atol=1e-4)
@test isapprox(RCSPdata.old_bound[1], 32.0, atol=1e-9)
constr = model.linconstr[complicating.idx]
@test isapprox(getdual(complicating), -0.5, atol=1e-4)
@test isapprox(constr.ub, 32.0, atol=1e-9)
@test constr.lb == -Inf

println("\n**** Solved using the level method. **** ")

levelmethod = LevelMethod(initialbound=dualbound, quadsolver=IpoptSolver(print_level=0))
RCSPdata    = LinearProgramData(model.obj,
                                [complicating],
                                [relaxed_bound],
                                method=levelmethod)

status, bestvalue = lagrangiansolve!(RCSPdata, model, π0)
println("The optimal value is $bestvalue")
println("and the best multiplier is $π0.")

# In this problem the Lagrangian and LP relaxation bounds are the same
@test isapprox(bestvalue, lpbound, atol=1e-9)
@test isapprox(π0[1], 0.5, atol=1e-4)
