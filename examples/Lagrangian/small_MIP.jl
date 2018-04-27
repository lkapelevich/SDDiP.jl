using JuMP, Ipopt, SDDiP, GLPKMathProgInterface, Base.Test

const JuMPversion = Pkg.installed("JuMP")

type MIPData
    A::Array{Float64,2}
    b::Array{Float64,1}
    c::Array{Float64,1}
    C::Array{Float64,2}
    d::Array{Float64,1}
end

function MIPData()
  # Easy constraints Ax = b
  A = float([2	1	0	1	0	0	1	1	2;
            1	0	0	1	1	1	0	2	1;
            1	0	0	2	1	2	1	1	0;
            1	1	1	0	1	2	1	1	1
            ])
  b = float([4; 2; 2; 3])
  # Objective is c'x
  c = float([3;	7;	1;	7;	4;	3;	3;	8;	4])
  # Complicating constraints are Cx = d
  C = float([0	2	1	0	2	1	1	1	0;
            2	1	1	2	1	0	1	1	1])
  d = float([1; 2])

  MIPData(A, b, c, C, d)
end

function getmodel(data::MIPData)

    data = MIPData()
    N = size(data.A, 2)

    m=Model(solver=GLPKSolverMIP())
    @variable(m, x[1:N], Bin)
    @constraints(m, begin
        # Easy constraints
        dot(data.A[1,:], x) <= data.b[1]
        dot(data.A[2,:], x) == data.b[2]
        dot(data.A[3,:], x) >= data.b[3]
        dot(data.A[4,:], x) <= data.b[4]
        # Complicating constraints
        c1, dot(data.C[1,:], x) == 1
        c2, dot(data.C[2,:], x) >= 2
    end)
    @objective(m, :Min, dot(data.c, x))

    m, [c1; c2]
end

function solve_simple_MIP()

    data = MIPData()

    # If we were to go ahead and solve the MIP:
    model, complicating_constrs = getmodel(data)
    solve(model)
    println("MIP objective:", getobjectivevalue(model))

    # Solve via a relaxation:

    # Guess some initial duals
    π0 = -[15.0; 20.0]
    # Pick some relaxed bounds on the == and >= constraints
    relaxed_bounds = [100.0; -100.0]
    # Pick a starting upper bound
    objectivebound = 25.0

    # Use the Level Method
    levelmethod = LevelMethod(initialbound=objectivebound, quadsolver=IpoptSolver(print_level=0))
    # Data for the Lagrangian solver
    MIPdata = LinearProgramData(model.obj,             # objective
                                complicating_constrs,  # relaxed constraints
                                relaxed_bounds,        # RHS of relaxed constraints
                                method=levelmethod)    # method to solve
    # Solve
    status, bestvalue = lagrangiansolve!(MIPdata, model, π0)
    @test status == :Optimal
    println("\n**** Solved using the level method. **** ")
    println("The optimal value is $bestvalue")
    println("and the best multipliers are $(π0).")

    c1, c2 = complicating_constrs[1], complicating_constrs[2]
    @test isapprox(model.linconstr[c1.idx].ub, 1, atol=1e-9)
    @test isapprox(model.linconstr[c1.idx].lb, 1, atol=1e-9)
    @test isapprox(model.linconstr[c2.idx].lb, 2, atol=1e-9)
    @test isapprox(getobjectivevalue(model), 6.0, atol=1e-9)
    if JuMPversion < v"0.17"
        @test isapprox(getvalue(dot(data.c, getvariable(model, :x))), 6.0, atol=1e-9)
    else
        @test isapprox(getvalue(dot(data.c, model[:x])), 6.0, atol=1e-9)
    end

    # Use subgradient descent
    π0 = -[15.0; 20.0]
    subgradient = SubgradientMethod(initialbound=objectivebound)
    # Data for the Lagrangian solver
    MIPdata = LinearProgramData(model.obj,            # objective
                                complicating_constrs, # relaxed constraints
                                relaxed_bounds,       # relaxed RHS of relaxed constraints
                                method=subgradient)   # method to solve

    # Solve
    status, bestvalue = lagrangiansolve!(MIPdata, model, π0)
    @test status == :Optimal
    println("\n**** Solved using subgradient descent. **** ")
    println("The optimal value is $bestvalue")
    println("and the best multipliers are $π0.")
    @test isapprox(getobjectivevalue(model), 6.0, atol=1e-9)
    if JuMPversion < v"0.17"
        @test isapprox(getvalue(dot(data.c, getvariable(model, :x))), 6.0, atol=1e-9)
    else
        @test isapprox(getvalue(dot(data.c, model[:x])), 6.0, atol=1e-9)
    end

    # Maximisation check
    data.c *= (-1)
    setobjectivesense(model, :Max)
    objectivebound = -25.0
    π0 = [15.0; 20.0]
    subgradient   = SubgradientMethod(initialbound=objectivebound)
    MIPdata = LinearProgramData(-model.obj,           # objective
                                complicating_constrs, # relaxed constraints
                                relaxed_bounds,       # relaxed RHS of relaxed constraints
                                method=subgradient)   # method to solve
    status, bestvalue = lagrangiansolve!(MIPdata, model, π0)
    @test status == :Optimal
    println("\n**** Formulated as a maximisation and solved using subgradient descent. **** ")
    println("The optimal value is $bestvalue")
    println("and the best multipliers are $π0.")
    @test isapprox(getobjectivevalue(model), -6.0, atol=1e-9)
    if JuMPversion < v"0.17"
        @test isapprox(getvalue(dot(data.c, getvariable(model, :x))), -6.0, atol=1e-9)
    else
        @test isapprox(getvalue(dot(data.c, model[:x])), -6.0, atol=1e-9)
    end

end

solve_simple_MIP()
