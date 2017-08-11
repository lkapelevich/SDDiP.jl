@testset "Utils" begin
    @testset "Caching" begin
        m = Model()
        @variable(m, x)
        c = @constraint(m, x <= 8.)
        @test Lagrangian.sense(c) == :le
        @test Lagrangian.getslack(c) == AffExpr(x - 8)
        m.linconstr[c.idx].lb = 8.
        @test Lagrangian.sense(c) == :eq
        @test Lagrangian.getslack(c) == AffExpr(x - 8)
        m.linconstr[c.idx].ub = Inf
        @test Lagrangian.sense(c) == :ge
        @test Lagrangian.getslack(c) == AffExpr(x - 8)
        m.linconstr[c.idx].ub = 10.
        @test_throws Exception Lagrangian.sense(c)
        @test Lagrangian.getslack(c) == AffExpr(x - 10)
    end
    @testset "Satisfied" begin
        m = Model()
        x = @variable(m, [i=1:2])
        m.colVal[x[1].col] = 1.
        m.colVal[x[2].col] = 2.
        @test Lagrangian.issatisfied( 1e-9, :le, 1e-8) == true
        @test Lagrangian.issatisfied(-1e-9, :le, 1e-8) == true
        @test Lagrangian.issatisfied(  -1., :le, 1e-8) == true
        @test Lagrangian.issatisfied(   1., :le, 1e-8) == false
        @test Lagrangian.issatisfied( 1e-9, :eq, 1e-8) == true
        @test Lagrangian.issatisfied(-1e-9, :eq, 1e-8) == true
        @test Lagrangian.issatisfied(  -1., :eq, 1e-8) == false
        @test Lagrangian.issatisfied(   1., :eq, 1e-8) == false
        @test Lagrangian.issatisfied( 1e-9, :ge, 1e-8) == true
        @test Lagrangian.issatisfied(-1e-9, :ge, 1e-8) == true
        @test Lagrangian.issatisfied(  -1., :ge, 1e-8) == false
        @test Lagrangian.issatisfied(   1., :ge, 1e-8) == true
        @test_throws Exception Lagrangian.issatisfied(AffExpr(), :x)
    end
    @testset "LevelMethod" begin
        @test_throws Exception levelmethod = LevelMethod(0.0, level=2, quadsolver=GLPKSolverMIP())
        # Need to specify quadsolver
        @test_throws Exception levelmethod = LevelMethod(0.0)
    end
    @testset "SubgradientDescent" begin
        @testset "CorrectSign" begin
            pi = [1.; -1.; 1.; -1.; 0.]
            Lagrangian.correctsign!(pi, [:le; :ge; :ge; :le; :anything])
            @test isapprox(norm(pi - [1.; -1.; -1.; 1.; 0.]), 0., atol=1e-9)
        end
    end
    @testset "Tolerance" begin
        tol = Absolute(0.1)
        @test isclose(100.0, 101.0, tol) == false
        @test isclose(0.01, 0.012, tol)  == true
        tol = Relative(0.1)
        @test isclose(100.0, 101.0, tol) == true
        @test isclose(-100.0, -109.9, tol) == true
        @test isclose(0.01, 0.012, tol)  == false
        @test isclose(1e-9, 1.1e-9, tol) == true
        tol = Unit(0.1)
        @test isclose(100.0, 101.0, tol) == true
        @test isclose(0.01, 0.012, tol)  == true
        @test isclose(0.01, 1.012, tol)  == false
    end
end

@testset "Lagrangian" begin
    @testset "Relax and Recover" begin
        m=Model(solver=GLPKSolverMIP())
        @variable(m, -1 <= x[1:2] <=2)
        @constraints(m, begin
            c1, x[1] <= 0.
            c2, x[1] >= 0.
            c3, x[2] == 0.
        end)
        @objective(m, :Max, sum(x))
        solve(m)

        subgradient = LevelMethod(-5., quadsolver=IpoptSolver(print_level=0))
        relaxedbounds = [10.; -10.; 10.]
        l  = LinearProgramData(m.obj,
            [c1; c2; c3],
            relaxedbounds,
            method=subgradient)
        @testset "Relax" begin
            Lagrangian.relaxandcache!(l, m)
            @test l.old_bound ≈ zeros(3)
            @test m.linconstr[c1.idx].lb == -Inf
            @test m.linconstr[c1.idx].ub ≈ 10.
            @test m.linconstr[c2.idx].lb ≈ -10.
            @test m.linconstr[c2.idx].ub == Inf
            @test m.linconstr[c3.idx].lb == -Inf
            @test m.linconstr[c3.idx].ub ≈ 10.
        end
        @testset "Recover" begin
            Lagrangian.recover!(l, m, -ones(3))
            @test m.linconstr[c1.idx].lb == -Inf
            @test m.linconstr[c1.idx].ub ≈ 0.
            @test m.linconstr[c2.idx].lb ≈ 0.
            @test m.linconstr[c2.idx].ub == Inf
            @test m.linconstr[c3.idx].lb == 0.
            @test m.linconstr[c3.idx].ub ≈ 0.
            @test getdual.([c1; c2; c3]) ≈ ones(3)
        end
    end
end

const examples_dir = joinpath(dirname(dirname(@__FILE__)), "examples")

@testset "Lagrangian Examples" begin
    for example in [
            "small_MIP.jl",
            "rcsp.jl"
        ]
        @testset "$example" begin
            include(joinpath(examples_dir, "Lagrangian", example))
        end
    end
end
