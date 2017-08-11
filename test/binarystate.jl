# This file includes modified source code from https://github.com/odow/SDDP.jl
# as at 5b3ba3910f6347765708dd6e7058e2fcf7d13ae5

#  Copyright 2017, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

rhs   = [1, 2, 3]
duals = [0.5, 1.5, 2.5]
newrhs = [1.1, 1.2, 1.3]
bounds = [1.0, 10]

@testset "Single element" begin
    @testset "Binary variable" begin
        # Binary states should do exactly the same things as states in SDDP.jl
        m = SDDP.Subproblem()
        @binarystate(m, x, x0 == rhs[1], Bin)
        @test SDDP.nstates(m) == 1
        @binarystate(m, y >= 0, y0 == rhs[2])
        @test SDDP.nstates(m) == 2
        @binarystate(m, 1 <= z <= 10, z0 == rhs[3])
        @test SDDP.nstates(m) == 3
        v = [x, y, z]
        v0 = [x0, y0, z0]
        m.linconstrDuals = duals
        for i in 1:3
            s = SDDP.states(m)[i]
            @test s.variable == v[i]
            c = m.linconstr[s.constraint.idx]
            @test c.lb == c.ub == rhs[i]
            SDDP.setvalue!(s, newrhs[i])
            @test c.lb == c.ub == newrhs[i]
            @test c.terms == JuMP.AffExpr(v0[i])
            @test JuMP.getdual(s) == duals[i]
        end
    end

    @testset "Integer variable" begin
        m = SDDP.Subproblem()

        @binarystate(m, y <= 10, y0 == 10, Int)
        @test isa(y, JuMP.Variable)
        @test isa(y0, JuMP.Variable)
        @test SDDP.nstates(m) == 4
    #     # when moved from set sddip solver, test ub/lb to be 0/1
        @test isapprox(m.colUpper[y.col], 10.0)
        @test m.colLower[y.col] == -Inf

        @binarystate(m, 1 <= z <= 10, z0 == 3, Int)
        @test SDDP.nstates(m) == 4 + 4
        @test isapprox(m.colUpper[z.col], 10.0)
        @test isapprox(m.colLower[z.col], 1.0)

        @binarystate(m, bounds[1] <= w <= bounds[2], w0 == rhs[3], Int)
        @test SDDP.nstates(m) == 4 + 4 + 4
        @test isapprox(m.colUpper[w.col], 10.0)
        @test isapprox(m.colLower[w.col], 1.0)

        @binarystate(m, bounds[1] <= x <= bounds[2] + 4, x0 == rhs[3] - rhs[1], Int)
        @test SDDP.nstates(m) == 4 + 4 + 4 + 4
        @test isapprox(m.colUpper[x.col], 14.0)
        @test isapprox(m.colLower[x.col], 1.0)
    end

    @testset "Continuous variable" begin
        m = SDDP.Subproblem()
        @binarystate(m, x <= 10, x0 == 10, Cont)
        @test length(SDDP.states(m)) == 7
        @binarystate(m, y <= 10, y0 == 10, Cont, 0.01)
        @test SDDP.nstates(m) == 7 + 10
    end
end

@testset "Vector elements" begin
    @testset "Binary variable" begin
        m = SDDP.Subproblem()
        @binarystate(m, bounds[1] <= x[i=1:3] <= bounds[2], x0 == rhs[i], Bin)
        @test SDDP.nstates(m) == 3

        # Binary states should do exactly the same things as states in SDDP.jl
        v = [x[1], x[2], x[3]]
        v0 = [x0[1], x0[2], x0[3]]
        for i in 1:3
            s = SDDP.states(m)[i]
            @test s.variable == v[i]
            c = m.linconstr[s.constraint.idx]
            @test c.lb == c.ub == rhs[i]
            @test c.terms == JuMP.AffExpr(v0[i])
        end
    end

    @testset "Integer variable" begin
        m = SDDP.Subproblem()
        @binarystate(m, x[i=1:3] <= 10, x0 == rhs[i], Int)
        @test isa(x, Vector{JuMP.Variable})
        @test isa(x0, Vector{JuMP.Variable})
        @test SDDP.nstates(m) == 4 * 3
        m.solver = GLPKSolverLP()
        JuMP.solve(m)
        @test isapprox(JuMP.getvalue(x0), rhs)

        letters = [:a, :b]
        idxs = Dict(zip(letters, [1; 2]))
        @binarystate(m, y[i=1:3, j=letters] <= 10, y0 == ones(3, 2)[i, idxs[j]], Int)
        @test isa(y, JuMP.JuMPArray)
        @test isa(y0, JuMP.JuMPArray)
        @test SDDP.nstates(m) == 4 * 3 + 4 * 6
        JuMP.solve(m)
        @test isapprox(JuMP.getvalue(y0).innerArray, ones(3, 2))

    end

    @testset "Continuous variable" begin
        m = SDDP.Subproblem()
        @binarystate(m, 1 <= x[i=1:3] <= 10, x0 == rhs[i], Cont)
        @test SDDP.nstates(m) == 7 * 3
        @binarystate(m, y[i=1:3] <= 10, y0 == rhs[i], Cont, 0.01)
        @test length(SDDP.states(m)) == 7 * 3 + 10 * 3
        m.solver = GLPKSolverLP()
        JuMP.solve(m)
        @test isapprox(JuMP.getvalue(y0), rhs)
    end

    @testset "@binarystates" begin

        m = SDDP.Subproblem()
        @binarystates(m, begin
            x[i=1:3] <= bounds[2], x0 == rhs[i], Bin
            y <= 1, y0==0.5, Cont, 0.01
            z <= 5, z0==5, Int
        end)

        # 3 state variables for x, nstates(1*0.01)=7 for y, and nstates(5)=3 for z
        @test SDDP.nstates(m) == 3 + 7 + 3

        v = [x[1], x[2], x[3]]
        v0 = [x0[1], x0[2], x0[3]]
        for i in 1:3
            s = SDDP.states(m)[i]
            @test s.variable == v[i]
            c = m.linconstr[s.constraint.idx]
            @test c.lb == c.ub == rhs[i]
            @test c.terms == JuMP.AffExpr(v0[i])
        end
        for v in [y, y0, z, z0]
            @test isa(v, JuMP.Variable)
        end
        m.solver = GLPKSolverLP()
        JuMP.solve(m)
        @test isapprox(JuMP.getvalue(y0), 0.5)
        @test isapprox(JuMP.getvalue(z0), 5)
    end
end
