using JuMP, Gurobi, SDDiP, SDDP, Base.Test, GLPKMathProgInterface

function build_xor(x0::Vector{Float64})
    m = SDDPModel(stages=2, objective_bound=-5.0, sense=:Min, solver=GLPKSolverMIP()) do sp, stage
        @binarystates(sp, begin
            x1, x1_0 == x0[1], Bin
            x2, x2_0 == x0[2], Bin
        end)
        if stage == 1
            @stageobjective(sp, 0.0)
        else
            @variable(sp, y)
            @stageobjective(sp, y)
            @constraints(sp, begin
                y >= x1_0 - x2_0
                y >= x2_0 - x1_0
                y <= x1_0 + x2_0
                y <= 2 - x1_0 - x2_0
            end)
        end
        # Set Lagrangian solver
        setSDDiPsolver!(sp, method=KelleyMethod())
    end
end
m = build_xor(zeros(2))
solvestatus = SDDP.solve(m,
    iteration_limit = 2
)
@test isapprox(getbound(m), 0.0, atol=1e-4)
