using SDDiP, GLPKMathProgInterface, JuMP, SDDP, Base.Test
using Ipopt

srand(11111)

struct AllBlacksData
  T::Int                  # Number of time periods
  N::Int                  # Number of seats
  R::Array{Float64,2}     # R_ij = evenue from selling seat i at time j
  offer::Array{Float64,2} # offer_ij = whether an offer for seat i will come at time j
end

data = AllBlacksData(3, 2, [3 3 6; 3 3 6], [1 1 0; 1 0 1])

function build_model(lagrangian_method::Lagrangian.AbstractLagrangianMethod)
    m=SDDPModel(stages=data.T, objective_bound=100.0, sense=:Max, solver=GLPKSolverMIP()) do sp, stage

        # Seat remaining?
        @binarystate(sp, x[i=1:data.N], x0==1, Bin)

        # Action: accept offer, or don't accept offer
        @variable(sp, accept_offer, Bin)

        @constraints(sp, begin
            # Balance on seats
            dynamics[i=1:data.N], x[i] == x0[i] - data.offer[i, stage] * accept_offer
        end)

        @stageobjective(sp, sum(data.R[i, stage] * data.offer[i, stage] * accept_offer for i=1:data.N))

        # Set Lagrangian solver
        setSDDiPsolver!(sp, method=lagrangian_method)

    end
end

for lagrangian_method in [KelleyMethod(),
                BinaryMethod(),
                LevelMethod(quadsolver=IpoptSolver(print_level=0)),
                SubgradientMethod()
            ]
    m = build_model(lagrangian_method)
    solvestatus = SDDP.solve(m,
        max_iterations = 8
    )
    @test isapprox(getbound(m), 9.0)
end
