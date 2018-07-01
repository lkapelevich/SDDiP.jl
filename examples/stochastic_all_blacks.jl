#
# Selling tickets to a sports match
#
using SDDP, GLPKMathProgInterface, JuMP, SDDiP, Base.Test
using Ipopt

struct AllBlacksData2
    T::Int                        # Number of time periods
    N::Int                        # Number of seats
    R::Array{Float64,2}           # R_ij = price of seat i at time j
    s::Int                        # Number of scenarios
    offer::Array{Vector{Int},2}   # offer_ij = a vector of scenarios length s for whether an offer for seat i will come at time j
end

function makedata(T, N, R, s)
    offers = [rand([0, 1], s) for n in 1:N, t in 1:T]
    AllBlacksData2(T, N, R, s, offers)
end

srand(11111)
data = makedata(3, 2, [3 3 6; 3 3 6], 3)

function build_stoch_all_blacks(lagrangian_method)
    m=SDDPModel(stages=data.T, objective_bound=100.0, sense=:Max, solver=GLPKSolverMIP()) do sp, stage

        # Seat remaining?
        @binarystate(sp, x[i=1:data.N], x0==1, Bin)

        # Action: accept offer, or don't accept offer
        # We are allowed to accpect some of the seats offered but not others in this formulation
        @variable(sp, 0 <= accept_offer[i=1:data.N] <= 1, Int)

        # Balance on seats
        @constraint(sp, balance[i=1:data.N], x0[i] - x[i] == accept_offer[i])

        for i in 1:data.N
          # Can't sell a seat if there is no offer for it
          @rhsnoise(sp, ω = data.offer[i, stage], accept_offer[i] <= ω)
        end

        @stageobjective(sp, sum(data.R[i, stage] * accept_offer[i] for i=1:data.N))

        # Call for using a Lagrangian solver
        setSDDiPsolver!(sp, method=lagrangian_method)
    end
end

for lagrangian_method in [KelleyMethod(),
                # BinaryMethod(),
                LevelMethod(quadsolver=IpoptSolver(print_level=0)),
                SubgradientMethod()
            ]
    srand(11111)
    m = build_stoch_all_blacks(lagrangian_method)
    solvestatus = SDDP.solve(m,
        max_iterations = 10
    )
    @test isapprox(getbound(m), 8.0, atol=1e-3)
end
