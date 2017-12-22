using SDDiP, GLPKMathProgInterface, JuMP, SDDP, Base.Test
# using Ipopt  # uncomment for the Level Method

immutable AllBlacksData
  T::Int                  # Number of time periods
  N::Int                  # Number of seats
  R::Array{Float64,2}     # R_ij = evenue from selling seat i at time j
  offer::Array{Float64,2} # offer_ij = whether an offer for seat i will come at time j
end

data = AllBlacksData(3, 2, [3 3 6; 3 3 6], [1 1 0; 1 0 1])

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

    # Level method example:
    # setSDDiPsolver!(sp, method=LevelMethod(-100.0, quadsolver=IpoptSolver(print_level=0)))
    # Subgradient descent example:
    setSDDiPsolver!(sp, method=SubgradientMethod(-100.0))
    # setSDDiPsolver!(sp, method=KelleyMethod(-100.0))

end

srand(11111)
solution = solve(m, max_iterations=8)
@test isapprox(getbound(m), 9.0)
