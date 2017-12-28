# This file includes modified source code from https://github.com/odow/SDDP.jl
# as at 5b3ba3910f6347765708dd6e7058e2fcf7d13ae5

#  Copyright 2017, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

using SDDiP, JuMP, GLPKMathProgInterface, Base.Test, SDDP
# using Ipopt  # uncomment for the Level Method

srand(11111)

# Demand for newspapers
# There are two equally probable scenarios in each stage
#   Demand[stage, noise]
Demand = [
    10.0 15.0;
    12.0 20.0;
    8.0  20.0
]

# Markov state purchase prices
PurchasePrice = [5.0, 8.0]

RetailPrice = 7.0

# Transition matrix
Transition = Array{Float64, 2}[
    [1.0]',
    [0.6 0.4],
    [0.3 0.7; 0.3 0.7]
  ]

# Initialise SDDP Model
m = SDDPModel(
        sense             = :Max,
        stages            = 3,
        objective_bound   = 1000,
        markov_transition = Transition,
        solver            = GLPKSolverMIP()
                                                ) do sp, stage, markov_state

    # ====================
    #   State variable: binary decomposition of stock
    @binarystate(sp, stock <= 100, stock0 == 5, Int)

    # ====================
    #   Other variables
    @variables(sp, begin
        buy >= 0, Int  # Quantity to buy
        sell>= 0, Int  # Quantity to sell
    end)


    # ====================
    #   Scenarios
    @rhsnoises(sp, D=Demand[stage,:], begin
        sell <= D
        sell >= 0.5D
    end)

    # ====================
    #   Objective
    @stageobjective(sp, sell * RetailPrice - buy * PurchasePrice[markov_state])

    # ====================
    #   Dynamics constraint
    @constraint(sp, stock == stock0 + buy - sell)

    # Call to solve via Lagrangians
    # setSDDiPsolver!(sp, method=LevelMethod(-200.0, quadsolver=IpoptSolver(print_level=0)))
    # setSDDiPsolver!(sp, method=SubgradientMethod(-200.0, wait=10))
    setSDDiPsolver!(sp, method=KelleyMethod(-200.0))

end

@time solvestatus = SDDP.solve(m,
    max_iterations = 70
)

@test isapprox(getbound(m), 97.9, atol=1e-3)
