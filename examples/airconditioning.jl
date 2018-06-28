# This file includes modified code from https://github.com/odow/SDDP.jl as at
# commit a6662eecff6d5de5c499b760773a8569a0979b9b.

#  Copyright 2017, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

#=
    The air conditioning example from Anthony Papavasiliou
    https://perso.uclouvain.be/anthony.papavasiliou/public_html/SDDP.pdf

    Consider the following problem
        Produce air conditioners for 3 months
        200 units/month at 100 $/unit
        Overtime costs 300 $/unit
        Known demand of 100 units for period 1
        Equally likely demand, 100 or 300 units, for periods 2, 3
        Storage cost is 50 $/unit
        All demand must be met

    Optimal bound $62,500
=#
using SDDP, JuMP, GLPKMathProgInterface, Base.Test, SDDiP

function airconditioningmodel()
    m = SDDPModel(
                 stages = 3,
                 # example for issue #64
        objective_bound = (t, i) -> 0.0,
                  sense = :Min,
                 # solver = GLPKSolverMIP()
                 solver = GLPKSolverMIP()
                            ) do sp, stage
        # number of units
        @binarystate(sp, 0 <= stored_production <= 1023, incoming_storage == 0, Int)
        # @state(sp, stored_production >= 0, incoming_storage == 0)
        @variables(sp, begin
            # number of units produced during normal production
            # circumstances
            0 <= production <= 200
            # number of units produced during overtime production
            # circumstances
            overtime   >= 0
        end)
        # balance constraints
        if stage == 1
            @constraint(sp, stored_production == incoming_storage + production + overtime - 100)
        else
            @rhsnoise(sp, demand=[100, 300], stored_production == incoming_storage + production + overtime - demand)
        end
        @stageobjective(sp, 100 * production + 300 * overtime + 50 * stored_production)
        # Set integer solver
        setSDDiPsolver!(sp, method=KelleyMethod(), LPsolver=GLPKSolverLP(), pattern=Pattern(lagrangian=5, benders=1))
    end
end

srand(1234)
m = airconditioningmodel()
solve(m, max_iterations=10)
@test isapprox(getbound(m), 62_500.0)
