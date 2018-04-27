# This file includes modified code from https://github.com/odow/SDDP.jl as at
# commit a6662eecff6d5de5c499b760773a8569a0979b9b.

#  Copyright 2017,  Oscar Dowson, Eyob Zewdie
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

#=
This problem is a version of the Ambulance dispatch problem. A hospital is
located at 0 on the number line that stretches from 0 to 100. Ambulance bases
are located at points 20, 40, 60, 80, and 100. When not responding to a call,
Ambulances must be located at a base, or the hospital. In this example there are
three ambulances.

Example location:

    H       B       B       B       B       B
    0 ---- 20 ---- 40 ---- 60 ---- 80 ---- 100

Each stage, a call comes in from somewhere on the number line. The agent must
decide which ambulance to dispatch. They pay the cost of twice the driving
distance. If an ambulance is not dispatched in a stage, the ambulance can be
relocated to a different base in preparation for future calls. This incurrs a
cost of the driving distance.
=#
using SDDP, JuMP, GLPKMathProgInterface, Base.Test, SDDiP

function vehiclelocationmodel(nvehicles, baselocations, requestlocations)
    Locations = baselocations       # Points on the number line where emergency bases are located
    Vehicles = 1:nvehicles          # ambulances
    nbases = length(baselocations)
    Bases = 1:nbases                # base 1 = hostpital
    Requests = collect(requestlocations)     # Points on the number line where calls come from

    shiftcost(src, dest) = abs(Locations[src] - Locations[dest])
    dispatchcost(request, base) = 2 * (abs(request - Locations[1]) + abs(request-Locations[base]))

    #Initial State of emergency vehicles at bases
    Q0 = zeros(Int64, (length(Bases), length(Vehicles)))
    Q0[1,:] = 1 # all ambulances start at hospital

    m = SDDPModel(
                 stages = 10,
        objective_bound = 0.0,
                  sense = :Min,
                 solver = GLPKSolverMIP()
                            ) do sp, t

        # Vehicles at which bases?
        @binarystate(sp, 0 <= q[b=Bases, v=Vehicles] <= 1, q0 == Q0[b, v])

        @variables(sp, begin
            # which vehicle is dipatched?
            0 <= dispatch[src=Bases, v=Vehicles] <= 1
            # shifting vehicles between bases
            0 <= shift[src=Bases, v=Vehicles, dest=Bases] <= 1
        end)

        @expression(sp, basebalance[b in Bases, v in Vehicles],
            # flow of vehicles in and out of bases:
            #initial - dispatch      - shifted from      + shifted to
            q0[b, v] - dispatch[b,v] - sum(shift[b,v,:]) + sum(shift[:,v,b])
        )
        @constraints(sp, begin
            # only one vehicle dispatched to call
            sum(dispatch) == 1
            # can only dispatch vehicle from base if vehicle is at that base
            [b in Bases, v in Vehicles], dispatch[b,v] <= q0[b,v]
            # can only shift vehicle if vehicle is at that src base
            # use formulation that gives stronger bounds
            [b in Bases, v in Vehicles, c in Bases], shift[b,v,c] <= q0[b, v]
            # can only shift vehicle if vehicle is not being dispatched
            [b in Bases, v in Vehicles, c in Bases], shift[b,v,c] + dispatch[b,v] <= 1
            # can't shift to same base
            [b in Bases, v in Vehicles], shift[b,v,b] == 0
            # Update states for non-home/non-hospital bases
            [b in Bases[2:end], v in Vehicles], q[b, v] == basebalance[b,v]
            # Update states for home/hospital bases
            [v in Vehicles], q[1, v] == basebalance[1,v] + sum(dispatch[:,v])
        end)

        if t == 1
            @stageobjective(sp, sum(
                    #distance to travel from base to emergency and back to home base
                    dispatch[b,v] * dispatchcost(50, b) +
                    #distance travelled by vehilces relocating bases
                    sum(shiftcost(b, dest) * shift[b, v, dest] for dest in Bases)
                for b in Bases, v in Vehicles)
            )
        else
            @stageobjective(sp, request=Requests, sum(
                    #distance to travel from base to emergency and back to home base
                    dispatch[b,v] * dispatchcost(request, b) +
                    #distance travelled by vehilces relocating bases
                    sum(shiftcost(b, dest) * shift[b, v, dest] for dest in Bases)
                for b in Bases, v in Vehicles)
        )
        end
        setSDDiPsolver!(sp,
            method=KelleyMethod(),
            LPsolver = GLPKSolverLP(),
            pattern=Pattern(benders=4, lagrangian=1, strengthened_benders=1)
            )
    end
end

srand(1234)
ambulancemodel = vehiclelocationmodel(3, [0, 20, 40, 60, 80, 100], 0:10:100)
@assert solve(ambulancemodel, max_iterations=20) == :max_iterations
@test getbound(ambulancemodel) >= 1206.0

# Symmetry breaking constraints
# @variables(sp, begin
#     # Let s[u, v, b] = 1 if vehicles u and v are both at base b
#     s[u=1:nvehicles-1, v=u+1:nvehicles, Bases], Bin
# end)
# @constraints(sp, begin
#     # Definition of s
#     [u=1:nvehicles-1, v=u+1:nvehicles, b in Bases], s[u, v, b] <= q0[b, u]
#     [u=1:nvehicles-1, v=u+1:nvehicles, b in Bases], s[u, v, b] <= q0[b, v]
#     [u=1:nvehicles-1, v=u+1:nvehicles, b in Bases], q0[b, u] + q0[b, v] <= s[u, v, b] + 1
#     # If multiple vehicles are at the same base, dispatch vehicle v before v+i
#     [b in Bases, u=1:nvehicles-1, v=u+1:nvehicles], dispatch[b, u] - dispatch[b, v] >= s[u,v, b] - 1
#     # From any base, shift u to b and v to c if b has a lower index than c, unless u is being dispatched
#     [b in Bases, u=1:nvehicles-1, v=u+1:nvehicles, c=1:nbases, d=c:nbases],
#         shift[b, u, c] + dispatch[b, u] - shift[b, v, d] >= s[u, v, b] - 1
# end)
