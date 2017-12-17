#
# Power generation expansion problem
#
using SDDP, GLPKMathProgInterface, JuMP, SDDiP, Ipopt

immutable GenTech
    capacity::Float64   # maximum MW this generator technology can be built for
    b_cost::Float64     # cost to build
    g_cost::Float64     # cost to use
    nunits::Int         # maximum number of units we are happy to build
    init::Vector{Int}   # initial state
end
immutable GEPData
    T::Int                                  # planning duration
    demand::Array{Float64,2}                # power demand in each stage
    penalty::Float64                        # penalty per MW not met
    hours::Float64                          # hours in each stage
    rho::Float64                            # discounting rate
    gentypes::Dict{Symbol, GenTech}         # all technologies available
end

function makedata()
    T = 5
    S = 8
    inv_c = 1e4 * ones(T)
    gen_c = 4 * ones(T)
    # Some random demand
    d = [ 5   5  5   5   5   5   5   5;
          4   3  1   3   0   9   8  17;
          0   9  4   2  19  19  13   7;
         25  11  4  14   4   6  15  12;
          6   7  5   3   8   4  17  13]
    p = 5e5
    hours = 1.0 # scaled for nice numbers...
    gentypes = Dict(:TechA => GenTech(1.0, 1e4, 4.0, 20.0, zeros(20)))
    GEPData(T, d, p, hours, 0.99, gentypes)
end

data = makedata()
gentypes = collect(keys(data.gentypes))

getgen(s::Symbol) = data.gentypes[s]
maxgen(s::Symbol) = getgen(s).capacity
nunits(s::Symbol) = getgen(s).nunits
build(s::Symbol)  = getgen(s).b_cost
use(s::Symbol)    = getgen(s).g_cost
init(s::Symbol)   = getgen(s).init

m=SDDPModel(stages=data.T, objective_bound=0.0, sense=:Min, solver=GLPKSolverMIP()) do sp, stage
    @binarystate(sp, invested[i = gentypes, j = 1:nunits(i)], invested0 == init(i)[j], Bin)
    @variables(sp, begin
        generation[gentypes] >= 0
        penalty >= 0
        demand
    end)

    @constraints(sp, begin
        # Can't un-invest
        investment[i = gentypes, j = 1:nunits(i)], invested[i, j] >= invested0[i, j]
        # Generation capacity
        [i = gentypes], sum(invested[i, j] * maxgen(i) for j = 1:nunits(i)) >= generation[i]
        # Meet demand or pay a penalty
        penalty >= demand - sum(generation)
        # Order the units to break symmetry, units are identical
        [i = gentypes, j = 1:nunits(i)-1], invested[i, j] <= invested[i, j+1]
    end)

    # Demand is uncertain
    @rhsnoise(sp, D=data.demand[stage,:], demand == D)

    # Handy calculation of the investment cost in this stage
    @expression(sp, investment_cost[i = gentypes], build(i) * sum(invested[i, j] - invested0[i, j] for j = 1:nunits(i)))

    @stageobjective(sp,
        sum(investment_cost[i] for i = gentypes) * data.rho ^ (stage - 1) +
        sum(use(i) * generation[i] for i = gentypes) * data.hours * data.rho ^ (stage - 1) +
        penalty * data.penalty * data.hours)

    # Solve with the level method as the Lagrangian solver
    setSDDiPsolver!(sp, method=LevelMethod(5e6, quadsolver=IpoptSolver(print_level=0)),
                        pattern = Pattern(benders=1, lagrangian=5, strengthened_benders=1),
                        LPsolver = GLPKSolverLP()
                        )

end

srand(11111)
status = solve(m, max_iterations=60)
@assert isapprox(getbound(m), 460533, atol=1e3)
