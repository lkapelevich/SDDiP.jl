# Relaxation of the TSP, just a toy

module TSP

using JuMP, Gurobi, SDDiP, Combinatorics, GLPKMathProgInterface, Ipopt, PyPlot

function getdistances(x::Vector{Float64}, y::Vector{Float64})
    n = min(length(x), length(y))
    mat = zeros(n, n)
    @inbounds for j=1:n, i=1:j-1
        mat[i, j] = norm([x[i]; y[i]] - [x[j]; y[j]])
    end
    mat
end

srand(32)

const ncities = 12
const x_coords = rand(ncities)
const y_coords = rand(ncities)
const distance_matrix = getdistances(x_coords, y_coords)
const blocked_roads = Tuple{Int, Int}[]

function plottour(tour::Vector{Tuple{Int,Int}}, s::AbstractString)
    lines = Vector[]
    for t in tour
        xs = x_coords[[t...]]
        ys = y_coords[[t...]]
        push!(lines, collect(zip(xs, ys)))
    end

    line_segments = matplotlib[:collections][:LineCollection](lines)

    fig = figure(s)
    ax = axes()
    ax[:add_collection](line_segments)
    axis("image")
end

function nCx(n::Int, x::Int)
    n < x && return 0
    ret = 1
    for i=1:x
        ret *= (n - i + 1)
    end
    ret = div(ret, factorial(x))
end
function strict_subsets{T}(x::Vector{T})
    n = length(x)
    n <= 1 && error("Vector length $n has no strict subsets.")
    y = Array{Array}(n-2)
    @inbounds for len = 2:n-1
        y[len-1] = collect(combinations(x, len))
    end
    y
end

function naivemodel()
    m = Model(solver=GLPKSolverMIP())
    @variable(m, edges[i=1:ncities, j=i+1:ncities], Bin)

    @constraints(m, begin
        balance[i=1:ncities], sum(edges[i, k] for k=i+1:ncities) + sum(edges[k, i] for k=1:i-1) == 2
        blockedroads[r=blocked_roads], edges[r] <= 0
    end)

    @objective(m, :Min, sum(edges[i,j] * distance_matrix[i, j] for i=1:ncities, j=i+1:ncities))

    # Not being realistic here...
    subsets = strict_subsets(collect(1:ncities))
    for len = 1:ncities-2
        for subset in subsets[len]
            @constraint(m, sum(edges[pair[1], pair[2]] for pair in combinations(subset, 2)) <= len)
        end
    end
    m
end

function very_naive_solve(;relax=false)
    m = naivemodel()

    if relax
        π0 = rand(length(m[:balance]))
        dualbound = 100.
        relaxed_bound = ncities * ones(length(m[:balance]))

        method  = LevelMethod(initialbound = dualbound, quadsolver=GurobiSolver(OutputFlag=0))
        # method  = SubgradientMethod(dualbound)
        TSPdata = LinearProgramData(m.obj,
                                    m[:balance],
                                    relaxed_bound,
                                    method=method)

        status, bestvalue = lagrangiansolve!(TSPdata, m, π0)
    else
        solve(m)
    end

    println("The cost is ", getobjectivevalue(m))
    tour = Tuple{Int, Int}[]
    for i=1:ncities
        for j = i+1:ncities
            if getvalue(m[:edges][i, j]) ≈ 1.0
                println((i, j))
                push!(tour, (i, j))
            end
        end
    end
    plottour(tour, "Naive tour")

end

function kruskals(costs::Vector{Float64}, edges::Vector{Tuple{Int,Int}}, nnodes::Int)
    nedges = length(edges)
    @assert nedges == length(costs)

    # The order in which edges will be checked
    order = sortperm(costs)
    ret = zeros(Int, nedges)

    # How many edges we added so far
    count = 1

    # Build up an incidence matrix as we go
    incmat = zeros(Int, nnodes, nedges)

    # Add arcs one at a time
    for k in order
        # Incidence matrix so far
        temp = view(incmat, :, 1:count)
        # Try add kth cheapest edge
        temp[edges[k][1], count] = 1
        temp[edges[k][2], count] = -1
        # Check for cycles
        V = nullspace(temp)
        if isempty(V)
            # Lock in the edge
            count += 1
            ret[k] = 1
            # Stop if we have added enough edges
            count == nnodes && return ret
        else
            # Edge didn't work out
            temp[edges[k][1], count] = 0
            temp[edges[k][2], count] = 0
        end
    end
    error("Something went wrong in Kruskal's. Perhaps you specified the wrong number of nodes.")
end

immutable SpanningTree <: Lagrangian.AbstractProblemClass end

# This could just be a solvehook with the current forumlation.
# But if we get away from having to formulate the primal as a maths programming model altogether, even better...
function Lagrangian.solve_primal{M<:AbstractLagrangianMethod, C<:SpanningTree}(m::JuMP.Model, d::LinearProgramData{M, C}, π::Vector{Float64})
    # we arbitrarily chose to leave out the last city, we could leave out
    # any city or even better, try leave out one city at a time and find the worst case
    arcs = [(i, j) for i=1:ncities-2 for j=i+1:ncities-1]

    # Prepare an array of costs
    costs = [distance_matrix[a...] for a in arcs][:]

    # Costs include Lagrangian penalties
    for (i, a) in enumerate(arcs)
        costs[i] += π[a[1]] + π[a[2]]
    end

    # Fix n-1 arcs to Kruskal's algorithm's output
    fixed_vals = kruskals(costs, arcs, ncities-1)
    for (i, a) in enumerate(arcs)
        JuMP.setRHS(m[:fixvals][a], fixed_vals[i])
    end

    # Note: we don't just find the final two arcs by picking the cheapest ones
    # because the final two arcs affect the degree of the first n-1 nodes

    @objective(m, :Min, d.obj + dot(π, d.slacks))
    @assert solve(m, ignore_solve_hook=true) == :Optimal

    getobjectivevalue(m), getvalue(d.slacks)

end

# ============================================================================

function less_naive_relaxed()

    # Represent the TSP as a MIP, although this could just be some network logic

    m = Model(solver=GLPKSolverMIP())
    @variable(m, edges[i=1:ncities-1, j=i+1:ncities], Bin)

    @constraints(m, begin
        # Each node has a degree of two- this will be relaxed
        balance[i=1:ncities-1], sum(edges[i, k] for k=i+1:ncities) + sum(edges[k, i] for k=1:i-1) == 2
        # We do require exactly 2 arcs at the final node
        ensure1tree, sum(edges[k, ncities] for k=1:ncities-1) == 2
        # Just for fun
        blockedroads[r=blocked_roads], edges[r] <= 0
    end)

    # Min cost tour
    @objective(m, :Min, sum(edges[i,j] * distance_matrix[i, j] for i=1:ncities-1, j=i+1:ncities))

    # =========================================================================
    # Prepare a dummy constraint, we will choose the RHS while solving

    # Arcs in our one tree (a tree with all nodes except the final node)
    arcs = [(i, j) for i=1:ncities-2 for j=i+1:ncities-1]
    # Edges = whatever value Kruskal's algorithm says they should be
    @constraint(m,fixvals[a in arcs], edges[a...] == 0)

    # =========================================================================
    # The actual Lagrangian solving:

    π0 = rand(length(m[:balance]))                          # random duals to start with
    dualbound = 200.                                        # upper bound for the dual problem
    relaxed_bound = ncities * ones(length(m[:balance]))     # upper bound for the constraint wer are relaxing (something > 2, like ncities)

    # Method:
    method  = LevelMethod(initialbound = dualbound, quadsolver=GurobiSolver(OutputFlag=0))
    # method  = SubgradientMethod(dualbound, wait=20)

    # Linear program data
    TSPdata = LinearProgramData(m.obj,                  # objective
                                m[:balance],            # constraints being relaxed
                                relaxed_bound,          # a large number for RHS of relaxed constraints
                                method=method,
                                problem_class=SpanningTree()
                                )

    # Solve
    status, bestvalue = lagrangiansolve!(TSPdata, m, π0)

    println("The cost is ", getobjectivevalue(m))
    println("The duals are ", getdual(m[:balance]))
    tour = Tuple{Int, Int}[]
    for i=1:ncities
        for j = i+1:ncities
            if getvalue(m[:edges][i, j]) ≈ 1.0
                println((i, j))
                push!(tour, (i, j))
            end
        end
    end
    plottour(tour, "Relaxation tour")
end

end

@time TSP.very_naive_solve(relax=false)
# @time TSP.very_naive_solve(relax=true)
@time TSP.less_naive_relaxed()

using Base.Test
@test TSP.kruskals([1.; 2.; 3.], [(1,2), (2, 3), (1, 3)], 3) == [1, 1, 0]
@test TSP.kruskals([1.; 2.; 3.], [(1,2), (2, 3), (1, 3)], 3) == [1, 1, 0]
@test TSP.kruskals([1.; 0; 2.; 3.], [(1,2), (2, 1), (2, 3), (1, 3)], 3) == [0, 1, 1, 0]
@test TSP.kruskals([5.; 3.; 4.; 1.; 2.], [(1,2), (2, 3), (1, 3), (3, 4), (2, 4)], 4) == [0, 0, 1, 1, 1]
