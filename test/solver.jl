
@testset "initialize_dual!" begin
    include(joinpath(dirname(dirname(@__FILE__())), "examples/xor.jl"))
    # =
    m = build_xor(zeros(2))
    sp = m.stages[2].subproblems[1]
    @assert solve(sp) == :Optimal
    π0 = zeros(2)
    SDDiP.initialize_dual!(sp, π0, false)
    @test all(-π0 .≈ 1.0)
    # =
    π0 = zeros(2)
    SDDiP.initialize_dual!(sp, π0, true)
    @test all(-π0 .≈ [1.5, 1.5])
    # =
    m = build_xor([1.0, 0.0])
    # @assert solve(m, iteration_limit = 1) .== :iteration_limit
    sp = m.stages[2].subproblems[1]
    @assert solve(sp) == :Optimal
    π0 .= 0.0
    SDDiP.initialize_dual!(sp, π0, false)
    # We are at (1, 0), comparint with (0, 1). Objective difference is 0.
    # First pi = 1 so needs perturbing down, second pi needs perturbing up.
    @test all(-π0 .≈ [-1.0, 1.0])
    # =
    π0 .= 0.0
    # We are at (1, 0), movint to (0, 0) and then (1, 1).
    # Objective differences are -1 and -1. Subgradients are 1 and -1.
    SDDiP.initialize_dual!(sp, π0, true)
    @test all(-π0 .≈ [0.0, 0.0])
end
