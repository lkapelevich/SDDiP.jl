
@testset "initialize_dual!" begin
    include(joinpath(dirname(dirname(@__FILE__())), "examples/xor.jl"))
    m = build_xor(zeros(2))
    @assert solve(m, max_iterations = 1) .== :iteration_limit
    sp = m.stages[2].subproblems[1]
    π0 = zeros(2)
    SDDiP.initialize_dual!(sp, π0)
    @test all(π0 .≈ 1.0)
    m = build_xor([1.0, 0.0])
    @assert solve(m, max_iterations = 1) .== :iteration_limit
    sp = m.stages[2].subproblems[1]
    π0 .= 0.0
    SDDiP.initialize_dual!(sp, π0)
    @test all(π0 .≈ 1.0)
end
