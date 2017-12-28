@testset "Cut Type" begin
    p = Pattern(strengthened_benders=1, lagrangian=0)
    @test SDDiP.getcuttype(1, p) == :strengthened_benders
    @test SDDiP.getcuttype(2, p) == :strengthened_benders

    p = Pattern(benders=1, lagrangian=0)
    @test SDDiP.getcuttype(1, p) == :benders
    @test SDDiP.getcuttype(2, p) == :benders

    p = Pattern(benders=1, strengthened_benders=0, lagrangian=1)
    @test SDDiP.getcuttype(1, p) == :lagrangian
    @test SDDiP.getcuttype(2, p) == :benders
    @test SDDiP.getcuttype(3, p) == :lagrangian
    @test SDDiP.getcuttype(4, p) == :benders

    p = Pattern(benders=2, strengthened_benders=1, lagrangian=0)
    @test SDDiP.getcuttype(1, p) == :benders
    @test SDDiP.getcuttype(2, p) == :strengthened_benders
    @test SDDiP.getcuttype(3, p) == :benders
    @test SDDiP.getcuttype(4, p) == :benders
    @test SDDiP.getcuttype(5, p) == :strengthened_benders

    p = Pattern(benders=1, strengthened_benders=1, lagrangian=1)
    @test SDDiP.getcuttype(1, p) == :strengthened_benders
    @test SDDiP.getcuttype(2, p) == :lagrangian
    @test SDDiP.getcuttype(3, p) == :benders
    @test SDDiP.getcuttype(4, p) == :strengthened_benders
end

@testset "Binary Expansion" begin
    @test_throws Exception binexpand(0)
    @test binexpand(1) == [1]
    @test binexpand(2) == [0, 1]
    @test binexpand(3) == [1, 1]
    @test binexpand(4) == [0, 0, 1]
    @test binexpand(5) == [1, 0, 1]
    @test binexpand(6) == [0, 1, 1]
    @test binexpand(7) == [1, 1, 1]

    @test binexpand(typemax(Int)) == ones(Int, SDDiP._2i_L)

    @test 0 == bincontract([0])
    @test 1 == bincontract([1])
    @test 0 == bincontract([0, 0])
    @test 1 == bincontract([1, 0])
    @test 2 == bincontract([0, 1])
    @test 3 == bincontract([1, 1])
    @test 2 == bincontract([0, 1, 0])
    @test 3 == bincontract([1, 1, 0])
    @test 4 == bincontract([0, 0, 1])
    @test 5 == bincontract([1, 0, 1])
    @test 6 == bincontract([0, 1, 1])
    @test 7 == bincontract([1, 1, 1])
    @test typemax(Int) == bincontract(ones(Int, SDDiP._2i_L))

    @test binexpand(1, length=3) == [1, 0, 0]
    @test binexpand(2, length=3) == [0, 1, 0]
    @test binexpand(4, length=3) == [0, 0, 1]
    @test_throws Exception binexpand(8, length=3)

    @test binexpand(1, maximum=7) == [1, 0, 0]
    @test binexpand(2, maximum=7) == [0, 1, 0]
    @test binexpand(4, maximum=7) == [0, 0, 1]
    @test binexpand(4, maximum=7, length=2) == [0, 0, 1]
    @test_throws Exception binexpand(8, maximum=7)

    @test_throws(Exception, bincontract(Float64, [1, 0], -0.1))
    @test isapprox(bincontract(Float64, [0],       0.1), 0.0, atol=1e-4)
    @test isapprox(bincontract(Float64, [1],       0.1), 0.1, atol=1e-4)
    @test isapprox(bincontract(Float64, [0, 1],    0.1), 0.2, atol=1e-4)
    @test isapprox(bincontract(Float64, [1, 1],    0.1), 0.3, atol=1e-4)
    @test isapprox(bincontract(Float64, [0, 1, 0], 0.1), 0.2, atol=1e-4)
    @test isapprox(bincontract(Float64, [1, 1, 0], 0.1), 0.3, atol=1e-4)
    @test isapprox(bincontract(Float64, [1, 0, 1], 0.1), 0.5, atol=1e-4)
    @test binexpand(0.5)        == binexpand(5)
    @test binexpand(0.54)       == binexpand(5)
    @test binexpand(0.56, 0.1)  == binexpand(6)
    @test binexpand(0.5, 0.01)  == binexpand(50)
    @test binexpand(0.54, 0.01) == binexpand(54)
    @test binexpand(0.56, 0.01) == binexpand(56)

    @test binexpand(0.5, length=5)        == binexpand(5, length=5)
    @test binexpand(0.56, 0.01, length=8) == binexpand(56, length=8)

    @test binexpand(0.5, maximum=3.1)         == binexpand(5, length=5)
    @test binexpand(0.56, 0.01, maximum=2.55) == binexpand(56, length=8)
end

@testset "SDDiP Examples" begin
for example in [
        "SDDiPnewsvendor.jl",
        "all_blacks.jl",
        "generation_expansion.jl",
        "stochastic_all_blacks.jl",
        "booking_management.jl",
        "airconditioning.jl",
        "vehicle_location.jl"
    ]
    @testset "$example" begin
        include(joinpath(examples_dir, example))
    end
end
end
