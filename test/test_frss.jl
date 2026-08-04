using LocalSmoothers
using Test

@testitem "FRSS constructor and fields" begin
    sm = FRSS([0.05, 0.1, 0.5], 0.2, 0.2)
    @test sm.initial_Js == [0.05, 0.1, 0.5]
    @test sm.medium_J == 0.2
    @test sm.final_J == 0.2
    @test String(sm) == "Smoothed_FRSS"
    # duplicates removed, values sorted
    sm2 = FRSS([0.5, 0.1, 0.1, 0.05], 0.2, 0.2)
    @test sm2.initial_Js == [0.05, 0.1, 0.5]
    # scalar accepted
    @test FRSS(0.1, 0.2, 0.3).initial_Js == [0.1]
    @test_throws ArgumentError FRSS([], 0.2, 0.2)
end

@testitem "FRSS smooths noisy data" begin
    using Statistics
    n = 300
    x = collect(range(0.0, 4.0, length=n))
    f = sin
    y = f.(x) .+ 0.2 .* randn(n)
    sm = FRSS([0.05, 0.1, 0.5], 0.2, 0.2)
    out = do_smoothing(x, y, sm)
    @test length(out) == n
    @test all(isfinite, out)
    rmse = sqrt(mean((out .- f.(x)) .^ 2))
    @test rmse < 0.15
end

@testitem "FRSS works with 2 and 4 candidate scales" begin
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    y = exp.(x) .+ 0.05 .* randn(n)
    for Js in ([0.05, 0.5], [0.02, 0.05, 0.1, 0.5])
        out = do_smoothing(x, y, FRSS(Js, 0.2, 0.2))
        @test all(isfinite, out)
        @test length(out) == n
    end
end

@testitem "FRSS accepts views" begin
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    y = cos.(4 .* x)
    idx = 1:2:n
    outv = do_smoothing(@view(x[idx]), @view(y[idx]), FRSS([0.1, 0.3], 0.2, 0.2))
    outd = do_smoothing(x[idx], y[idx], FRSS([0.1, 0.3], 0.2, 0.2))
    @test outv ≈ outd
end

@testitem "FRSS type stability" begin
    n = 150
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(x)
    @inferred do_smoothing(x, y, FRSS([0.1, 0.3], 0.2, 0.2))
end
