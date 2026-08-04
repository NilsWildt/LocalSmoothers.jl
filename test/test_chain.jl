using LocalSmoothers
using Test

@testitem "chaining equals sequential application" begin
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(5 .* x) .+ 0.05 .* randn(n)
    sms = [LASb(11), LLSSb(11)]
    chained = do_smoothing(x, y, sms)
    sequential = do_smoothing(x, do_smoothing(x, y, LASb(11)), LLSSb(11))
    @test chained ≈ sequential
end

@testitem "variadic chaining equals vector chaining" begin
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    y = cos.(4 .* x)
    a = do_smoothing(x, y, LAS(9), LLSS(9))
    b = do_smoothing(x, y, [LAS(9), LLSS(9)])
    @test a ≈ b
end

@testitem "in-place chaining" begin
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(x)
    sms = [LASb(11), LLSSb(11)]
    out = similar(y)
    do_smoothing!(out, x, y, sms)
    @test out ≈ do_smoothing(x, y, sms)
end

@testitem "mixed smoother vector with kernels" begin
    n = 150
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(x) .+ 0.01 .* randn(n)
    sms = [LAS(9), NWKernelsmooth(Gaussian(0.2))]
    out = do_smoothing(x, y, sms)
    @test all(isfinite, out)
    @test length(out) == n
end
