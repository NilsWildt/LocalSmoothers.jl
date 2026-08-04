using LocalSmoothers
using Test

@testitem "Aqua quality checks" begin
    using Aqua
    Aqua.test_all(LocalSmoothers)
end

@testitem "JET static analysis" begin
    using JET
    JET.test_package(LocalSmoothers)
end
