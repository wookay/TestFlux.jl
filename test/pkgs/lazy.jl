module test_pkgs_lazy

using Base.Test
using Lazy: @>

@test '가'         == @> 0xac00 Char
@test ['가', '가'] == @> 0xac00 Char fill(2)

end # module test_pkgs_lazy
