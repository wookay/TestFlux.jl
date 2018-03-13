module test_pkgs_macrotools

using Base.Test
using MacroTools # @capture
using MacroTools: prewalk, postwalk

@capture :[1, 2, 3, 4, 5, 6, 7]  [1, a_, 3, b__, c_]
@test (a, b, c) == (2, Any[4, 5, 6], 7)

end  # module test_pkgs_macrotools
