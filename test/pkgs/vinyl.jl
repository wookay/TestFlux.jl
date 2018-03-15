using Base.Test
using Vinyl: @overdub, @hook

buf = IOBuffer()

struct TraceCtx end
@hook    TraceCtx   (f::Any)(xs...) = println(buf, "Called $(:($f($(xs...))))")
@overdub TraceCtx() 1+2

@test String(take!(buf)) == """
Called (+)(1, 2)
Called (add_int)(1, 2)
"""
