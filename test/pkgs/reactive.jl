using Base.Test
using Reactive

# Processes 1 messages from the Reactive event queue.
onestep() = Reactive.run(1)

x = Signal(2)
@test value(x) == 2

mul3 = map(a ->  3a, x)
@test value(mul3) == 6

push!(x, 3)
onestep()

@test value(x) == 3
@test value(mul3) == 9
