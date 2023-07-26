from collections import OrderedDict
from numerical_comparator import NumericalComparator
from subroutine_generator import FortranSubroutineGenerator

from psyclone.psyir.nodes import Literal
from psyclone.psyir.symbols import REAL_DOUBLE_TYPE

from psyclone.autodiff.utils import (
    datanode,
    one,
    zero,
    minus,
    inverse,
    power,
    sqrt,
    log,
    mul,
    sub,
    add,
    increment,
    assign,
    # assign_zero,
    sin,
    cos,
    square,
    div,
    exp,
    sign,
    own_routine_symbol,
)

bar = FortranSubroutineGenerator('bar')

a = bar.new_in_arg('a')
b = bar.new_inout_arg('b')
bar.new_assignment(b, add(a, b))


foo = FortranSubroutineGenerator('foo')

x = foo.new_in_arg('x')
w = foo.new_in_arg('w')
f = foo.new_inout_arg('f')
g = foo.new_out_arg('g')

a = foo.new_variable('a')

foo.new_assignment(f, x)
foo.new_assignment(a, mul(f, w))
foo.new_assignment(g, w)
foo.new_call(bar, [f, g])
foo.new_assignment(f, add(mul(x, w), w))
foo.new_call(bar, [f, g])
foo.new_assignment(f, add(x,w))
foo.new_assignment(g, power(add(x,w), Literal('1.23456789', REAL_DOUBLE_TYPE)))
foo.new_assignment(f, add(mul(f,f), div(f,f)))
foo.new_call(bar, [f, g])
#foo.new_assignment(g, x)
foo.new_assignment(g, power(g, Literal('3.0', REAL_DOUBLE_TYPE)))
foo.new_assignment(f, sub(exp(add(f, x)), g))

foo.new_call(bar, [f, g])


with open('foo.f90', 'w') as file:
    file.write(bar.write())
    file.write(foo.write())

from psyclone.autodiff.ad_reversal_schedule import ADSplitReversalSchedule

schedule = ADSplitReversalSchedule()
# ComparatorGenerator('./tapenade_3.16/bin/tapenade', 'foo_bar.f90', 'foo', ['f', 'g'], ['x', 'w'], schedule, {'verbose': True})

argument_values = {"w": [4, 5, 6, 5.1], "x": [1, 2, 3, 5.2], "f":[1]}
NumericalComparator.compare(
    "./tapenade_3.16",
    "foo.f90",
    "foo",
    ["f", "g"],
    ["x", "w"],
    argument_values,
    schedule,
    "Linf_error",
    {"verbose": True},
)