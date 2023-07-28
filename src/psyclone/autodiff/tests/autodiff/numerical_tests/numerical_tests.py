# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2021-2023, Science and Technology Facilities Council.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
# Author J. Remy, Inria

"""This file contains numerical tests of `psyclone.autodiff` reverse-mode \
transformations by comparing to results obtained using Tapenade.
NOTE: this is upcoming work and nothing much has been done here."""

from psyclone.autodiff import NumericalComparator, SubroutineGenerator

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

bar = SubroutineGenerator("bar")

a = bar.new_in_arg("a")
b = bar.new_inout_arg("b")
bar.new_assignment(b, add(a, b))


foo = SubroutineGenerator("foo")

x = foo.new_in_arg("x")
w = foo.new_in_arg("w")
f = foo.new_inout_arg("f")
g = foo.new_out_arg("g")

a = foo.new_variable("a")

foo.new_assignment(f, x)
foo.new_assignment(a, mul(f, w))
foo.new_assignment(g, w)
foo.new_call(bar, [f, g])
foo.new_assignment(f, add(mul(x, w), w))
foo.new_call(bar, [f, g])
foo.new_assignment(f, add(x, w))
foo.new_assignment(g, power(add(x, w), Literal("1.23456789", REAL_DOUBLE_TYPE)))
foo.new_assignment(f, add(mul(f, f), div(f, f)))
foo.new_call(bar, [f, g])
foo.new_assignment(g, x)
foo.new_assignment(g, power(g, Literal("3.0", REAL_DOUBLE_TYPE)))
foo.new_assignment(f, sub(exp(add(f, x)), g))

foo.new_call(bar, [f, g])


with open("foo.f90", "w") as file:
    file.write(bar.write())
    file.write(foo.write())

from psyclone.autodiff.ad_reversal_schedule import (
    ADSplitReversalSchedule,
    ADJointReversalSchedule,
)

# schedule = ADJointReversalSchedule()
schedule = ADSplitReversalSchedule()
# ComparatorGenerator('./tapenade_3.16/bin/tapenade', 'foo_bar.f90', 'foo', ['f', 'g'], ['x', 'w'], schedule, {'verbose': True})

argument_values = {"w": [4, 5, 6, 5.1], "x": [1, 2, 3, 5.2], "f": [1]}
NumericalComparator.compare(
    "./tapenade_3.16",
    "foo.f90",
    "foo",
    ["f", "g"],
    ["x", "w"],
    argument_values,
    schedule,
    "Linf_error",
    {"verbose": True, "inline_operation_adjoints": True},
)
