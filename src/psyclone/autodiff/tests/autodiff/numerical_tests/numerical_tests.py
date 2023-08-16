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

from itertools import product
import numpy as np

from psyclone.autodiff import NumericalComparator, SubroutineGenerator

from psyclone.psyir.nodes import Literal, UnaryOperation, BinaryOperation, Reference
from psyclone.psyir.symbols import REAL_SINGLE_TYPE

from psyclone.autodiff.ad_reversal_schedule import (
    ADSplitReversalSchedule,
    ADJointReversalSchedule,
)
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

# TODO: solve datatype problem with abs
def test_unary():
    for op in (
        UnaryOperation.Operator.PLUS,
        UnaryOperation.Operator.MINUS,
        UnaryOperation.Operator.SQRT,
        UnaryOperation.Operator.EXP,
        UnaryOperation.Operator.LOG,
        UnaryOperation.Operator.LOG10,
        UnaryOperation.Operator.COS,
        UnaryOperation.Operator.SIN,
        UnaryOperation.Operator.TAN,
        UnaryOperation.Operator.ACOS,
        UnaryOperation.Operator.ASIN,
        UnaryOperation.Operator.ATAN,
        #UnaryOperation.Operator.ABS, #datatype problem
    ):
        print(f"Testing unary operator {op}")

        foo = SubroutineGenerator("foo")

        x = foo.new_in_arg("x")
        f = foo.new_out_arg("f")

        for iterative in (True, False):
            if iterative:
                print("with an iterative assignment to f")
                foo.new_assignment(f, x)
                foo.new_assignment(f, UnaryOperation.create(op, Reference(f)))
            else:
                print("without iterative assignment")
                foo.new_assignment(f, UnaryOperation.create(op, Reference(x)))

            with open("foo.f90", "w") as file:
                file.write(foo.write())

            if op in (UnaryOperation.Operator.SQRT, UnaryOperation.Operator.LOG, UnaryOperation.Operator.LOG10):
                x_val = np.random.uniform(0, 1e2, 100)
            else:
                x_val = np.random.uniform(-1e2, 1e2, 100)

            rev_schedule = ADJointReversalSchedule()
            for inline in (True, False):
                print(f"with option 'inline_operation_adjoints' = {inline}")

                max_error, associated_values = NumericalComparator.compare(
                "./tapenade_3.16",
                "foo.f90",
                "foo",
                ["f"],
                ["x"],
                {"x" : x_val.tolist()},
                rev_schedule,
                "Linf_error",
                {"verbose": True, "inline_operation_adjoints": inline},
                )

                if max_error != 0:
                    raise ValueError(f"Test failed, Linf_error = {max_error} for argument values {associated_values}")
                print("passed")
                print("-----------------------")

        print("===============================\n")

def test_binary():
    for op in (
        BinaryOperation.Operator.ADD,
        BinaryOperation.Operator.SUB,
        BinaryOperation.Operator.MUL,
        BinaryOperation.Operator.DIV,
        BinaryOperation.Operator.POW,
    ):
        print(f"Testing binary operator {op}")

        foo = SubroutineGenerator("foo")

        x = foo.new_in_arg("x")
        y = foo.new_in_arg("y")
        f = foo.new_out_arg("f")

        for iterative in (True, False):
            if iterative:
                print("with an iterative assignment to f")
                foo.new_assignment(f, x)
                foo.new_assignment(f, BinaryOperation.create(op, Reference(f), Reference(y)))
            else:
                print("without iterative assignment")
                foo.new_assignment(f, BinaryOperation.create(op, Reference(x), Reference(y)))

            #foo.new_assignment(f, BinaryOperation.create(op, Reference(x), Reference(y)))

            with open("foo.f90", "w") as file:
                file.write(foo.write())

            x_val = np.random.uniform(-1e2, 1e2, 1000)
            y_val = np.random.uniform(-1e2, 1e2, 1000)

            rev_schedule = ADJointReversalSchedule()
            for inline in (True, False):
                print(f"with option 'inline_operation_adjoints' = {inline}")

                max_error, associated_values = NumericalComparator.compare(
                "./tapenade_3.16",
                "foo.f90",
                "foo",
                ["f"],
                ["x", "y"],
                {"x" : x_val.tolist(), "y" : y_val.tolist()},
                rev_schedule,
                "Linf_error",
                {"verbose": True, "inline_operation_adjoints": inline},
                )

                if max_error != 0:
                    raise ValueError(f"Test failed, Linf_error = {max_error} for argument values {associated_values}")
                print("passed")

                print("-----------------------")

        print("===============================\n")
        
"""def test_vars():
    for dep_n in range(1,11):
        for indep_n in range(1,11):
            print(f"Testing with {dep_n} dependent variables and {indep_n} independent_variables")

            foo = SubroutineGenerator("foo")

            dep_1 = foo.new_out_arg("dep_1")
            for j in range(2, dep_n + 1):
                foo.new_out_arg(f"dep_{j}")

            for i in range(1, indep_n + 1):
                indep = foo.new_in_arg(f"indep_{i}")
                foo.new_assignment(dep_1, indep)

            #############################################

            with open("foo.f90", "w") as file:
                file.write(foo.write())

            rev_schedule = ADJointReversalSchedule()
            max_error, associated_values = NumericalComparator.compare(
                "./tapenade_3.16",
                "foo.f90",
                "foo",
                [f"dep_{j}" for j in range(1, dep_n + 1)],
                [f"indep_{i}" for i in range(1, indep_n + 1)],
                {f"indep_{i}": [1] for i in range(1, indep_n + 1)},
                rev_schedule,
                "Linf_error",
                {"verbose": True},
                )

            if max_error != 0:
                raise ValueError(f"Test failed, Linf_error = {max_error} for argument values {associated_values}")
            print("passed")

            print("-----------------------")"""

if __name__ == "__main__":
    test_unary()
    test_binary()
    #test_vars()
