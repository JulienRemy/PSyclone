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
from psyclone.psyir.symbols import REAL_DOUBLE_TYPE

from psyclone.autodiff.ad_reversal_schedule import (
    ADSplitReversalSchedule,
    ADJointReversalSchedule,
    ADLinkReversalSchedule,
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
        UnaryOperation.Operator.ABS,
    ):
        print(f"Testing unary operator {op}")

        for iterative in (True, False):
            routine = SubroutineGenerator("routine_unary")

            x = routine.new_in_arg("x")
            f = routine.new_out_arg("f")

            if iterative:
                print("with an iterative assignment to f")
                routine.new_assignment(f, x)
                routine.new_assignment(f, UnaryOperation.create(op, Reference(f)))
            else:
                print("without iterative assignment")
                routine.new_assignment(f, UnaryOperation.create(op, Reference(x)))

            with open("routine.f90", "w") as file:
                file.write(routine.write())

            if op in (
                UnaryOperation.Operator.SQRT,
                UnaryOperation.Operator.LOG,
                UnaryOperation.Operator.LOG10,
            ):
                x_val = np.random.uniform(0, 1e2, 100)
            elif op in (UnaryOperation.Operator.ACOS, UnaryOperation.Operator.ASIN):
                x_val = np.random.uniform(-1, 1, 100)
            else:
                x_val = np.random.uniform(-1e2, 1e2, 100)

            rev_schedule = ADJointReversalSchedule()
            for inline in (True, False):
                print(f"with option 'inline_operation_adjoints' = {inline}")

                max_error, associated_values = NumericalComparator.compare(
                    "./tapenade_3.16",
                    "routine.f90",
                    "routine_unary",
                    ["f"],
                    ["x"],
                    {"x": x_val.tolist()},
                    rev_schedule,
                    "Linf_error",
                    {"verbose": True, "inline_operation_adjoints": inline},
                )

                if max_error != 0:
                    raise ValueError(
                        f"Test failed, Linf_error = {max_error} for argument values {associated_values}"
                    )
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

        for iterative in (True, False):
            routine = SubroutineGenerator("routine_binary")

            x = routine.new_in_arg("x")
            y = routine.new_in_arg("y")
            f = routine.new_out_arg("f")

            if iterative:
                print("with an iterative assignment to f")
                routine.new_assignment(f, x)
                routine.new_assignment(
                    f, BinaryOperation.create(op, Reference(f), Reference(y))
                )
            else:
                print("without iterative assignment")
                routine.new_assignment(
                    f, BinaryOperation.create(op, Reference(x), Reference(y))
                )

            # routine.new_assignment(f, BinaryOperation.create(op, Reference(x), Reference(y)))

            with open("routine.f90", "w") as file:
                file.write(routine.write())

            x_val = np.random.uniform(-1e2, 1e2, 1000)
            y_val = np.random.uniform(-1e2, 1e2, 1000)

            rev_schedule = ADJointReversalSchedule()
            for inline in (True, False):
                print(f"with option 'inline_operation_adjoints' = {inline}")

                max_error, associated_values = NumericalComparator.compare(
                    "./tapenade_3.16",
                    "routine.f90",
                    "routine_binary",
                    ["f"],
                    ["x", "y"],
                    {"x": x_val.tolist(), "y": y_val.tolist()},
                    rev_schedule,
                    "Linf_error",
                    {"verbose": True, "inline_operation_adjoints": inline},
                )

                if max_error != 0:
                    raise ValueError(
                        f"Test failed, Linf_error = {max_error} for argument values {associated_values}"
                    )
                print("passed")

                print("-----------------------")

        print("===============================\n")


def test_unary_composition():
    unary_operators = (
        UnaryOperation.Operator.PLUS,
        UnaryOperation.Operator.MINUS,
        # UnaryOperation.Operator.SQRT,      #positive arg only
        UnaryOperation.Operator.EXP,
        # UnaryOperation.Operator.LOG,       #positive arg only
        # UnaryOperation.Operator.LOG10,     #positive arg only
        UnaryOperation.Operator.COS,
        UnaryOperation.Operator.SIN,
        UnaryOperation.Operator.TAN,
        # UnaryOperation.Operator.ACOS,      #[1,1] arg only
        # UnaryOperation.Operator.ASIN,      #[1,1] arg only
        UnaryOperation.Operator.ATAN,
        UnaryOperation.Operator.ABS,
    )

    for unary_1, unary_2 in product(unary_operators, unary_operators):
        print(f"Testing composition of {unary_1} and {unary_2}")

        for iterative in (True, False):
            routine = SubroutineGenerator("routine_unary_composition")

            x = routine.new_in_arg("x")
            f = routine.new_out_arg("f")

            if iterative:
                print("with an iterative assignment to f")
                routine.new_assignment(f, x)
                routine.new_assignment(
                    f,
                    UnaryOperation.create(
                        unary_1, UnaryOperation.create(unary_2, Reference(f))
                    ),
                )
            else:
                print("without iterative assignment")
                routine.new_assignment(
                    f,
                    UnaryOperation.create(
                        unary_1, UnaryOperation.create(unary_2, Reference(x))
                    ),
                )

            with open("routine.f90", "w") as file:
                file.write(routine.write())

            x_val = np.random.uniform(-1e2, 1e2, 10)

            rev_schedule = ADJointReversalSchedule()
            for inline in (True, False):
                print(f"with option 'inline_operation_adjoints' = {inline}")

                max_error, associated_values = NumericalComparator.compare(
                    "./tapenade_3.16",
                    "routine.f90",
                    "routine_unary_composition",
                    ["f"],
                    ["x"],
                    {"x": x_val.tolist()},
                    rev_schedule,
                    "Linf_error",
                    {"verbose": True, "inline_operation_adjoints": inline},
                )

                if max_error != 0:
                    raise ValueError(
                        f"Test failed, Linf_error = {max_error} for argument values {associated_values}"
                    )
                print("passed")
                print("-----------------------")

        print("===============================\n")


def test_binary_composition():
    binary_operators = (
        BinaryOperation.Operator.ADD,
        BinaryOperation.Operator.SUB,
        BinaryOperation.Operator.MUL,
        BinaryOperation.Operator.DIV,
        BinaryOperation.Operator.POW,
    )

    for binary_1, binary_2 in product(binary_operators, binary_operators):
        print(f"Testing composition of {binary_1} and {binary_2}")

        for iterative in (True, False):
            routine = SubroutineGenerator("routine_binary_composition")

            x = routine.new_in_arg("x")
            y = routine.new_in_arg("y")
            f = routine.new_out_arg("f")

            if iterative:
                print("with an iterative assignment to f")
                routine.new_assignment(f, x)
                routine.new_assignment(
                    f,
                    BinaryOperation.create(
                        binary_1,
                        BinaryOperation.create(binary_2, Reference(f), Reference(x)),
                        Reference(y),
                    ),
                )
            else:
                print("without iterative assignment")
                routine.new_assignment(
                    f,
                    BinaryOperation.create(
                        binary_1,
                        BinaryOperation.create(binary_2, Reference(x), Reference(y)),
                        Reference(y),
                    ),
                )

            with open("routine.f90", "w") as file:
                file.write(routine.write())

            x_val = np.random.uniform(-1e2, 1e2, 10)
            y_val = np.random.uniform(-1e2, 1e2, 10)

            rev_schedule = ADJointReversalSchedule()
            for inline in (True, False):
                print(f"with option 'inline_operation_adjoints' = {inline}")

                max_error, associated_values = NumericalComparator.compare(
                    "./tapenade_3.16",
                    "routine.f90",
                    "routine_binary_composition",
                    ["f"],
                    ["x", "y"],
                    {"x": x_val.tolist(), "y": y_val.tolist()},
                    rev_schedule,
                    "Linf_error",
                    {"verbose": True, "inline_operation_adjoints": inline},
                )

                if max_error != 0:
                    raise ValueError(
                        f"Test failed, Linf_error = {max_error} for argument values {associated_values}"
                    )
                print("passed")
                print("-----------------------")

        print("===============================\n")


def _create_taping_routine(name):
    # Only use non linear operators to force reuse of taped values
    unary_operators = (
        # UnaryOperation.Operator.PLUS,
        # UnaryOperation.Operator.MINUS,
        UnaryOperation.Operator.SQRT,
        UnaryOperation.Operator.EXP,
        UnaryOperation.Operator.LOG,
        UnaryOperation.Operator.LOG10,
        UnaryOperation.Operator.COS,
        UnaryOperation.Operator.SIN,
        UnaryOperation.Operator.TAN,
        # UnaryOperation.Operator.ACOS,
        # UnaryOperation.Operator.ASIN,
        UnaryOperation.Operator.ATAN,
        # UnaryOperation.Operator.ABS,
    )
    binary_operators = (
        # BinaryOperation.Operator.ADD,
        # BinaryOperation.Operator.SUB,
        BinaryOperation.Operator.MUL,
        BinaryOperation.Operator.DIV,
        BinaryOperation.Operator.POW,
    )
    routine = SubroutineGenerator(name)

    x = routine.new_inout_arg("x")
    a = routine.new_variable("a")
    f = routine.new_out_arg("f")

    for unary in unary_operators:
        routine.new_assignment(
            x,
            BinaryOperation.create(
                BinaryOperation.Operator.MUL,
                Literal("1.01", REAL_DOUBLE_TYPE),
                Reference(x),
            ),
        )

        routine.new_assignment(
            f,
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                Reference(f),
                UnaryOperation.create(unary, Reference(x)),
            ),
        )

    for binary in binary_operators:
        routine.new_assignment(
            x,
            BinaryOperation.create(
                BinaryOperation.Operator.MUL,
                Literal("1.01", REAL_DOUBLE_TYPE),
                Reference(x),
            ),
        )

        routine.new_assignment(
            a, Literal(str(np.random.uniform(-1, 1)), REAL_DOUBLE_TYPE)
        )

        routine.new_assignment(
            f,
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                Reference(f),
                BinaryOperation.create(binary, Reference(x), Reference(a)),
            ),
        )

    return routine


def test_taping():
    routine = _create_taping_routine("routine_taping")

    with open("routine.f90", "w") as file:
        file.write(routine.write())

    rev_schedule = ADJointReversalSchedule()
    for inline in (True, False):
        print(f"Testing a routine with many (used) taped values")
        print(f"with option 'inline_operation_adjoints' = {inline}")

        max_error, associated_values = NumericalComparator.compare(
            "./tapenade_3.16",
            "routine.f90",
            "routine_taping",
            ["f"],
            ["x"],
            {"x": [0.1]},
            rev_schedule,
            "Linf_error",
            {"verbose": True, "inline_operation_adjoints": inline},
        )

        if max_error != 0:
            raise ValueError(f"Test failed, Linf_error = {max_error}")
        print("passed")
        print("-----------------------")

    print("===============================\n")


def test_reversal_schedules():
    # Ensure that both routines tape and reuse taped values
    called_routine_1 = _create_taping_routine("called_1")
    called_routine_2 = _create_taping_routine("called_2")

    calling_routine = SubroutineGenerator("calling")

    x = calling_routine.new_inout_arg("x")
    f = calling_routine.new_out_arg("f")

    for i in range(3):
        calling_routine.new_assignment(
            x,
            BinaryOperation.create(
                BinaryOperation.Operator.MUL,
                Literal("1.01", REAL_DOUBLE_TYPE),
                Reference(x),
            ),
        )
        calling_routine.new_call(called_routine_1, [x, f])

    for i in range(3):
        calling_routine.new_assignment(
            x,
            BinaryOperation.create(
                BinaryOperation.Operator.MUL,
                Literal("1.01", REAL_DOUBLE_TYPE),
                Reference(x),
            ),
        )
        calling_routine.new_call(called_routine_2, [x, f])

    with open("routine.f90", "w") as file:
        file.write(called_routine_1.write())
        file.write(called_routine_2.write())
        file.write(calling_routine.write())

    reversal_schedules = (
        ADJointReversalSchedule(),
        ADSplitReversalSchedule(),
        ADLinkReversalSchedule(
            strong_links=[["calling", "called_1"]], weak_links=[["calling", "called_2"]]
        ),
    )

    for schedule in reversal_schedules:
        print(f"Testing reversal schedule {type(schedule).__name__}")

        max_error, associated_values = NumericalComparator.compare(
            "./tapenade_3.16",
            "routine.f90",
            "calling",
            ["f"],
            ["x"],
            {"x": [0.1]},
            schedule,
            "Linf_error",
            {"verbose": True, "inline_operation_adjoints": False},
        )

        if max_error != 0:
            raise ValueError(
                f"Test failed, Linf_error = {max_error} for schedule {type(schedule).__name__}"
            )
        print("passed")
        print("-----------------------")

from psyclone.line_length import FortLineLength

def test_many_arguments():
    routine = SubroutineGenerator("routine_many")

    in_args = []
    inout_args = []
    out_args = []
    undef_args = []
    non_args = []

    n = 4

    for i in range(n * 5):
        if i < n*1:
            arg = routine.new_in_arg(f"arg{i}")
            in_args.append(arg)
        elif i < n*2:
            arg = routine.new_out_arg(f"arg{i}")
            out_args.append(arg)
        elif i < n*3:
            arg = routine.new_inout_arg(f"arg{i}")
            inout_args.append(arg)
        elif i < n*4:
            arg = routine.new_arg(f"arg{i}")
            undef_args.append(arg)
        else:
            arg = routine.new_variable(f"arg{i}")
            routine.new_assignment(arg, Literal(f"{i}.0", REAL_DOUBLE_TYPE))
            non_args.append(arg)

    for i in range(n):
        routine.new_assignment(out_args[i], in_args[i])
        routine.new_assignment(
            out_args[i],
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                Reference(out_args[i]),
                Reference(inout_args[i]),
            ),
        )
        routine.new_assignment(
            out_args[i],
            BinaryOperation.create(
                BinaryOperation.Operator.ADD,
                Reference(out_args[i]),
                Reference(undef_args[i]),
            ),
        )
        routine.new_assignment(
            out_args[i],
            BinaryOperation.create(
                BinaryOperation.Operator.ADD, Reference(out_args[i]), Reference(non_args[i])
            ),
        )

        for modified_args in (inout_args, undef_args):
            routine.new_assignment(
                modified_args[i],
                BinaryOperation.create(
                    BinaryOperation.Operator.ADD,
                    Reference(modified_args[i]),
                    Reference(in_args[i]),
                ),
            )
            routine.new_assignment(
                modified_args[i],
                BinaryOperation.create(
                    BinaryOperation.Operator.ADD,
                    Reference(modified_args[i]),
                    Reference(inout_args[i]),
                ),
            )
            routine.new_assignment(
                modified_args[i],
                BinaryOperation.create(
                    BinaryOperation.Operator.ADD,
                    Reference(modified_args[i]),
                    Reference(undef_args[i]),
                ),
            )
            routine.new_assignment(
                modified_args[i],
                BinaryOperation.create(
                    BinaryOperation.Operator.ADD, Reference(modified_args[i]), Reference(non_args[i])
                ),
            )

    with open("routine.f90", "w") as file:
        file.write(routine.write())

    rev_schedule = ADJointReversalSchedule()
    print(f"Testing a routine with many arguments")

    independent_names = []
    dependent_names = []

    for arg in in_args + inout_args + undef_args:
        independent_names.append(arg.name)

    for arg in out_args + inout_args + undef_args:
        dependent_names.append(arg.name)

    values = {}
    for i, arg in enumerate(in_args + inout_args + undef_args):
        values[arg.name] = [float(i)]

    max_error, associated_values = NumericalComparator.compare(
        "./tapenade_3.16",
        "routine.f90",
        "routine_many",
        dependent_names,
        independent_names,
        values,
        rev_schedule,
        "Linf_error",
        {"verbose": True, "inline_operation_adjoints": False},
    )

    if max_error != 0:
        raise ValueError(f"Test failed, Linf_error = {max_error}")
    print("passed")
    print("-----------------------")

    print("===============================\n")

if __name__ == "__main__":
    test_unary()
    test_binary()
    test_unary_composition()
    test_binary_composition()
    test_taping()
    test_reversal_schedules()
    test_many_arguments()
    # test_vars()
