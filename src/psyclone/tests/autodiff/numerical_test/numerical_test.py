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
# Author: J. Remy, Université Grenoble Alpes, Inria

"""This file contains numerical tests of `psyclone.autodiff` forward- and reverse-mode \
transformations by comparing to results obtained using Tapenade.
NOTE: this is a work in progress."""

from itertools import product
import numpy as np
import pytest

from psyclone.autodiff import NumericalComparator, SubroutineGenerator

from psyclone.psyir.nodes import Literal, UnaryOperation, BinaryOperation, Reference
from psyclone.psyir.symbols import REAL_TYPE, ArrayType

from psyclone.autodiff.ad_reversal_schedule import (
    ADSplitReversalSchedule,
    ADJointReversalSchedule,
    ADLinkReversalSchedule,
)

MAX_ERROR = 1e-5

MODES = ("forward", "reverse")

def _iterative_and_inline_modes(mode):
    if mode == "forward":
        return (False, )
    else:
        return (True, False)

def _datatype(vector):
    if vector:
        return ArrayType(REAL_TYPE, [5])
    return REAL_TYPE

def _input_shape(n, vector):
    if vector:
        return (n, 5)
    return n

@pytest.mark.parametrize("mode, vector, op",
                         product(("forward", "reverse"),
                                 (False, True),
                                 (UnaryOperation.Operator.PLUS,
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
                                  UnaryOperation.Operator.ABS)))
def test_unary(mode, vector, op):
    """Test all unary operators in both modes, by applying `psyclone.autodiff` \
    and Tapenade transformations to a subroutine computing `f = op(x)` or \
    `f = x; f = op(f)` and comparing numerical results for random values of `x`.

    :raises ValueError: if the error is above MAX_ERROR or is NaN.
    """
    print(f"Testing unary operator {op} in {mode}-mode with{'' if vector else 'out'} vectors")

    for iterative in _iterative_and_inline_modes(mode):
        routine = SubroutineGenerator("routine_unary")

        x = routine.new_in_arg("x", _datatype(vector))
        f = routine.new_out_arg("f", _datatype(vector))

        if iterative:
            print("with an iterative assignment to f")
            routine.new_assignment(f, x)
            routine.new_assignment(f, UnaryOperation.create(op, Reference(f)))
        else:
            print("without iterative assignment")
            routine.new_assignment(f, UnaryOperation.create(op, Reference(x)))

        with open("./outputs/routine.f90", "w") as file:
            file.write(routine.write())

        rg = np.random.default_rng(123456)
        # Operators requiring a positive argument
        if op in (
            UnaryOperation.Operator.SQRT,
            UnaryOperation.Operator.LOG,
            UnaryOperation.Operator.LOG10,
        ):
            x_val = rg.uniform(0, 2, _input_shape(10, vector))
        # Operators requiring an argument in [-1, 1]
        elif op in (UnaryOperation.Operator.ACOS, UnaryOperation.Operator.ASIN):
            x_val = rg.uniform(-1, 1, _input_shape(10, vector))
        else:
            x_val = rg.uniform(-2, 2, _input_shape(10, vector))

        # Whichever schedule will work
        rev_schedule = ADJointReversalSchedule()

        for inline in _iterative_and_inline_modes(mode):
            print(f"with option 'inline_operation_adjoints' = {inline}")

            max_error, associated_values = NumericalComparator.compare(
                "./tapenade_3.16",
                "./outputs/routine.f90",
                "routine_unary",
                ["f"],
                ["x"],
                {"x": x_val.tolist()},
                "Linf_error",
                {"verbose": True, "inline_operation_adjoints": inline},
                mode,
                rev_schedule
            )

            if max_error > MAX_ERROR or np.isnan(max_error):
                raise ValueError(
                    f"Test failed, Linf_error = {max_error} for argument values {associated_values}"
                )
            print("passed")
            print("-----------------------")

    print("===============================\n")

@pytest.mark.parametrize("mode, vector, op",
                         product(("forward", "reverse"),
                                 (False, True),
                                 (BinaryOperation.Operator.ADD,
                                  BinaryOperation.Operator.SUB,
                                  BinaryOperation.Operator.MUL,
                                  BinaryOperation.Operator.DIV,
                                  BinaryOperation.Operator.POW,)))
def test_binary(mode, vector, op):
    """Test all binary operators in both modes, by applying `psyclone.autodiff` \
    and Tapenade transformations to a subroutine computing `f = x (op) y` or \
    `f = x; f = f (op) y` and comparing numerical results for random values of \
    `x` and `y`.

    :raises ValueError: if the error is above MAX_ERROR or is NaN.
    """
    print(f"Testing binary operator {op} in {mode}-mode")

    for iterative in _iterative_and_inline_modes(mode):
        routine = SubroutineGenerator("routine_binary")

        x = routine.new_in_arg("x", _datatype(vector))
        y = routine.new_in_arg("y", _datatype(vector))
        f = routine.new_out_arg("f", _datatype(vector))

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

        with open("./outputs/routine.f90", "w") as file:
            file.write(routine.write())

        rg = np.random.default_rng(123456)
        if op is BinaryOperation.Operator.POW:
            x_val = rg.uniform(0, 2, _input_shape(10, vector))
        else:
            x_val = rg.uniform(-2, 2,_input_shape(10, vector))
        y_val = rg.uniform(-2, 2, _input_shape(10, vector))

        rev_schedule = ADJointReversalSchedule()
        for inline in _iterative_and_inline_modes(mode):
            print(f"with option 'inline_operation_adjoints' = {inline}")

            max_error, associated_values = NumericalComparator.compare(
                "./tapenade_3.16",
                "./outputs/routine.f90",
                "routine_binary",
                ["f"],
                ["x", "y"],
                {"x": x_val.tolist(), "y": y_val.tolist()},
                "Linf_error",
                {"verbose": True, "inline_operation_adjoints": inline},
                mode,
                rev_schedule
            )

            if max_error > MAX_ERROR or np.isnan(max_error):
                raise ValueError(
                    f"Test failed, Linf_error = {max_error} for argument values {associated_values}"
                )
            print("passed")

            print("-----------------------")

    print("===============================\n")

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
@pytest.mark.parametrize("mode, vector, unaries",
                         product(("forward", "reverse"),
                                 (False, True),
                                 product(unary_operators, unary_operators)))
def test_unary_composition(mode, vector, unaries):
    """Test composition of unary operators in both modes, by applying \
    `psyclone.autodiff` and Tapenade transformations to a subroutine computing \
    `f = op1(op2(x))` or `f = x; f = op1(op2(f))` and comparing numerical \
    results for random values of `x`.

    :raises ValueError: if the error is above MAX_ERROR or is NaN.
    """
    unary_1, unary_2 = unaries
    print(f"Testing composition of {unary_1} and {unary_2} in {mode}-mode")

    for iterative in _iterative_and_inline_modes(mode):
        routine = SubroutineGenerator("routine_unary_composition")

        x = routine.new_in_arg("x", _datatype(vector))
        f = routine.new_out_arg("f", _datatype(vector))

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

        with open("./outputs/routine.f90", "w") as file:
            file.write(routine.write())

        rg = np.random.default_rng(123456)
        x_val = rg.uniform(-1, 1, _input_shape(10, vector))

        rev_schedule = ADJointReversalSchedule()
        for inline in _iterative_and_inline_modes(mode):
            print(f"with option 'inline_operation_adjoints' = {inline}")

            max_error, associated_values = NumericalComparator.compare(
                "./tapenade_3.16",
                "./outputs/routine.f90",
                "routine_unary_composition",
                ["f"],
                ["x"],
                {"x": x_val.tolist()},
                "Linf_error",
                {"verbose": True, "inline_operation_adjoints": inline},
                mode,
                rev_schedule,
            )

            if max_error > MAX_ERROR or np.isnan(max_error):
                raise ValueError(
                    f"Test failed, Linf_error = {max_error} for argument values {associated_values}"
                )
            print("passed")
            print("-----------------------")

    print("===============================\n")

binary_operators = (
                BinaryOperation.Operator.ADD,
                BinaryOperation.Operator.SUB,
                BinaryOperation.Operator.MUL,
                BinaryOperation.Operator.DIV,
                #BinaryOperation.Operator.POW,
            )
@pytest.mark.parametrize("mode, vector, binaries",
                         product(("forward", "reverse"),
                                 (False, True),
                                 product(binary_operators, binary_operators)))
def test_binary_composition(mode, vector, binaries):
    """Test composition of binary operators in both modes, by applying \
    `psyclone.autodiff` and Tapenade transformations to a subroutine computing \
    `f = (x (op1) y) (op2) z` or `f = x; f = (f (op1) y) (op2) z` and comparing \
    numerical results for random values of `x`, `y` and `z`.

    :raises ValueError: if the error is above MAX_ERROR or is NaN.
    """
    binary_1, binary_2 = binaries
    print(f"Testing composition of {binary_1} and {binary_2} in {mode}-mode")

    for iterative in _iterative_and_inline_modes(mode):
        routine = SubroutineGenerator("routine_binary_composition")

        x = routine.new_in_arg("x", _datatype(vector))
        y = routine.new_in_arg("y", _datatype(vector))
        z = routine.new_in_arg("z", _datatype(vector))
        f = routine.new_out_arg("f", _datatype(vector))

        if iterative:
            print("with an iterative assignment to f")
            routine.new_assignment(f, x)
            routine.new_assignment(
                f,
                BinaryOperation.create(
                    binary_1,
                    BinaryOperation.create(binary_2, Reference(f), Reference(y)),
                    Reference(z),
                ),
            )
        else:
            print("without iterative assignment")
            routine.new_assignment(
                f,
                BinaryOperation.create(
                    binary_1,
                    BinaryOperation.create(binary_2, Reference(x), Reference(y)),
                    Reference(z),
                ),
            )

        with open("./outputs/routine.f90", "w") as file:
            file.write(routine.write())

        rg = np.random.default_rng(123456)
        x_val = rg.uniform(1, 3, _input_shape(10, vector))
        y_val = rg.uniform(1, 3, _input_shape(10, vector))
        z_val = rg.uniform(1, 3, _input_shape(10, vector))

        rev_schedule = ADJointReversalSchedule()
        for inline in _iterative_and_inline_modes(mode):
            print(f"with option 'inline_operation_adjoints' = {inline}")

            max_error, associated_values = NumericalComparator.compare(
                "./tapenade_3.16",
                "./outputs/routine.f90",
                "routine_binary_composition",
                ["f"],
                ["x", "y"],
                {"x": x_val.tolist(), "y": y_val.tolist(), "z": z_val.tolist()},
                "Linf_error",
                {"verbose": True, "inline_operation_adjoints": inline},
                mode,
                rev_schedule,
            )

            if max_error > MAX_ERROR or np.isnan(max_error):
                raise ValueError(
                    f"Test failed, Linf_error = {max_error} for argument values {associated_values}"
                )
            print("passed")
            print("-----------------------")

    print("===============================\n")


def _create_taping_routine(name, vector):
    """Generate a subroutine using non-linear unary and binary operators.

    :param name: name of the routine to generate.
    :type name: Str

    :return: subroutine generator.
    :rtype: :py:class:`psyclone.autodiff.SubroutineGenerator`
    """
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
        # BinaryOperation.Operator.POW,         # TODO: this one crashes the test!
    )
    routine = SubroutineGenerator(name)

    x = routine.new_inout_arg("x", _datatype(vector))
    a = routine.new_variable("a")
    f = routine.new_out_arg("f", _datatype(vector))

    rg = np.random.default_rng(123456)

    # x = 1.01 * x          ! to force taping argument
    # f = f + unary_op(x)
    for unary in unary_operators:
        routine.new_assignment(
            x,
            BinaryOperation.create(
                BinaryOperation.Operator.MUL,
                Literal("1.01", REAL_TYPE),
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

    # x = 1.01 * x          ! to force taping argument
    # a = random(-1, 1)     ! to force taping literal argument
    # f = f + x (binary_op) a
    for binary in binary_operators:
        routine.new_assignment(
            x,
            BinaryOperation.create(
                BinaryOperation.Operator.MUL,
                Literal("1.01", REAL_TYPE),
                Reference(x),
            ),
        )

        routine.new_assignment(
            a, Literal(str(rg.uniform(-1, 1)), REAL_TYPE)
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

@pytest.mark.parametrize("vector, inline",
                         product((False, True),
                                 (False, True)))
def test_taping(vector, inline):
    """Test taping function values in reverse-mode, by applying `psyclone.autodiff` \
    and Tapenade transformations to a subroutine computing non-linear operations,
    both unary and binary, and comparing numerical results.

    :raises ValueError: if the error is above MAX_ERROR or is NaN.
    """
    routine = _create_taping_routine("routine_taping", vector)

    with open("./outputs/routine.f90", "w") as file:
        file.write(routine.write())

    rev_schedule = ADJointReversalSchedule()

    print(f"Testing a routine with many (used) taped values")
    print(f"with option 'inline_operation_adjoints' = {inline}")

    max_error, associated_values = NumericalComparator.compare(
        "./tapenade_3.16",
        "./outputs/routine.f90",
        "routine_taping",
        ["f"],
        ["x"],
        {"x": [0.1, 0.1, 0.1, 0.1, 0.1]},
        "Linf_error",
        {"verbose": True, "inline_operation_adjoints": inline},
        "reverse",
        rev_schedule,
    )

    if max_error > MAX_ERROR or np.isnan(max_error):
        raise ValueError(f"Test failed, Linf_error = {max_error}")
    print("passed")
    print("-----------------------")

    print("===============================\n")

@pytest.mark.parametrize("vector, schedule",
                         product((False, True),
                                 (ADJointReversalSchedule(),
                                  ADSplitReversalSchedule(),
                                  ADLinkReversalSchedule(strong_links=[["calling", "called_1"]], weak_links=[["calling", "called_2"]]),
                                  None
                         )))
def test_nested_calls(vector, schedule):
    """Test nested subroutine calls in both modes and with all three possible 
    reversal schedules in reverse-mode by applying `psyclone.autodiff` \
    and Tapenade transformations to a subroutine calling two inner subroutines 
    that compute non-linear operations, both unary and binary, and comparing \
    numerical results.

    :raises ValueError: if the error is above MAX_ERROR or is NaN.
    """
    print("Testing nested calls")

    # Ensure that both routines tape and reuse taped values
    called_routine_1 = _create_taping_routine("called_1", vector)
    called_routine_2 = _create_taping_routine("called_2", vector)

    calling_routine = SubroutineGenerator("calling")

    x = calling_routine.new_inout_arg("x", _datatype(vector))
    f = calling_routine.new_out_arg("f", _datatype(vector))

    # f = 0.0
    calling_routine.new_assignment(f, Literal("0.0", REAL_TYPE))

    # Thrice
    # x = 1.01 * x
    # call called_routine_1(x, f)
    # x = 1.01 * x
    # call called_routine_2(x, f)
    for i in range(3):
        calling_routine.new_assignment(
            x,
            BinaryOperation.create(
                BinaryOperation.Operator.MUL,
                Literal("1.01", REAL_TYPE),
                Reference(x),
            ),
        )
        calling_routine.new_call(called_routine_1, [x, f])

        calling_routine.new_assignment(
            x,
            BinaryOperation.create(
                BinaryOperation.Operator.MUL,
                Literal("1.01", REAL_TYPE),
                Reference(x),
            ),
        )
        calling_routine.new_call(called_routine_2, [x, f])

    with open("./outputs/routine.f90", "w") as file:
        file.write(called_routine_1.write())
        file.write(called_routine_2.write())
        file.write(calling_routine.write())

    print(f"Testing in reverse-mode using reversal schedule {type(schedule).__name__}")

    if schedule:
        max_error, associated_values = NumericalComparator.compare(
            "./tapenade_3.16",
            "./outputs/routine.f90",
            "calling",
            ["f"],
            ["x"],
            {"x": [0.1, 0.1, 0.1, 0.1, 0.1]},
            "Linf_error",
            {"verbose": True, "inline_operation_adjoints": False},
            "reverse",
            schedule,
        )
    else:
        max_error, associated_values = NumericalComparator.compare(
            "./tapenade_3.16",
            "./outputs/routine.f90",
            "calling",
            ["f"],
            ["x"],
            {"x": [0.1, 0.1, 0.1, 0.1, 0.1]},
            "Linf_error",
            {"verbose": True, "inline_operation_adjoints": False},
            "forward",
            schedule,
        )

    if max_error > MAX_ERROR or np.isnan(max_error):
        raise ValueError(
            f"Test failed, Linf_error = {max_error} for schedule {type(schedule).__name__}"
        )
    print("passed")
    print("-----------------------")

@pytest.mark.parametrize("mode, vector",
                         product(("forward", "reverse"),
                                 (False, True)))
def test_many_arguments(mode, vector):
    """Test applying `psyclone.autodiff` and Tapenade transformations to a \
    subroutine with arguments of all possible intents (in, out, inout, undefined) \
    and comparing numerical results.

    :raises ValueError: if the error is above MAX_ERROR or is NaN.
    """
    print(f"Testing a routine with many arguments  in {mode}-mode")

    routine = SubroutineGenerator("routine_many")

    in_args = []
    inout_args = []
    out_args = []
    undef_args = []
    non_args = []

    n = 4

    for i in range(n * 5):
        if i < n*1:
            arg = routine.new_in_arg(f"arg{i}", _datatype(vector))
            in_args.append(arg)
        elif i < n*2:
            arg = routine.new_out_arg(f"arg{i}", _datatype(vector))
            out_args.append(arg)
        elif i < n*3:
            arg = routine.new_inout_arg(f"arg{i}", _datatype(vector))
            inout_args.append(arg)
        elif i < n*4:
            arg = routine.new_arg(f"arg{i}", _datatype(vector))
            undef_args.append(arg)
        else:
            arg = routine.new_variable(f"arg{i}")
            routine.new_assignment(arg, Literal(f"{i}.0", REAL_TYPE))
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

    with open("./outputs/routine.f90", "w") as file:
        file.write(routine.write())

    rev_schedule = ADJointReversalSchedule()

    independent_names = []
    dependent_names = []

    for arg in in_args + inout_args + undef_args:
        independent_names.append(arg.name)

    for arg in out_args + inout_args + undef_args:
        dependent_names.append(arg.name)

    values = {}
    for i, arg in enumerate(in_args + inout_args + undef_args):
        values[arg.name] = [float(i)]*5

    max_error, associated_values = NumericalComparator.compare(
        "./tapenade_3.16",
        "./outputs/routine.f90",
        "routine_many",
        dependent_names,
        independent_names,
        values,
        "Linf_error",
        {"verbose": True, "inline_operation_adjoints": False},
        mode,
        rev_schedule,
    )

    if max_error > MAX_ERROR or np.isnan(max_error):
        raise ValueError(f"Test failed, Linf_error = {max_error}")
    print("passed")
    print("-----------------------")

    print("===============================\n")

if __name__ == "__main__":
    pytest.main(['numerical_test.py'])
    #test_unary()
    #test_binary()
    #test_unary_composition()
    #test_binary_composition()
    #test_taping()
    #test_nested_calls()
    #test_many_arguments()
    # test_vars()
