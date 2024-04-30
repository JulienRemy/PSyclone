# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2021-2022, Science and Technology Facilities Council.
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

"""A module to perform tests on the autodiff ADReverseOperationTrans class.
"""

import pytest

from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.symbols import (
    DataSymbol,
    INTEGER_TYPE,
    REAL_TYPE,
    ArrayType
)
from psyclone.psyir.nodes import (
    Literal,
    UnaryOperation,
    BinaryOperation,
    IntrinsicCall,
    Reference,
    Container,
)
from psyclone.psyir.transformations import TransformationError
from psyclone.autodiff.transformations import (
    ADReverseOperationTrans,
    ADReverseRoutineTrans,
    ADReverseContainerTrans,
)
from psyclone.autodiff import one, ADSplitReversalSchedule

AP = ADReverseRoutineTrans._differential_prefix
AS = ADReverseRoutineTrans._differential_postfix
OA = ADReverseRoutineTrans._operation_adjoint_name


def compare(nodes, strings, fortran_writer):
    assert len(nodes) == len(strings)
    for node, expected_line in zip(nodes, strings):
        line = fortran_writer(node)
        assert line == expected_line


def initialize_transformations(options=None):
    freader = FortranReader()
    reversal_schedule = ADSplitReversalSchedule()

    src = """subroutine foo()
    end subroutine foo"""
    psy = freader.psyir_from_source(src)
    container = psy.walk(Container)[0]

    ad_container_trans = ADReverseContainerTrans()
    ad_container_trans.apply(container, "foo", [], [], reversal_schedule, options)
    ad_routine_trans = ad_container_trans.routine_transformations[0]

    return ad_container_trans, ad_routine_trans, ad_routine_trans.operation_trans


def test_ad_operation_trans_initialization():
    with pytest.raises(TypeError) as info:
        ADReverseOperationTrans(None)
    assert "Argument should be of type 'ADRoutineTrans' but found 'NoneType'." in str(
        info.value
    )

    _, ad_routine_trans, ad_operation_trans = initialize_transformations()

    assert ad_operation_trans.routine_trans == ad_routine_trans


def test_ad_operation_trans_validate():
    _, ad_routine_trans, ad_operation_trans = initialize_transformations()

    unary_op = UnaryOperation.create(
        UnaryOperation.Operator.PLUS, Literal("1", INTEGER_TYPE)
    )

    with pytest.raises(TransformationError) as info:
        ad_operation_trans.validate(None, None)
    assert (
        "'operation' argument should be a "
        "PSyIR 'Operation' or 'IntrinsicCall' but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(TransformationError) as info:
        ad_operation_trans.validate(unary_op, None)
    assert (
        "'parent_adj' argument should be a "
        "PSyIR 'DataNode' but found 'NoneType'." in str(info.value)
    )

    sym = DataSymbol("var", REAL_TYPE)
    with pytest.raises(TransformationError) as info:
        ad_operation_trans.validate(unary_op, Reference(sym))
    assert (
        "'parent_adj.symbol' DataSymbol "
        "'var' cannot be found "
        "among the existing adjoint symbols." in str(info.value)
    )

    adj_sym = ad_routine_trans.create_differential_symbol(sym)
    unary_op = UnaryOperation.create(
        UnaryOperation.Operator.PLUS, Literal("1", INTEGER_TYPE)
    )
    ad_operation_trans.validate(unary_op, Reference(adj_sym))


def test_ad_operation_trans_differentiate():
    _, _, ad_operation_trans = initialize_transformations()

    unary_op = UnaryOperation.create(UnaryOperation.Operator.PLUS, one())
    binary_op = BinaryOperation.create(BinaryOperation.Operator.ADD, one(), one())

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate(None)
    assert (
        "Argument in differentiate should be a "
        "PSyIR 'Operation' or 'IntrinsicCall' but found 'NoneType'." in str(info.value)
    )

    assert ad_operation_trans.differentiate(unary_op) == [
        ad_operation_trans.differentiate_unary(unary_op)
    ]
    assert ad_operation_trans.differentiate(
        binary_op
    ) == ad_operation_trans.differentiate_binary(binary_op)


def test_ad_operation_trans_differentiate_unary(fortran_writer):
    _, _, ad_operation_trans = initialize_transformations()

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate_unary(None)
    assert (
        "Argument in differentiate_unary should be a "
        "PSyIR UnaryOperation but found 'NoneType'." in str(info.value)
    )

    ref = Reference(DataSymbol("var", REAL_TYPE))
    plus = UnaryOperation.create(UnaryOperation.Operator.PLUS, ref.copy())
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, ref.copy())

    assert fortran_writer(ad_operation_trans.differentiate_unary(plus)) == "1.0"
    assert fortran_writer(ad_operation_trans.differentiate_unary(minus)) == "-1.0"

    not_ = UnaryOperation.create(UnaryOperation.Operator.NOT, ref)
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_unary(not_)
    assert (
        "Differentiating UnaryOperation with "
        "operator 'Operator.NOT' is not implemented yet." in str(info.value)
    )


def test_ad_operation_trans_differentiate_binary(fortran_writer):
    # ad_container_trans = ADReverseContainerTrans()
    # ad_routine_trans = ADReverseRoutineTrans(ad_container_trans)
    _, _, ad_operation_trans = initialize_transformations()

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate_binary(None)
    assert (
        "Argument in differentiate_binary should be a "
        "PSyIR BinaryOperation but found 'NoneType'." in str(info.value)
    )

    ref1 = Reference(DataSymbol("var1", REAL_TYPE))
    ref2 = Reference(DataSymbol("var2", REAL_TYPE))
    vec1_sym = DataSymbol("vec1", ArrayType(REAL_TYPE, [3]))
    vec1 = Reference(vec1_sym)
    vec2_sym = DataSymbol("vec2", ArrayType(REAL_TYPE, [3]))
    vec2 = Reference(vec2_sym)
    mat1_sym = DataSymbol("mat1", ArrayType(REAL_TYPE, [3, 3]))
    mat1 = Reference(mat1_sym)
    mat2_sym = DataSymbol("mat2", ArrayType(REAL_TYPE, [3, 3]))
    mat2 = Reference(mat2_sym)
    add = BinaryOperation.create(BinaryOperation.Operator.ADD, ref1.copy(), ref2.copy())
    sub = BinaryOperation.create(BinaryOperation.Operator.SUB, ref1.copy(), ref2.copy())
    mul = BinaryOperation.create(BinaryOperation.Operator.MUL, ref1.copy(), ref2.copy())
    div = BinaryOperation.create(BinaryOperation.Operator.DIV, ref1.copy(), ref2.copy())
    power = BinaryOperation.create(
        BinaryOperation.Operator.POW, ref1.copy(), ref2.copy()
    )
    power_literal = BinaryOperation.create(
        BinaryOperation.Operator.POW, ref1.copy(), Literal("1.35", REAL_TYPE)
    )

    # TODO: implement and test MATMUL
    ops = (add, sub, mul, div, power, power_literal)
    expected = (
        ("1.0", "1.0"),
        ("1.0", "-1.0"),
        ("var2", "var1"),
        ("1.0 / var2", "-var1 / var2 ** 2"),
        ("var2 * var1 ** (var2 - 1)", "var1 ** var2 * LOG(var1)"),
        ("1.35 * var1 ** 0.35", "var1 ** 1.35 * LOG(var1)")
    )

    for operation, exp in zip(ops, expected):
        partials = ad_operation_trans.differentiate_binary(operation)
        assert len(partials) == 2
        compare(partials, exp, fortran_writer)

    eq = BinaryOperation.create(BinaryOperation.Operator.EQ, ref1, ref2)
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_binary(eq)
    assert (
        "Differentiating BinaryOperation with "
        "operator 'Operator.EQ' is not implemented yet." in str(info.value)
    )

def test_ad_operation_trans_differentiate_intrinsic(fortran_writer):
    _, _, ad_operation_trans = initialize_transformations()

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate_intrinsic(None)
    assert (
        "Argument in differentiate_intrinsic should be a "
        "PSyIR IntrinsicCall but found 'NoneType'." in str(info.value)
    )

    ref = Reference(DataSymbol("var", REAL_TYPE))
    sqrt = IntrinsicCall.create(IntrinsicCall.Intrinsic.SQRT, [ref.copy()])
    exp = IntrinsicCall.create(IntrinsicCall.Intrinsic.EXP, [ref.copy()])
    log = IntrinsicCall.create(IntrinsicCall.Intrinsic.LOG, [ref.copy()])
    log10 = IntrinsicCall.create(IntrinsicCall.Intrinsic.LOG10, [ref.copy()])
    cos = IntrinsicCall.create(IntrinsicCall.Intrinsic.COS, [ref.copy()])
    sin = IntrinsicCall.create(IntrinsicCall.Intrinsic.SIN, [ref.copy()])
    tan = IntrinsicCall.create(IntrinsicCall.Intrinsic.TAN, [ref.copy()])
    acos = IntrinsicCall.create(IntrinsicCall.Intrinsic.ACOS, [ref.copy()])
    asin = IntrinsicCall.create(IntrinsicCall.Intrinsic.ASIN, [ref.copy()])
    atan = IntrinsicCall.create(IntrinsicCall.Intrinsic.ATAN, [ref.copy()])
    abs_val = IntrinsicCall.create(IntrinsicCall.Intrinsic.ABS, [ref.copy()])

    # NOTE: the 'call ' and '\n' parts in the compared strings aren't actually
    # printed by the backend when the nodes have a parent

    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(sqrt)[0])
        == "1.0 / (2 * SQRT(var))"
    )
    assert fortran_writer(ad_operation_trans.differentiate_intrinsic(exp)[0]) == "call EXP(var)\n"
    assert fortran_writer(ad_operation_trans.differentiate_intrinsic(log)[0]) == "1.0 / var"
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(log10)[0])
        == "1.0 / (var * LOG(10.0))"
    )
    assert fortran_writer(ad_operation_trans.differentiate_intrinsic(cos)[0]) == "-SIN(var)"
    assert fortran_writer(ad_operation_trans.differentiate_intrinsic(sin)[0]) == "call COS(var)\n"
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(tan)[0])
        == "1.0 + TAN(var) ** 2"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(acos)[0])
        == "-1.0 / SQRT(1.0 - var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(asin)[0])
        == "1.0 / SQRT(1.0 - var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(atan)[0])
        == "1.0 / (1.0 + var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(abs_val)[0])
        == "var / ABS(var)"
    )

    ceil = IntrinsicCall.create(IntrinsicCall.Intrinsic.CEILING, [ref])
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_intrinsic(ceil)[0]
    assert (
        "Differentiating unary IntrinsicCall with "
        "intrinsic 'CEILING' is not implemented yet." in str(info.value)
    )

    ref1 = Reference(DataSymbol("var1", REAL_TYPE))
    ref2 = Reference(DataSymbol("var2", REAL_TYPE))
    vec1_sym = DataSymbol("vec1", ArrayType(REAL_TYPE, [3]))
    vec1 = Reference(vec1_sym)
    vec2_sym = DataSymbol("vec2", ArrayType(REAL_TYPE, [3]))
    vec2 = Reference(vec2_sym)
    mat1_sym = DataSymbol("mat1", ArrayType(REAL_TYPE, [3, 3]))
    mat1 = Reference(mat1_sym)
    mat2_sym = DataSymbol("mat2", ArrayType(REAL_TYPE, [3, 3]))
    mat2 = Reference(mat2_sym)

    dot_product = IntrinsicCall.create(IntrinsicCall.Intrinsic.DOT_PRODUCT,
                                       [vec1.copy(), vec2.copy()])
    matmul_mat_vec = IntrinsicCall.create(IntrinsicCall.Intrinsic.MATMUL,
                                          [mat1.copy(), vec1.copy()])
    matmul_mat_mat = IntrinsicCall.create(IntrinsicCall.Intrinsic.MATMUL,
                                          [mat1.copy(), mat2.copy()])

    # TODO: implement and test MATMUL
    ops = (dot_product, )
    expected = (
        ("vec2", "vec1"),
    )

    for operation, exp in zip(ops, expected):
        partials = ad_operation_trans.differentiate_intrinsic(operation)
        assert len(partials) == 2
        compare(partials, exp, fortran_writer)

    eq = IntrinsicCall.create(IntrinsicCall.Intrinsic.DPROD, [ref1, ref2])
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_intrinsic(eq)
    assert (
        "Differentiating binary IntrinsicCall with "
        "intrinsic 'DPROD' is not implemented yet." in str(info.value)
    )

def test_ad_operation_trans_apply(fortran_writer):
    def initialize(options=None):
        (
            _,
            ad_routine_trans,
            ad_operation_trans,
        ) = initialize_transformations(options)

        sym = DataSymbol("var", REAL_TYPE)
        adj_sym = ad_routine_trans.create_differential_symbol(sym)
        assert adj_sym.name == f"{AP}var{AS}"

        sym2 = DataSymbol("var2", REAL_TYPE)
        adj_sym2 = ad_routine_trans.create_differential_symbol(sym2)
        assert adj_sym2.name == f"{AP}var2{AS}"

        sym3 = DataSymbol("var3", REAL_TYPE)
        adj_sym3 = ad_routine_trans.create_differential_symbol(sym3)
        assert adj_sym3.name == f"{AP}var3{AS}"

        return ad_operation_trans, sym, sym2, sym3, adj_sym, adj_sym2, adj_sym3

    # Without activity analysis
    options = {"activity_analysis": False}

    ad_operation_trans, sym, sym2, sym3, adj_sym, adj_sym2, adj_sym3 = initialize(options)

    ##########
    # Literals
    unary = UnaryOperation.create(UnaryOperation.Operator.MINUS, one())
    returning, assignment_lhs_incr = ad_operation_trans.apply(unary, Reference(adj_sym), options)
    assert len(returning) == 0
    assert len(assignment_lhs_incr) == 0

    binary = BinaryOperation.create(BinaryOperation.Operator.ADD, one(), one())
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym), options)
    assert len(returning) == 0
    assert len(assignment_lhs_incr) == 0

    ############
    # References, non-iterative
    unary = UnaryOperation.create(UnaryOperation.Operator.MINUS, Reference(sym2))
    returning, assignment_lhs_incr = ad_operation_trans.apply(unary, Reference(adj_sym), options)
    assert len(returning) == 1
    assert len(assignment_lhs_incr) == 0
    expected = f"{AP}var2{AS} = {AP}var2{AS} + {AP}var{AS} * (-1.0)\n"  # , "{AP}var{AS} = 0.0\n")
    assert fortran_writer(returning[0]) == expected

    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, Reference(sym2), Reference(sym3)
    )
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym), options)
    assert len(returning) == 2
    assert len(assignment_lhs_incr) == 0
    expected = (
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var{AS} * 1.0\n",
        f"{AP}var3{AS} = {AP}var3{AS} + {AP}var{AS} * (-1.0)\n",
    )
    compare(returning, expected, fortran_writer)

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), Reference(sym3)
    )
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym), options)
    assert len(returning) == 2
    assert len(assignment_lhs_incr) == 0
    expected = (
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var{AS} * var3\n",
        f"{AP}var3{AS} = {AP}var3{AS} + {AP}var{AS} * var2\n",
    )
    compare(returning, expected, fortran_writer)

    # Special case: var = var2 (op) var2
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), Reference(sym2)
    )
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym), options)
    assert len(returning) == 1
    assert len(assignment_lhs_incr) == 0
    expected = (
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var{AS} * (var2 + var2)\n",
        #f"{AP}var3{AS} = {AP}var3{AS} + {AP}var{AS} * var2\n",
    )
    compare(returning, expected, fortran_writer)

    ###########
    # Operations, non-iterative

    ad_operation_trans, sym, sym2, sym3, adj_sym, adj_sym2, adj_sym3 = initialize(options)
    # create sym2 * -sym3
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, Reference(sym3))
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), minus
    )
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym), options)
    assert len(returning) == 2
    assert len(assignment_lhs_incr) == 0
    expected = (
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var{AS} * (-var3)\n",
        # f"{OA} = {AP}var{AS} * var2\n",
        f"{AP}var3{AS} = {AP}var3{AS} + {AP}var{AS} * var2 * (-1.0)\n",
    )
    compare(returning, expected, fortran_writer)

    ad_operation_trans, sym, sym2, sym3, adj_sym, adj_sym2, adj_sym3 = initialize(options)
    # create sym2 * -(sym3*sym2)
    bin1 = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym3), Reference(sym2)
    )
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, bin1)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), minus
    )
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym), options)
    assert len(returning) == 3
    assert len(assignment_lhs_incr) == 0
    expected = (
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var{AS} * (-var3 * var2)\n",
        # f"{OA} = {AP}var{AS} * var2\n",
        # f"{OA}_1 = {AP}var{AS} * var2 * (-1.0)\n",
        f"{AP}var3{AS} = {AP}var3{AS} + {AP}var{AS} * var2 * (-1.0) * var2\n",
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var{AS} * var2 * (-1.0) * var3\n",
    )
    compare(returning, expected, fortran_writer)

    ##############
    # NOTE: the first element of assignment_lhs_incr is an incrementation
    # IT IS TRANSFORMED BY ADReverseAssignmentTrans to an assignment
    from psyclone.psyir.nodes import Assignment

    # References, iterative
    unary = UnaryOperation.create(UnaryOperation.Operator.MINUS, Reference(sym2))
    Assignment.create(
        Reference(sym2), unary
    )  # Attaches the operation as rhs of an assignment to var2
    returning, assignment_lhs_incr = ad_operation_trans.apply(unary, Reference(adj_sym2), options)
    assert len(returning) == 0
    assert len(assignment_lhs_incr) == 1
    expected = f"{AP}var2{AS} = {AP}var2{AS} + {AP}var2{AS} * (-1.0)\n"  # , "{AP}var{AS} = 0.0\n")
    assert fortran_writer(assignment_lhs_incr[0]) == expected

    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, Reference(sym2), Reference(sym3)
    )
    Assignment.create(
        Reference(sym2), binary
    )  # Attaches the operation as rhs of an assignment to var2
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym2), options)
    assert len(returning) == 1
    assert len(assignment_lhs_incr) == 1
    expected_ret = (f"{AP}var3{AS} = {AP}var3{AS} + {AP}var2{AS} * (-1.0)\n",)
    expected_lhs = (f"{AP}var2{AS} = {AP}var2{AS} + {AP}var2{AS} * 1.0\n",)
    compare(returning, expected_ret, fortran_writer)
    compare(assignment_lhs_incr, expected_lhs, fortran_writer)

    # binary, other way around
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, Reference(sym2), Reference(sym3)
    )
    Assignment.create(
        Reference(sym3), binary
    )  # Attaches the operation as rhs of an assignment to var2
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym2), options)
    assert len(returning) == 1
    assert len(assignment_lhs_incr) == 1
    expected_ret = (f"{AP}var2{AS} = {AP}var2{AS} + {AP}var2{AS} * 1.0\n",)
    expected_lhs = (f"{AP}var3{AS} = {AP}var3{AS} + {AP}var2{AS} * (-1.0)\n",)
    compare(returning, expected_ret, fortran_writer)
    compare(assignment_lhs_incr, expected_lhs, fortran_writer)

    # binary, both
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, Reference(sym2), Reference(sym2)
    )
    Assignment.create(
        Reference(sym2), binary
    )  # Attaches the operation as rhs of an assignment to var2
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym2), options)
    assert len(returning) == 0
    assert len(assignment_lhs_incr) == 1
    expected_lhs = (
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var2{AS} * (1.0 + (-1.0))\n",
        #f"{AP}var2{AS} = {AP}var2{AS} + {AP}var2{AS} * (-1)\n",
    )
    compare(assignment_lhs_incr, expected_lhs, fortran_writer)

    ###########
    # Operations, iterative

    ad_operation_trans, sym, sym2, sym3, adj_sym, adj_sym2, adj_sym3 = initialize(options)
    # create var2 = var2 * -var3
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, Reference(sym3))
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), minus
    )
    Assignment.create(
        Reference(sym2), binary
    )  # Attaches the operation as rhs of an assignment to var2
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym2), options)
    assert len(returning) == 1
    assert len(assignment_lhs_incr) == 1
    expected_ret = (
        #f"{OA} = {AP}var2{AS} * var2\n",
        f"{AP}var3{AS} = {AP}var3{AS} + {AP}var2{AS} * var2 * (-1.0)\n",
    )
    expected_lhs = (f"{AP}var2{AS} = {AP}var2{AS} + {AP}var2{AS} * (-var3)\n",)
    compare(returning, expected_ret, fortran_writer)
    compare(assignment_lhs_incr, expected_lhs, fortran_writer)

    ad_operation_trans, sym, sym2, sym3, adj_sym, adj_sym2, adj_sym3 = initialize(options)
    # create var2 = var2 + var3*var2
    bin1 = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym3), Reference(sym2)
    )
    binary = BinaryOperation.create(BinaryOperation.Operator.ADD, Reference(sym2), bin1)
    Assignment.create(
        Reference(sym2), binary
    )  # Attaches the operation as rhs of an assignment to var2
    returning, assignment_lhs_incr = ad_operation_trans.apply(binary, Reference(adj_sym2), options)
    assert len(returning) == 1
    assert len(assignment_lhs_incr) == 2
    expected_ret = (
        #f"{OA} = {AP}var2{AS} * 1.0\n",
        f"{AP}var3{AS} = {AP}var3{AS} + {AP}var2{AS} * 1.0 * var2\n",
    )
    expected_lhs = (
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var2{AS} * 1.0\n",
        f"{AP}var2{AS} = {AP}var2{AS} + {AP}var2{AS} * 1.0 * var3\n",
    )
    compare(returning, expected_ret, fortran_writer)
    compare(assignment_lhs_incr, expected_lhs, fortran_writer)
