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

"""A module to perform tests on the autodiff ADForwardOperationTrans class.
"""

import pytest

from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.symbols import (
    DataSymbol,
    INTEGER_TYPE,
    REAL_TYPE,
)
from psyclone.psyir.nodes import (
    Literal,
    UnaryOperation,
    BinaryOperation,
    NaryOperation,
    Reference,
    Container,
)
from psyclone.psyir.transformations import TransformationError
from psyclone.autodiff.transformations import (
    ADForwardOperationTrans,
    ADForwardRoutineTrans,
    ADForwardContainerTrans,
)
from psyclone.autodiff import one, ADSplitReversalSchedule

PRE = ADForwardRoutineTrans._differential_prefix
POST = ADForwardRoutineTrans._differential_postfix


def compare(nodes, strings, fortran_writer):
    assert len(nodes) == len(strings)
    for node, expected_line in zip(nodes, strings):
        line = fortran_writer(node)
        assert line == expected_line


def initialize_transformations():
    freader = FortranReader()

    src = """subroutine foo()
    end subroutine foo"""
    psy = freader.psyir_from_source(src)
    container = psy.walk(Container)[0]

    ad_container_trans = ADForwardContainerTrans()
    ad_container_trans.apply(container, "foo", [], [])
    ad_routine_trans = ad_container_trans.routine_transformations[0]

    return ad_container_trans, ad_routine_trans, ad_routine_trans.operation_trans


def test_ad_operation_trans_initialization():
    with pytest.raises(TypeError) as info:
        ADForwardOperationTrans(None)
    assert "Argument should be of type 'ADRoutineTrans' but found 'NoneType'." in str(
        info.value
    )

    _, ad_routine_trans, ad_operation_trans = initialize_transformations()

    assert ad_operation_trans.routine_trans == ad_routine_trans


def test_ad_operation_trans_validate():
    _, _, ad_operation_trans = initialize_transformations()

    unary_op = UnaryOperation.create(
        UnaryOperation.Operator.EXP, Literal("1", INTEGER_TYPE)
    )

    with pytest.raises(TransformationError) as info:
        ad_operation_trans.validate(None)
    assert (
        "'operation' argument should be a "
        "PSyIR 'Operation' but found 'NoneType'." in str(info.value)
    )

    # Should pass
    ad_operation_trans.validate(unary_op)


def test_ad_operation_trans_differentiate():
    _, _, ad_operation_trans = initialize_transformations()

    unary_op = UnaryOperation.create(UnaryOperation.Operator.EXP, one())
    binary_op = BinaryOperation.create(BinaryOperation.Operator.ADD, one(), one())
    nary_op = NaryOperation.create(NaryOperation.Operator.MAX, [one(), one(), one()])

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate(None)
    assert (
        "Argument in differentiate should be a "
        "PSyIR 'Operation' but found 'NoneType'." in str(info.value)
    )

    assert ad_operation_trans.differentiate(
        unary_op
    ) == ad_operation_trans.differentiate_unary(unary_op)
    assert ad_operation_trans.differentiate(
        binary_op
    ) == ad_operation_trans.differentiate_binary(binary_op)

    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate(nary_op)
    assert "Differentiating NaryOperation nodes " "isn't implement yet." in str(
        info.value
    )


def test_ad_operation_trans_differentiate_unary(fortran_writer):
    _, ad_routine_trans, ad_operation_trans = initialize_transformations()

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate_unary(None)
    assert (
        "Argument in differentiate_unary should be a "
        "PSyIR UnaryOperation but found 'NoneType'." in str(info.value)
    )

    sym = DataSymbol("var", REAL_TYPE)
    ref = Reference(sym)
    ad_routine_trans.create_differential_symbol(sym)

    plus = UnaryOperation.create(UnaryOperation.Operator.PLUS, ref.copy())
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, ref.copy())
    sqrt = UnaryOperation.create(UnaryOperation.Operator.SQRT, ref.copy())
    exp = UnaryOperation.create(UnaryOperation.Operator.EXP, ref.copy())
    log = UnaryOperation.create(UnaryOperation.Operator.LOG, ref.copy())
    log10 = UnaryOperation.create(UnaryOperation.Operator.LOG10, ref.copy())
    cos = UnaryOperation.create(UnaryOperation.Operator.COS, ref.copy())
    sin = UnaryOperation.create(UnaryOperation.Operator.SIN, ref.copy())
    tan = UnaryOperation.create(UnaryOperation.Operator.TAN, ref.copy())
    acos = UnaryOperation.create(UnaryOperation.Operator.ACOS, ref.copy())
    asin = UnaryOperation.create(UnaryOperation.Operator.ASIN, ref.copy())
    atan = UnaryOperation.create(UnaryOperation.Operator.ATAN, ref.copy())
    abs_val = UnaryOperation.create(UnaryOperation.Operator.ABS, ref.copy())

    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(plus))
        == f"{PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(minus))
        == f"-{PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(sqrt))
        == f"{PRE}var{POST} / (2 * SQRT(var))"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(exp))
        == f"EXP(var) * {PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(log))
        == f"{PRE}var{POST} / var"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(log10))
        == f"{PRE}var{POST} / (var * LOG(10.0))"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(cos))
        == f"-SIN(var) * {PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(sin))
        == f"COS(var) * {PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(tan))
        == f"(1.0 + TAN(var) ** 2) * {PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(acos))
        == f"-{PRE}var{POST} / SQRT(1.0 - var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(asin))
        == f"{PRE}var{POST} / SQRT(1.0 - var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(atan))
        == f"{PRE}var{POST} / (1.0 + var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(abs_val))
        == f"var / ABS(var) * {PRE}var{POST}"
    )

    ceil = UnaryOperation.create(UnaryOperation.Operator.CEIL, ref)
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_unary(ceil)
    assert (
        "Differentiating UnaryOperation with "
        "operator 'Operator.CEIL' is not implemented yet." in str(info.value)
    )


def test_ad_operation_trans_differentiate_binary(fortran_writer):
    # ad_container_trans = ADForwardContainerTrans()
    # ad_routine_trans = ADForwardRoutineTrans(ad_container_trans)
    _, ad_routine_trans, ad_operation_trans = initialize_transformations()

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate_binary(None)
    assert (
        "Argument in differentiate_binary should be a "
        "PSyIR BinaryOperation but found 'NoneType'." in str(info.value)
    )

    sym1 = DataSymbol("var1", REAL_TYPE)
    ref1 = Reference(sym1)
    ad_routine_trans.create_differential_symbol(sym1)
    sym2 = DataSymbol("var2", REAL_TYPE)
    ref2 = Reference(sym2)
    ad_routine_trans.create_differential_symbol(sym2)
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

    ops = (add, sub, mul, div, power, power_literal)
    expected = (
        f"{PRE}var1{POST} + {PRE}var2{POST}",
        f"{PRE}var1{POST} - {PRE}var2{POST}",
        f"{PRE}var1{POST} * var2 + {PRE}var2{POST} * var1",
        f"({PRE}var1{POST} - {PRE}var2{POST} * var1 / var2) / var2",
        f"{PRE}var1{POST} * (var2 * var1 ** (var2 - 1)) + {PRE}var2{POST} * (var1 ** var2 * LOG(var1))",
        f"{PRE}var1{POST} * (1.35 * var1 ** 0.35) + 0 * (var1 ** 1.35 * LOG(var1))",
    )

    transformed = [ad_operation_trans.differentiate_binary(op) for op in ops]

    compare(transformed, expected, fortran_writer)

    eq = BinaryOperation.create(BinaryOperation.Operator.EQ, ref1, ref2)
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_binary(eq)
    assert (
        "Differentiating BinaryOperation with "
        "operator 'Operator.EQ' is not implemented yet." in str(info.value)
    )


def test_ad_operation_trans_apply(fortran_writer):
    def initialize():
        (
            _,
            ad_routine_trans,
            ad_operation_trans,
        ) = initialize_transformations()

        sym = DataSymbol("var", REAL_TYPE)
        d_sym = ad_routine_trans.create_differential_symbol(sym)
        assert d_sym.name == f"{PRE}var{POST}"

        sym2 = DataSymbol("var2", REAL_TYPE)
        d_sym2 = ad_routine_trans.create_differential_symbol(sym2)
        assert d_sym2.name == f"{PRE}var2{POST}"

        sym3 = DataSymbol("var3", REAL_TYPE)
        d_sym3 = ad_routine_trans.create_differential_symbol(sym3)
        assert d_sym3.name == f"{PRE}var3{POST}"

        return ad_operation_trans, sym, sym2, sym3, d_sym, d_sym2, d_sym3

    ad_operation_trans, sym, sym2, sym3, d_sym, d_sym2, d_sym3 = initialize()

    ##########
    # Literals
    unary = UnaryOperation.create(UnaryOperation.Operator.MINUS, one())
    transformed = ad_operation_trans.apply(unary)
    assert isinstance(transformed, Literal)
    assert fortran_writer(transformed) == "0"

    binary = BinaryOperation.create(BinaryOperation.Operator.ADD, one(), one())
    transformed = ad_operation_trans.apply(binary)
    assert isinstance(transformed, BinaryOperation)
    assert fortran_writer(transformed) == "0 + 0"

    ############
    # References
    unary = UnaryOperation.create(UnaryOperation.Operator.MINUS, Reference(sym2))
    transformed = ad_operation_trans.apply(unary)
    assert isinstance(transformed, UnaryOperation)
    assert fortran_writer(transformed) == f"-{PRE}var2{POST}"

    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, Reference(sym2), Reference(sym3)
    )
    transformed = ad_operation_trans.apply(binary)
    assert isinstance(transformed, BinaryOperation)
    assert fortran_writer(transformed) == f"{PRE}var2{POST} - {PRE}var3{POST}"

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), Reference(sym3)
    )
    transformed = ad_operation_trans.apply(binary)
    assert isinstance(transformed, BinaryOperation)
    assert (
        fortran_writer(transformed)
        == f"{PRE}var2{POST} * var3 + {PRE}var3{POST} * var2"
    )

    ###########
    # Operations

    ad_operation_trans, sym, sym2, sym3, d_sym, d_sym2, d_sym3 = initialize()
    # create sym2 * -sym3
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, Reference(sym3))
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), minus
    )
    transformed = ad_operation_trans.apply(binary)
    assert isinstance(transformed, BinaryOperation)
    assert (
        fortran_writer(transformed)
        == f"{PRE}var2{POST} * (-var3) + (-{PRE}var3{POST}) * var2"
    )

    ad_operation_trans, sym, sym2, sym3, d_sym, d_sym2, d_sym3 = initialize()
    # create sym2 * -(sym3*sym2)
    bin1 = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym3), Reference(sym2)
    )
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, bin1)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), minus
    )
    transformed = ad_operation_trans.apply(binary)
    assert isinstance(transformed, BinaryOperation)
    assert (
        fortran_writer(transformed)
        == f"{PRE}var2{POST} * (-var3 * var2) + (-({PRE}var3{POST} * var2 + {PRE}var2{POST} * var3)) * var2"
    )
