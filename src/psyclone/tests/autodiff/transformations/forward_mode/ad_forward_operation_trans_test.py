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
    ArrayType
)
from psyclone.psyir.nodes import (
    Literal,
    UnaryOperation,
    BinaryOperation,
    IntrinsicCall,
    Reference,
    Container,
    Assignment
)
from psyclone.psyir.transformations import TransformationError
from psyclone.autodiff.transformations import (
    ADForwardOperationTrans,
    ADForwardRoutineTrans,
    ADForwardContainerTrans,
)
from psyclone.autodiff import one

PRE = ADForwardRoutineTrans._differential_prefix
POST = ADForwardRoutineTrans._differential_postfix


def compare(nodes, strings, fortran_writer):
    assert len(nodes) == len(strings)
    for node, expected_line in zip(nodes, strings):
        line = fortran_writer(node)
        assert line == expected_line


def initialize_transformations(options=None):
    freader = FortranReader()

    src = """subroutine foo()
    end subroutine foo"""
    psy = freader.psyir_from_source(src)
    container = psy.walk(Container)[0]

    ad_container_trans = ADForwardContainerTrans()
    ad_container_trans.apply(container, "foo", [], [], options)
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
        UnaryOperation.Operator.PLUS, Literal("1", INTEGER_TYPE)
    )

    with pytest.raises(TransformationError) as info:
        ad_operation_trans.validate(None)
    assert (
        "'operation' argument should be a "
        "PSyIR 'Operation' or 'IntrinsicCall' but found 'NoneType'." in str(info.value)
    )

    # Should pass
    ad_operation_trans.validate(unary_op)


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

    assert ad_operation_trans.differentiate(
        unary_op
    ) == ad_operation_trans.differentiate_unary(unary_op)
    assert ad_operation_trans.differentiate(
        binary_op
    ) == ad_operation_trans.differentiate_binary(binary_op)


def test_ad_operation_trans_differentiate_unary(fortran_writer):
    # Without activity analysis
    options = {"activity_analysis": False}
    _, ad_routine_trans, ad_operation_trans = initialize_transformations(options)

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate_unary(None, options)
    assert (
        "Argument in differentiate_unary should be a "
        "PSyIR UnaryOperation but found 'NoneType'." in str(info.value)
    )

    sym = DataSymbol("var", REAL_TYPE)
    ref = Reference(sym)
    ad_routine_trans.create_differential_symbol(sym)

    plus = UnaryOperation.create(UnaryOperation.Operator.PLUS, ref.copy())
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, ref.copy())

    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(plus, options))
        == f"{PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_unary(minus, options))
        == f"-{PRE}var{POST}"
    )

    not_ = UnaryOperation.create(UnaryOperation.Operator.NOT, ref)
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_unary(not_, options)
    assert (
        "Differentiating UnaryOperation with "
        "operator 'Operator.NOT' is not implemented yet." in str(info.value)
    )

    # TODO: with activity analysis


def test_ad_operation_trans_differentiate_binary(fortran_writer):
    # Witout activity analysis
    options = {"activity_analysis": False}
    _, ad_routine_trans, ad_operation_trans = initialize_transformations(options)

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
    vec1_sym = DataSymbol("vec1", ArrayType(REAL_TYPE, [3]))
    vec1 = Reference(vec1_sym)
    ad_routine_trans.create_differential_symbol(vec1_sym)
    vec2_sym = DataSymbol("vec2", ArrayType(REAL_TYPE, [3]))
    vec2 = Reference(vec2_sym)
    ad_routine_trans.create_differential_symbol(vec2_sym)
    mat1_sym = DataSymbol("mat1", ArrayType(REAL_TYPE, [3, 3]))
    mat1 = Reference(mat1_sym)
    ad_routine_trans.create_differential_symbol(mat1_sym)
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
        f"{PRE}var1{POST} * (1.35 * var1 ** 0.35) + 0 * (var1 ** 1.35 * LOG(var1))"
    )

    transformed = [ad_operation_trans.differentiate_binary(op, options) for op in ops]

    compare(transformed, expected, fortran_writer)

    eq = BinaryOperation.create(BinaryOperation.Operator.EQ, ref1, ref2)
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_binary(eq)
    assert (
        "Differentiating BinaryOperation with "
        "operator 'Operator.EQ' is not implemented yet." in str(info.value)
    )

    # TODO: with activy analysis

def test_ad_operation_trans_differentiate_intrinsic(fortran_writer):
    # Without activity analysis
    options = {"activity_analysis": False}
    _, ad_routine_trans, ad_operation_trans = initialize_transformations(options)

    with pytest.raises(TypeError) as info:
        ad_operation_trans.differentiate_intrinsic(None, options)
    assert (
        "Argument in differentiate_intrinsic should be a "
        "PSyIR IntrinsicCall but found 'NoneType'." in str(info.value)
    )

    sym = DataSymbol("var", REAL_TYPE)
    ref = Reference(sym)
    ad_routine_trans.create_differential_symbol(sym)

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

    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(sqrt, options))
        == f"{PRE}var{POST} / (2 * SQRT(var))"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(exp, options))
        == f"EXP(var) * {PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(log, options))
        == f"{PRE}var{POST} / var"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(log10, options))
        == f"{PRE}var{POST} / (var * LOG(10.0))"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(cos, options))
        == f"-SIN(var) * {PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(sin, options))
        == f"COS(var) * {PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(tan, options))
        == f"(1.0 + TAN(var) ** 2) * {PRE}var{POST}"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(acos, options))
        == f"-{PRE}var{POST} / SQRT(1.0 - var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(asin, options))
        == f"{PRE}var{POST} / SQRT(1.0 - var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(atan, options))
        == f"{PRE}var{POST} / (1.0 + var ** 2)"
    )
    assert (
        fortran_writer(ad_operation_trans.differentiate_intrinsic(abs_val, options))
        == f"var / ABS(var) * {PRE}var{POST}"
    )

    ceil = IntrinsicCall.create(IntrinsicCall.Intrinsic.CEILING, [ref])
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_intrinsic(ceil, options)
    assert (
        "Differentiating unary IntrinsicCall with "
        "intrinsic 'CEILING' is not implemented yet." in str(info.value)
    )

    sym1 = DataSymbol("var1", REAL_TYPE)
    ref1 = Reference(sym1)
    ad_routine_trans.create_differential_symbol(sym1)
    sym2 = DataSymbol("var2", REAL_TYPE)
    ref2 = Reference(sym2)
    ad_routine_trans.create_differential_symbol(sym2)
    vec1_sym = DataSymbol("vec1", ArrayType(REAL_TYPE, [3]))
    vec1 = Reference(vec1_sym)
    ad_routine_trans.create_differential_symbol(vec1_sym)
    vec2_sym = DataSymbol("vec2", ArrayType(REAL_TYPE, [3]))
    vec2 = Reference(vec2_sym)
    ad_routine_trans.create_differential_symbol(vec2_sym)
    mat1_sym = DataSymbol("mat1", ArrayType(REAL_TYPE, [3, 3]))
    mat1 = Reference(mat1_sym)
    ad_routine_trans.create_differential_symbol(mat1_sym)

    dot_product = IntrinsicCall.create(IntrinsicCall.Intrinsic.DOT_PRODUCT,
                                       [vec1.copy(), vec2.copy()])
    matmul = IntrinsicCall.create(IntrinsicCall.Intrinsic.MATMUL,
                                  [mat1.copy(), vec1.copy()])

    ops = (dot_product, matmul)
    expected = (
        # NOTE: 'call ' and '\n' aren't actually printed in practice by the
        # backend when the SUM is in a Schedule
        f"call SUM(vec2 * {PRE}vec1{POST} + vec1 * {PRE}vec2{POST})\n",
        f"MATMUL({PRE}mat1{POST}, vec1) + MATMUL(mat1, {PRE}vec1{POST})"
    )

    transformed = [ad_operation_trans.differentiate_intrinsic(op, options) for op in ops]

    compare(transformed, expected, fortran_writer)

    eq = IntrinsicCall.create(IntrinsicCall.Intrinsic.DPROD, [ref1, ref2])
    with pytest.raises(NotImplementedError) as info:
        ad_operation_trans.differentiate_intrinsic(eq)
    assert (
        "Differentiating binary IntrinsicCall with "
        "intrinsic 'DPROD' is not implemented yet." in str(info.value)
    )

    # TODO: with activity analysis

def test_ad_operation_trans_apply(fortran_writer):
    def initialize(options=None):
        (
            _,
            ad_routine_trans,
            ad_operation_trans,
        ) = initialize_transformations(options)

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

    # Without activity analysis
    options = {"activity_analysis": False}

    ad_operation_trans, sym, sym2, sym3, d_sym, d_sym2, d_sym3 = initialize(options)

    ##########
    # Literals
    unary = UnaryOperation.create(UnaryOperation.Operator.MINUS, one())
    transformed = ad_operation_trans.apply(unary, options)
    assert isinstance(transformed, Literal)
    assert fortran_writer(transformed) == "0"

    binary = BinaryOperation.create(BinaryOperation.Operator.ADD, one(), one())
    transformed = ad_operation_trans.apply(binary, options)
    assert isinstance(transformed, BinaryOperation)
    assert fortran_writer(transformed) == "0 + 0"

    ############
    # References
    unary = UnaryOperation.create(UnaryOperation.Operator.MINUS, Reference(sym2))
    transformed = ad_operation_trans.apply(unary, options)
    assert isinstance(transformed, UnaryOperation)
    assert fortran_writer(transformed) == f"-{PRE}var2{POST}"

    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, Reference(sym2), Reference(sym3)
    )
    transformed = ad_operation_trans.apply(binary, options)
    assert isinstance(transformed, BinaryOperation)
    assert fortran_writer(transformed) == f"{PRE}var2{POST} - {PRE}var3{POST}"

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), Reference(sym3)
    )
    transformed = ad_operation_trans.apply(binary, options)
    assert isinstance(transformed, BinaryOperation)
    assert (
        fortran_writer(transformed)
        == f"{PRE}var2{POST} * var3 + {PRE}var3{POST} * var2"
    )

    ###########
    # Operations

    ad_operation_trans, sym, sym2, sym3, d_sym, d_sym2, d_sym3 = initialize(options)
    # create sym2 * -sym3
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, Reference(sym3))
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), minus
    )
    transformed = ad_operation_trans.apply(binary, options)
    assert isinstance(transformed, BinaryOperation)
    assert (
        fortran_writer(transformed)
        == f"{PRE}var2{POST} * (-var3) + (-{PRE}var3{POST}) * var2"
    )

    ad_operation_trans, sym, sym2, sym3, d_sym, d_sym2, d_sym3 = initialize(options)
    # create sym2 * -(sym3*sym2)
    bin1 = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym3), Reference(sym2)
    )
    minus = UnaryOperation.create(UnaryOperation.Operator.MINUS, bin1)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, Reference(sym2), minus
    )
    transformed = ad_operation_trans.apply(binary, options)
    assert isinstance(transformed, BinaryOperation)
    assert (
        fortran_writer(transformed)
        == f"{PRE}var2{POST} * (-var3 * var2) + (-({PRE}var3{POST} * var2 + {PRE}var2{POST} * var3)) * var2"
    )

    # TODO: with activity analysis
