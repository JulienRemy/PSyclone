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

"""A module to perform tests on the code in the
simplify.py file within the autodiff directory.

"""
import pytest

from psyclone.psyir.nodes import (
    UnaryOperation,
    BinaryOperation,
    Literal,
    Assignment,
    Reference,
)
from psyclone.psyir.symbols import INTEGER_TYPE, REAL_TYPE, DataSymbol
from psyclone.psyir.backend.fortran import FortranWriter

from psyclone.autodiff.simplify import (
    _typecheck_add,
    _typecheck_add_sub,
    _typecheck_assignment,
    _typecheck_at_least_one_literal_operand,
    _valuecheck_at_least_one_literal_operand,
    _typecheck_binary_operation,
    _typecheck_minus,
    _typecheck_mul_div,
    _typecheck_sub,
    _typecheck_unary_operation,
    #simplify_add,
    simplify_add_minus_minus,
    simplify_add_plus_minus_or_minus_plus,
    simplify_add_zero,
    simplify_add_sub_factorize,
    simplify_add_twice_to_mul_by_2,
    #simplify_assignment,
    simplify_binary_operation,
    #simplify_mul_div,
    simplify_mul_by_one,
    simplify_mul_by_zero,
    simplify_mul_div_minus_minus,
    simplify_mul_div_plus_minus_or_minus_plus,
    simplify_node,
    simplify_self_assignment,
    #simplify_sub,
    simplify_sub_itself_to_zero,
    simplify_sub_minus_plus,
    simplify_sub_plus_minus,
    simplify_sub_zero,
)

#pylint: disable=invalid-name, redefined-outer-name, too-many-lines

def test__typecheck_binary_operation():
    """Test that the _typecheck_binary_operation function raises the correct
    errors.
    """
    with pytest.raises(TypeError) as info:
        _typecheck_binary_operation(None)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'NoneType'." in str(info.value)
    )


def test__typecheck_add():
    """Test that the _typecheck_add function raises the correct errors."""
    with pytest.raises(TypeError) as info:
        _typecheck_add(None)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'NoneType'." in str(info.value)
    )
    with pytest.raises(ValueError) as info:
        _typecheck_add(BinaryOperation(BinaryOperation.Operator.MUL))
    assert (
        "'binary_operation' argument should have "
        "operator 'BinaryOperation.Operator.ADD' "
        "but found 'Operator.MUL'." in str(info.value)
    )


def test__typecheck_sub():
    """Test that the _typecheck_sub function raises the correct errors."""
    with pytest.raises(TypeError) as info:
        _typecheck_sub(None)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'NoneType'." in str(info.value)
    )
    with pytest.raises(ValueError) as info:
        _typecheck_sub(BinaryOperation(BinaryOperation.Operator.MUL))
    assert (
        "'binary_operation' argument should have "
        "operator 'BinaryOperation.Operator.SUB' "
        "but found 'Operator.MUL'." in str(info.value)
    )


def test__typecheck_add_sub():
    """Test that the _typecheck_add_sub function raises the correct errors."""
    with pytest.raises(TypeError) as info:
        _typecheck_add_sub(None)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'NoneType'." in str(info.value)
    )
    with pytest.raises(ValueError) as info:
        _typecheck_add_sub(BinaryOperation(BinaryOperation.Operator.MUL))
    assert (
        "'binary_operation' argument should have "
        "operator either 'BinaryOperation.Operator.ADD' "
        "or 'BinaryOperation.Operator.SUB' "
        "but found 'Operator.MUL'." in str(info.value)
    )


def test__typecheck_mul_div():
    """Test that the _typecheck_mul_div function raises the correct errors."""
    with pytest.raises(TypeError) as info:
        _typecheck_mul_div(None)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'NoneType'." in str(info.value)
    )
    with pytest.raises(ValueError) as info:
        _typecheck_mul_div(BinaryOperation(BinaryOperation.Operator.ADD))
    assert (
        "'binary_operation' argument should have "
        "an operator among 'BinaryOperation.Operator.MUL' and "
        "'BinaryOperation.Operator.DIV' but found "
        "'Operator.ADD'." in str(info.value)
    )


def test__typecheck_unary_operation():
    """Test that the _typecheck_unary_operation function raises the correct 
    errors."""
    with pytest.raises(TypeError) as info:
        _typecheck_unary_operation(None)
    assert (
        "'unary_operation' argument should be "
        "of type 'UnaryOperation' but found "
        "'NoneType'." in str(info.value)
    )


def test__typecheck_minus():
    """Test that the _typecheck_minus function raises the correct errors."""
    with pytest.raises(TypeError) as info:
        _typecheck_minus(None)
    assert (
        "'unary_operation' argument should be "
        "of type 'UnaryOperation' but found "
        "'NoneType'." in str(info.value)
    )
    with pytest.raises(ValueError) as info:
        _typecheck_minus(UnaryOperation(UnaryOperation.Operator.EXP))
    assert (
        "'unary_operation' argument should have "
        "operator 'UnaryOperation.Operator.MINUS' "
        "but found 'Operator.EXP'." in str(info.value)
    )


def test__typecheck_assignment():
    """Test that the _typecheck_assignment function raises the correct 
    errors."""
    with pytest.raises(TypeError) as info:
        _typecheck_assignment(None)
    assert (
        "'assignment' argument should be "
        "of type 'Assignment' but found "
        "'NoneType'." in str(info.value)
    )


def test__typecheck_at_least_one_literal_operand():
    """Test that the _typecheck_at_least_one_literal_operand function raises 
    the correct errors."""
    with pytest.raises(TypeError) as info:
        _typecheck_at_least_one_literal_operand(None)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'NoneType'." in str(info.value)
    )

    ref = Reference(DataSymbol("a", INTEGER_TYPE))
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, ref.copy(), ref.copy()
    )

    with pytest.raises(TypeError) as info:
        _typecheck_at_least_one_literal_operand(binary)
    assert (
        "At least one of 'binary_operation' argument children should be "
        "of type 'Literal' but found "
        "'Reference' and "
        "'Reference'." in str(info.value)
    )


def test__valuecheck_at_least_one_literal_operand():
    """Test that the _valuecheck_at_least_one_literal_operand function raises 
    the correct errors."""
    with pytest.raises(TypeError) as info:
        _valuecheck_at_least_one_literal_operand(None, None)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'NoneType'." in str(info.value)
    )
    with pytest.raises(TypeError) as info:
        _valuecheck_at_least_one_literal_operand(
            BinaryOperation(BinaryOperation.Operator.MUL), None
        )
    assert (
        "'values' argument should be of "
        "type 'list[str]' or 'tuple[str]' but found "
        "'NoneType'." in str(info.value)
    )
    with pytest.raises(TypeError) as info:
        _valuecheck_at_least_one_literal_operand(
            BinaryOperation(BinaryOperation.Operator.MUL),
            [
                None,
            ],
        )
    assert (
        "'values' argument should be of "
        "type 'list[str]' or 'tuple[str]' but found an element of type "
        "'NoneType'." in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("0", INTEGER_TYPE),
        Literal("0", INTEGER_TYPE),
    )
    with pytest.raises(ValueError) as info:
        _valuecheck_at_least_one_literal_operand(binary, ["1"])
    assert (
        "At least one of 'binary_operation' argument children should be "
        "a Literal with value in ['1'] but found "
        "'0' and '0'." in str(info.value)
    )

    # Should pass
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("0", INTEGER_TYPE),
        Literal("1", INTEGER_TYPE),
    )
    _valuecheck_at_least_one_literal_operand(binary, ["1"])

    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        _valuecheck_at_least_one_literal_operand(binary, ["1"])
    assert (
        "At least one of 'binary_operation' argument children should be "
        "a Literal with value in ['1'] but found" in str(info.value)
    )


def test_simplify_binary_operation_errors():
    """Test that the simplify_binary_operation function raises the correct 
    errors."""
    with pytest.raises(TypeError) as info:
        simplify_binary_operation(
            BinaryOperation(BinaryOperation.Operator.MUL), None
        )
    assert (
        "'times' argument should be of "
        "type 'int' but found "
        "'NoneType'." in str(info.value)
    )

    with pytest.raises(ValueError) as info:
        simplify_binary_operation(
            BinaryOperation(BinaryOperation.Operator.MUL), 0
        )
    assert (
        "'times' argument should be at least "
        "'1' but found "
        "'0'." in str(info.value)
    )


def test_simplify_add_twice_to_mul_by_2_errors():
    """Test that the simplify_add_twice_to_mul_by_2 function raises the 
    correct errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_add_twice_to_mul_by_2(binary)
    assert (
        "'binary_operation' argument should have children "
        "of type 'Reference' and both equal "
        "but found" in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        Reference(DataSymbol("b", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_add_twice_to_mul_by_2(binary)
    assert (
        "'binary_operation' argument should have children "
        "of type 'Reference' and both equal "
        "but found" in str(info.value)
    )

    # Should pass
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    simplify_add_twice_to_mul_by_2(binary)


def test_simplify_add_minus_minus_errors():
    """Test that the simplify_add_minus_minus function raises the 
    correct errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_add_minus_minus(binary)
    assert (
        "'binary_operation' argument children should both be of type "
        "'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS'"
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        unary.copy(),
    )
    with pytest.raises(ValueError) as info:
        simplify_add_minus_minus(binary)
    assert (
        "'binary_operation' argument children should both be of type "
        "'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS'"
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, unary.copy(), unary.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_add_minus_minus(binary)
    assert (
        "'binary_operation' argument children should both be of type "
        "'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS'"
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    minus = UnaryOperation(UnaryOperation.Operator.MINUS)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, unary.copy(), minus.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_add_minus_minus(binary)
    assert (
        "'binary_operation' argument children should both be of type "
        "'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS'"
        "but found" in str(info.value)
    )

    # Should pass
    minus = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, minus.copy(), minus.copy()
    )
    simplify_add_minus_minus(binary)


def test_simplify_add_plus_minus_or_minus_plus_errors():
    """Test that the simplify_add_plus_minus_or_minus_plus function raises the 
    correct errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_add_plus_minus_or_minus_plus(binary)
    assert (
        "Exactly one of 'binary_operation' argument children should have "
        "type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        unary.copy(),
    )
    with pytest.raises(ValueError) as info:
        simplify_add_plus_minus_or_minus_plus(binary)
    assert (
        "Exactly one of 'binary_operation' argument children should have "
        "type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, unary.copy(), unary.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_add_plus_minus_or_minus_plus(binary)
    assert (
        "Exactly one of 'binary_operation' argument children should have "
        "type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    minus = UnaryOperation(UnaryOperation.Operator.MINUS)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, minus.copy(), minus.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_add_plus_minus_or_minus_plus(binary)
    assert (
        "Exactly one of 'binary_operation' argument children should have "
        "type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    # Should pass
    unary = UnaryOperation.create(
        UnaryOperation.Operator.EXP, Literal("0", INTEGER_TYPE)
    )
    minus = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, unary.copy(), minus.copy()
    )
    simplify_add_plus_minus_or_minus_plus(binary)


def test_simplify_add_zero_errors():
    """Test that the simplify_add_zero function raises the correct errors."""
    unary = UnaryOperation.create(
        UnaryOperation.Operator.MINUS,
        Literal("0", INTEGER_TYPE),
    )
    with pytest.raises(TypeError) as info:
        simplify_add_zero(unary)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'UnaryOperation'." in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_add_zero(binary)
    assert (
        "'binary_operation' argument should have "
        "operator 'BinaryOperation.Operator.ADD' "
        "but found" in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("1", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_add_zero(binary)
    assert (
        "At least one of 'binary_operation' argument children should "
        "be a Literal with value in ('0', '0.', '0.0') but found"
        in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    simplify_add_zero(binary)


def test_simplify_sub_itself_to_zero_errors():
    """Test that the simplify_sub_itself_to_zero function raises the 
    correct errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_itself_to_zero(binary)
    assert (
        "'binary_operation' argument should have children "
        "of type 'Reference' and both equal "
        "but found" in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        Reference(DataSymbol("b", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_itself_to_zero(binary)
    assert (
        "'binary_operation' argument should have children "
        "of type 'Reference' and both equal "
        "but found" in str(info.value)
    )

    # Should pass
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    simplify_sub_itself_to_zero(binary)


def test_simplify_sub_minus_plus_errors():
    """Test that the simplify_sub_minus_plus function raises the correct 
    errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_minus_plus(binary)
    assert (
        "'binary_operation' argument should have first child of "
        "type 'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        unary.copy(),
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_minus_plus(binary)
    assert (
        "'binary_operation' argument should have first child of "
        "type 'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, unary.copy(), unary.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_minus_plus(binary)
    assert (
        "'binary_operation' argument should have first child of "
        "type 'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    # Should pass
    unary = UnaryOperation.create(
        UnaryOperation.Operator.EXP, Literal("0", INTEGER_TYPE)
    )
    minus = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, minus.copy(), unary.copy()
    )
    simplify_sub_minus_plus(binary)

    # Should pass
    minus = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, minus.copy(), minus.copy()
    )
    simplify_sub_minus_plus(binary)


def test_simplify_sub_plus_minus_errors():
    """Test that the simplify_sub_plus_minus function raises the correct 
    errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_plus_minus(binary)
    assert (
        "'binary_operation' argument should have second child of "
        "type 'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        unary.copy(),
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_plus_minus(binary)
    assert (
        "'binary_operation' argument should have second child of "
        "type 'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, unary.copy(), unary.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_plus_minus(binary)
    assert (
        "'binary_operation' argument should have second child of "
        "type 'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    # Should pass
    unary = UnaryOperation.create(
        UnaryOperation.Operator.EXP, Literal("0", INTEGER_TYPE)
    )
    minus = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, unary.copy(), minus.copy()
    )
    simplify_sub_plus_minus(binary)

    # Should pass
    minus = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB, minus.copy(), minus.copy()
    )
    simplify_sub_plus_minus(binary)


def test_simplify_add_sub_factorize_errors():
    """Test that the simplify_add_sub_factorize function raises the correct 
    errors."""
    a = Reference(DataSymbol("a", INTEGER_TYPE))
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, a.copy(), a.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_add_sub_factorize(binary)
    assert (
        "'binary_operation' argument should have exactly one child "
        "of type 'Reference' "
        "but found" in str(info.value)
    )

    a = Reference(DataSymbol("a", INTEGER_TYPE))
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, a.copy(), Literal("0", INTEGER_TYPE)
    )
    with pytest.raises(TypeError) as info:
        simplify_add_sub_factorize(binary)
    assert (
        "'binary_operation' argument should have one child "
        "of type 'Reference' and the other of type 'BinaryOperation' "
        "but found" in str(info.value)
    )

    a = Reference(DataSymbol("a", INTEGER_TYPE))
    subbinary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, a.copy(), Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, a.copy(), subbinary.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_add_sub_factorize(binary)
    assert (
        "'binary_operation' argument should have one child "
        "of type 'Reference' and the other of type 'BinaryOperation' "
        "with operator 'BinaryOperation.Operator.MUL' but found"
        in str(info.value)
    )

    a = Reference(DataSymbol("a", INTEGER_TYPE))
    b = Reference(DataSymbol("b", INTEGER_TYPE))
    subbinary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, b.copy(), Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, a.copy(), subbinary.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_add_sub_factorize(binary)
    assert (
        "'binary_operation' argument should have one child 'operand' "
        "of type 'Reference' and the other of type 'BinaryOperation' "
        "with an operand equal to 'operand' but found" in str(info.value)
    )

    a = Reference(DataSymbol("a", INTEGER_TYPE))
    subbinary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, a.copy(), Literal("1", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.ADD, a.copy(), subbinary.copy()
    )
    simplify_add_sub_factorize(binary)


def test_simplify_sub_zero_errors():
    """Test that the simplify_sub_zero function raises the correct errors."""
    unary = UnaryOperation.create(
        UnaryOperation.Operator.MINUS,
        Literal("0", INTEGER_TYPE),
    )
    with pytest.raises(TypeError) as info:
        simplify_sub_zero(unary)
    assert (
        "'binary_operation' argument should be of "
        "type 'BinaryOperation' but found "
        "'UnaryOperation'." in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_zero(binary)
    assert (
        "'binary_operation' argument should have "
        "operator 'BinaryOperation.Operator.SUB' "
        "but found" in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Literal("1", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_sub_zero(binary)
    assert (
        "At least one of 'binary_operation' argument children should "
        "be a Literal with value in ('0', '0.', '0.0') but found"
        in str(info.value)
    )

    binary = BinaryOperation.create(
        BinaryOperation.Operator.SUB,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    simplify_sub_zero(binary)


def test_simplify_mul_div_minus_minus_errors():
    """Test that the simplify_mul_div_minus_minus function raises the 
    correct errors."""
    # _typecheck functions already tested

    # Should pass
    minus = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, minus.copy(), minus.copy()
    )
    simplify_mul_div_minus_minus(binary)


def test_simplify_mul_div_plus_minus_or_minus_plus_errors():
    """Test that the simplify_mul_div_plus_minus_or_minus_plus function raises 
    the correct errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    with pytest.raises(ValueError) as info:
        simplify_mul_div_plus_minus_or_minus_plus(binary)
    assert (
        "Exactly one of 'binary_operation' argument children should have "
        "type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        unary.copy(),
    )
    with pytest.raises(ValueError) as info:
        simplify_mul_div_plus_minus_or_minus_plus(binary)
    assert (
        "Exactly one of 'binary_operation' argument children should have "
        "type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    unary = UnaryOperation(UnaryOperation.Operator.EXP)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, unary.copy(), unary.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_mul_div_plus_minus_or_minus_plus(binary)
    assert (
        "Exactly one of 'binary_operation' argument children should have "
        "type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    minus = UnaryOperation(UnaryOperation.Operator.MINUS)
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, minus.copy(), minus.copy()
    )
    with pytest.raises(ValueError) as info:
        simplify_mul_div_plus_minus_or_minus_plus(binary)
    assert (
        "Exactly one of 'binary_operation' argument children should have "
        "type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
        "but found" in str(info.value)
    )

    # Should pass
    unary = UnaryOperation.create(
        UnaryOperation.Operator.EXP, Literal("0", INTEGER_TYPE)
    )
    minus = UnaryOperation.create(
        UnaryOperation.Operator.MINUS, Literal("0", INTEGER_TYPE)
    )
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL, unary.copy(), minus.copy()
    )
    simplify_mul_div_plus_minus_or_minus_plus(binary)


def test_simplify_mul_by_one_errors():
    """Test that the simplify_mul_by_one function raises the correct errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Literal("1", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    simplify_mul_by_one(binary)

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        Literal("1", INTEGER_TYPE),
    )
    simplify_mul_by_one(binary)

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Literal("1", INTEGER_TYPE),
        Literal("1", INTEGER_TYPE),
    )
    simplify_mul_by_one(binary)


def test_simplify_mul_by_zero_errors():
    """Test that the simplify_mul_by_zero function raises the correct errors."""
    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Literal("0", INTEGER_TYPE),
        Reference(DataSymbol("a", INTEGER_TYPE)),
    )
    simplify_mul_by_zero(binary)

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Reference(DataSymbol("a", INTEGER_TYPE)),
        Literal("0", INTEGER_TYPE),
    )
    simplify_mul_by_zero(binary)

    binary = BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Literal("0", INTEGER_TYPE),
        Literal("0", INTEGER_TYPE),
    )
    simplify_mul_by_zero(binary)


def test_simplify_self_assignment_errors():
    """Test that the simplify_self_assignment function raises the correct 
    errors."""

    a = Reference(DataSymbol("a", INTEGER_TYPE))
    b = Reference(DataSymbol("b", INTEGER_TYPE))
    assignment = Assignment.create(a.copy(), b.copy())
    with pytest.raises(ValueError) as info:
        simplify_self_assignment(assignment)
    assert (
        "'assignment' argument should satisfy "
        "'assignment.lhs == assignment.rhs' but found" in str(info.value)
    )

    ref = Reference(DataSymbol("a", INTEGER_TYPE))
    assignment = Assignment.create(ref.copy(), ref.copy())
    simplify_self_assignment(assignment)


def test_simplify_node_errors():
    """Test that the simplify_node function raises the correct errors."""

    with pytest.raises(TypeError) as info:
        simplify_node(None)
    assert (
        "'node' argument should be "
        "of type 'Node' but found "
        "'NoneType'." in str(info.value)
    )

    node = Literal("1", INTEGER_TYPE)
    with pytest.raises(TypeError) as info:
        simplify_node(node, None)
    assert (
        "'times' argument should be "
        "of type 'int' but found "
        "'NoneType'." in str(info.value)
    )

    node = Literal("1", INTEGER_TYPE)
    with pytest.raises(ValueError) as info:
        simplify_node(node, 0)
    assert (
        "'times' argument should be "
        "at least '1' but found "
        "'0'." in str(info.value)
    )


##############
# Values


def test_simplify_add_twice_to_mul_by_2(fortran_writer):
    """Test that the simplify_add_twice_to_mul_by_2 function returns the 
    expected result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    add = BinaryOperation.create(BinaryOperation.Operator.ADD, x, x.copy())
    assert fortran_writer(simplify_add_twice_to_mul_by_2(add)) == "2 * x"


def test_simplify_add_minus_minus(fortran_writer):
    """Test that the simplify_add_minus_minus function returns the 
    expected result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    y = Reference(DataSymbol("y", INTEGER_TYPE))
    minus_x, minus_y = [
        UnaryOperation.create(UnaryOperation.Operator.MINUS, var)
        for var in [x, y]
    ]
    add = BinaryOperation.create(
        BinaryOperation.Operator.ADD, minus_x, minus_y
    )
    assert fortran_writer(simplify_add_minus_minus(add)) == "-(x + y)"


def test_simplify_add_plus_minus_or_minus_plus(fortran_writer):
    """Test that the simplify_add_plus_minus_or_minus_plus function returns 
    the expected result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    y = Reference(DataSymbol("y", INTEGER_TYPE))
    minus_x, minus_y = [
        UnaryOperation.create(UnaryOperation.Operator.MINUS, var)
        for var in [x, y]
    ]
    sub = BinaryOperation.create(
        BinaryOperation.Operator.ADD, x.copy(), minus_y.copy()
    )
    assert (
        fortran_writer(simplify_add_plus_minus_or_minus_plus(sub)) == "x - y"
    )

    sub = BinaryOperation.create(
        BinaryOperation.Operator.ADD, minus_x.copy(), y.copy()
    )
    assert (
        fortran_writer(simplify_add_plus_minus_or_minus_plus(sub)) == "y - x"
    )


def test_simplify_add_zero(fortran_writer):
    """Test that the simplify_add_zero function returns the expected result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))

    # x + 0
    zero = Literal("0", INTEGER_TYPE)
    x_plus_zero = BinaryOperation.create(
        BinaryOperation.Operator.ADD, x.copy(), zero.copy()
    )
    assert fortran_writer(simplify_add_zero(x_plus_zero)) == "x"

    # 0 + x
    zero_plus_x = BinaryOperation.create(
        BinaryOperation.Operator.ADD, zero.copy(), x.copy()
    )
    assert fortran_writer(simplify_add_zero(zero_plus_x)) == "x"

    # 0. and 0.0
    for zero_str in ("0.", "0.0"):
        zero = Literal(zero_str, REAL_TYPE)

        # x + 0
        x_plus_zero = BinaryOperation.create(
            BinaryOperation.Operator.ADD, x.copy(), zero.copy()
        )
        assert fortran_writer(simplify_add_zero(x_plus_zero)) == "x"

        # 0 + x
        zero_plus_x = BinaryOperation.create(
            BinaryOperation.Operator.ADD, zero.copy(), x.copy()
        )
        assert fortran_writer(simplify_add_zero(zero_plus_x)) == "x"


def test_simplify_sub_itself_to_zero(fortran_writer):
    """Test that the simplify_sub_itself_to_zero function returns the expected 
    result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    sub = BinaryOperation.create(BinaryOperation.Operator.SUB, x, x.copy())
    assert fortran_writer(simplify_sub_itself_to_zero(sub)) == "0"


def test_simplify_sub_minus_plus(fortran_writer):
    """Test that the simplify_sub_minus_plus function returns the expected 
    result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    y = Reference(DataSymbol("y", INTEGER_TYPE))
    minus_x = UnaryOperation.create(UnaryOperation.Operator.MINUS, x)
    sub = BinaryOperation.create(
        BinaryOperation.Operator.SUB, minus_x, y.copy()
    )
    assert fortran_writer(simplify_sub_minus_plus(sub)) == "-(x + y)"


def test_simplify_sub_plus_minus(fortran_writer):
    """Test that the simplify_sub_plus_minus function returns the expected 
    result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    y = Reference(DataSymbol("y", INTEGER_TYPE))
    minus_y = UnaryOperation.create(UnaryOperation.Operator.MINUS, y)
    sub = BinaryOperation.create(
        BinaryOperation.Operator.SUB, x.copy(), minus_y
    )
    assert fortran_writer(simplify_sub_plus_minus(sub)) == "x + y"


def test_simplify_sub_zero(fortran_writer):
    """Test that the simplify_sub_zero function returns the expected result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))

    # x - 0
    zero = Literal("0", INTEGER_TYPE)
    x_minus_zero = BinaryOperation.create(
        BinaryOperation.Operator.SUB, x.copy(), zero.copy()
    )
    assert fortran_writer(simplify_sub_zero(x_minus_zero)) == "x"

    # 0 - x
    zero_minus_x = BinaryOperation.create(
        BinaryOperation.Operator.SUB, zero.copy(), x.copy()
    )
    assert fortran_writer(simplify_sub_zero(zero_minus_x)) == "-x"

    # 0. and 0.0
    for zero_str in ("0.", "0.0"):
        zero = Literal(zero_str, REAL_TYPE)

        # x - 0
        x_minus_zero = BinaryOperation.create(
            BinaryOperation.Operator.SUB, x.copy(), zero.copy()
        )
        assert fortran_writer(simplify_sub_zero(x_minus_zero)) == "x"

        # 0 - x
        zero_minus_x = BinaryOperation.create(
            BinaryOperation.Operator.SUB, zero.copy(), x.copy()
        )
        assert fortran_writer(simplify_sub_zero(zero_minus_x)) == "-x"


def test_simplify_add_sub_factorize(fortran_writer):
    """Test that the simplify_add_sub_factorize function returns the expected 
    result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    y = Reference(DataSymbol("y", INTEGER_TYPE))
    x_times_y = BinaryOperation.create(
        BinaryOperation.Operator.MUL, x.copy(), y.copy()
    )
    y_times_x = BinaryOperation.create(
        BinaryOperation.Operator.MUL, y.copy(), x.copy()
    )

    add = BinaryOperation.create(
        BinaryOperation.Operator.ADD, x.copy(), y_times_x.copy()
    )
    assert fortran_writer(simplify_add_sub_factorize(add)) == "(1 + y) * x"

    add = BinaryOperation.create(
        BinaryOperation.Operator.ADD, x.copy(), x_times_y.copy()
    )
    assert fortran_writer(simplify_add_sub_factorize(add)) == "x * (1 + y)"

    add = BinaryOperation.create(
        BinaryOperation.Operator.ADD, y_times_x.copy(), x.copy()
    )
    assert fortran_writer(simplify_add_sub_factorize(add)) == "(y + 1) * x"

    add = BinaryOperation.create(
        BinaryOperation.Operator.ADD, x_times_y.copy(), x.copy()
    )
    assert fortran_writer(simplify_add_sub_factorize(add)) == "x * (y + 1)"

    sub = BinaryOperation.create(
        BinaryOperation.Operator.SUB, x.copy(), y_times_x.copy()
    )
    assert fortran_writer(simplify_add_sub_factorize(sub)) == "(1 - y) * x"

    sub = BinaryOperation.create(
        BinaryOperation.Operator.SUB, x.copy(), x_times_y.copy()
    )
    assert fortran_writer(simplify_add_sub_factorize(sub)) == "x * (1 - y)"

    sub = BinaryOperation.create(
        BinaryOperation.Operator.SUB, y_times_x.copy(), x.copy()
    )
    assert fortran_writer(simplify_add_sub_factorize(sub)) == "(y - 1) * x"

    sub = BinaryOperation.create(
        BinaryOperation.Operator.SUB, x_times_y.copy(), x.copy()
    )
    assert fortran_writer(simplify_add_sub_factorize(sub)) == "x * (y - 1)"


def test_simplify_mul_div_minus_minus(fortran_writer):
    """Test that the simplify_mul_div_minus_minus function returns the expected 
    result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    y = Reference(DataSymbol("y", INTEGER_TYPE))
    minus_x, minus_y = [
        UnaryOperation.create(UnaryOperation.Operator.MINUS, var)
        for var in [x, y]
    ]
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, minus_x, minus_y
    )
    assert fortran_writer(simplify_mul_div_minus_minus(mul)) == "x * y"

    div = BinaryOperation.create(
        BinaryOperation.Operator.DIV, minus_x.copy(), minus_y.copy()
    )
    assert fortran_writer(simplify_mul_div_minus_minus(div)) == "x / y"


def test_simplify_mul_div_plus_minus_or_minus_plus(fortran_writer):
    """Test that the simplify_mul_div_plus_minus_or_minus_plus function returns 
    the expected result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    y = Reference(DataSymbol("y", INTEGER_TYPE))
    minus_x, minus_y = [
        UnaryOperation.create(UnaryOperation.Operator.MINUS, var)
        for var in [x, y]
    ]
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, minus_x.copy(), y.copy()
    )
    assert (
        fortran_writer(simplify_mul_div_plus_minus_or_minus_plus(mul))
        == "-x * y"
    )

    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, x.copy(), minus_y.copy()
    )
    assert (
        fortran_writer(simplify_mul_div_plus_minus_or_minus_plus(mul))
        == "-x * y"
    )

    div = BinaryOperation.create(
        BinaryOperation.Operator.DIV, minus_x.copy(), y.copy()
    )
    assert (
        fortran_writer(simplify_mul_div_plus_minus_or_minus_plus(div))
        == "-x / y"
    )

    div = BinaryOperation.create(
        BinaryOperation.Operator.DIV, x.copy(), minus_y.copy()
    )
    assert (
        fortran_writer(simplify_mul_div_plus_minus_or_minus_plus(div))
        == "-x / y"
    )


def test_simplify_mul_by_one(fortran_writer):
    """Test that the simplify_mul_by_one function returns the expected result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    one = Literal("1", INTEGER_TYPE)
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, x.copy(), one.copy()
    )
    assert fortran_writer(simplify_mul_by_one(mul)) == "x"
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, one.copy(), x.copy()
    )
    assert fortran_writer(simplify_mul_by_one(mul)) == "x"

    one = Literal("1.", REAL_TYPE)
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, x.copy(), one.copy()
    )
    assert fortran_writer(simplify_mul_by_one(mul)) == "x"
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, one.copy(), x.copy()
    )
    assert fortran_writer(simplify_mul_by_one(mul)) == "x"

    one = Literal("1.0", REAL_TYPE)
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, x.copy(), one.copy()
    )
    assert fortran_writer(simplify_mul_by_one(mul)) == "x"
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, one.copy(), x.copy()
    )
    assert fortran_writer(simplify_mul_by_one(mul)) == "x"


def test_simplify_mul_by_zero(fortran_writer):
    """Test that the simplify_mul_by_zero function returns the expected 
    result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    zero = Literal("0", INTEGER_TYPE)
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, x.copy(), zero.copy()
    )
    assert fortran_writer(simplify_mul_by_zero(mul)) == "0"
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, zero.copy(), x.copy()
    )
    assert fortran_writer(simplify_mul_by_zero(mul)) == "0"

    zero = Literal("0.0", REAL_TYPE)
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, x.copy(), zero.copy()
    )
    assert fortran_writer(simplify_mul_by_zero(mul)) == "0"
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, zero.copy(), x.copy()
    )
    assert fortran_writer(simplify_mul_by_zero(mul)) == "0"

    zero = Literal("0.", REAL_TYPE)
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, x.copy(), zero.copy()
    )
    assert fortran_writer(simplify_mul_by_zero(mul)) == "0"
    mul = BinaryOperation.create(
        BinaryOperation.Operator.MUL, zero.copy(), x.copy()
    )
    assert fortran_writer(simplify_mul_by_zero(mul)) == "0"


def test_simplify_self_assignment():
    """Test that the simplify_self_assignment function returns the expected 
    result."""
    x = Reference(DataSymbol("x", INTEGER_TYPE))
    assignment = Assignment.create(x, x.copy())
    assert simplify_self_assignment(assignment) is None


if __name__ == "__main__":
    print("Testing simplify")
    test__typecheck_binary_operation()
    test__typecheck_add()
    test__typecheck_sub()
    test__typecheck_add_sub()
    test__typecheck_mul_div()
    test__typecheck_at_least_one_literal_operand()
    test__valuecheck_at_least_one_literal_operand()

    test__typecheck_unary_operation()
    test__typecheck_minus()

    test__typecheck_assignment()

    test_simplify_add_twice_to_mul_by_2_errors()
    test_simplify_binary_operation_errors()
    test_simplify_add_minus_minus_errors()
    test_simplify_add_plus_minus_or_minus_plus_errors()
    test_simplify_add_zero_errors()
    test_simplify_sub_itself_to_zero_errors()
    test_simplify_sub_minus_plus_errors()
    test_simplify_sub_plus_minus_errors()
    test_simplify_sub_zero_errors()
    test_simplify_add_sub_factorize_errors()
    test_simplify_mul_div_minus_minus_errors()
    test_simplify_mul_div_plus_minus_or_minus_plus_errors()
    test_simplify_mul_by_one_errors()
    test_simplify_mul_by_zero_errors()

    test_simplify_self_assignment_errors()

    test_simplify_node_errors()

    # Values
    fortran_writer = FortranWriter()
    test_simplify_add_twice_to_mul_by_2(fortran_writer)
    test_simplify_add_minus_minus(fortran_writer)
    test_simplify_add_plus_minus_or_minus_plus(fortran_writer)
    test_simplify_add_zero(fortran_writer)
    test_simplify_sub_itself_to_zero(fortran_writer)
    test_simplify_sub_minus_plus(fortran_writer)
    test_simplify_sub_plus_minus(fortran_writer)
    test_simplify_sub_zero(fortran_writer)
    test_simplify_add_sub_factorize(fortran_writer)
    test_simplify_mul_div_minus_minus(fortran_writer)
    test_simplify_mul_div_plus_minus_or_minus_plus(fortran_writer)
    test_simplify_mul_by_one(fortran_writer)
    test_simplify_mul_by_zero(fortran_writer)
    test_simplify_self_assignment()
    print("passed")
