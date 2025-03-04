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
utils.py file within the autodiff directory.

"""
import pytest

from psyclone.psyir.symbols import (
    DataSymbol,
    REAL_TYPE,
    INTEGER_TYPE,
    CHARACTER_TYPE,
    ArrayType,
    RoutineSymbol,
)

from psyclone.psyir.nodes import Reference, Routine, UnaryOperation

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
    assign_zero,
    sin,
    cos,
    square,
    div,
    exp,
    sign,
    own_routine_symbol
)


def test_datanode_error():
    """Test that the datanode function raises the correct exception."""

    with pytest.raises(TypeError) as info:
        datanode(None)
    assert (
        "The sym_or_datanode argument of datanode must be of "
        "type 'DataNode' or 'DataSymbol' but found "
        "'NoneType'." in str(info.value)
    )


def test_datanode():
    var_sym = DataSymbol("var", REAL_TYPE)
    var_ref = Reference(var_sym)

    assert datanode(var_sym) == var_ref
    assert datanode(var_ref) == var_ref


def test_assign_error():
    """Test that the assign function raises the correct exceptions."""

    var_sym = DataSymbol("var", REAL_TYPE)
    val_sym = DataSymbol("val", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        assign(None, val_sym)
    assert (
        "The variable argument of assign must be of "
        "type 'Reference' or 'DataSymbol' but found "
        "'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        assign(var_sym, None)
    assert (
        "The value argument of assign must be of "
        "type 'DataNode' or 'DataSymbol' but found "
        "'NoneType'." in str(info.value)
    )


def test_assign(fortran_writer):
    var_sym = DataSymbol("var", REAL_TYPE)
    var_ref = Reference(var_sym)

    val_sym = DataSymbol("val", REAL_TYPE)
    val_ref = Reference(val_sym)

    result = assign(var_sym, val_sym)
    assert fortran_writer(result) == "var = val\n"

    assert (
        assign(var_sym, val_sym)
        == assign(var_ref, val_sym)
        == assign(var_sym, val_ref)
        == assign(var_ref, val_ref)
    )


def test_assign_zero_error():
    with pytest.raises(TypeError) as info:
        assign_zero(None)
    assert (
        "The variable argument of assign_zero must be of "
        "type 'Reference' or 'DataSymbol' but found "
        "'NoneType'." in str(info.value)
    )


def test_assign_zero(fortran_writer):
    real_sym = DataSymbol("var", REAL_TYPE)
    real_ref = Reference(real_sym)

    int_sym = DataSymbol("var", INTEGER_TYPE)

    real_result = assign_zero(real_sym)
    assert fortran_writer(real_result) == "var = 0.0\n"

    int_result = assign_zero(int_sym)
    assert fortran_writer(int_result) == "var = 0\n"

    assert assign_zero(real_sym) == assign_zero(real_ref)


def test_increment_error():
    var_sym = DataSymbol("var", REAL_TYPE)
    val_sym = DataSymbol("val", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        increment(None, val_sym)
    assert (
        "The variable argument of increment must be of "
        "type 'Reference' or 'DataSymbol' but found "
        "'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        increment(var_sym, None)
    assert (
        "The value argument of increment must be of "
        "type 'DataNode' or 'DataSymbol' but found "
        "'NoneType'." in str(info.value)
    )


def test_increment(fortran_writer):
    var_sym = DataSymbol("var", REAL_TYPE)
    var_ref = Reference(var_sym)

    val_sym = DataSymbol("val", REAL_TYPE)
    val_ref = Reference(val_sym)

    result = increment(var_sym, val_sym)
    assert fortran_writer(result) == "var = var + val\n"

    assert (
        increment(var_sym, val_sym)
        == increment(var_ref, val_sym)
        == increment(var_sym, val_ref)
        == increment(var_ref, val_ref)
    )


def test_zero_error():
    with pytest.raises(TypeError) as info:
        zero(None)
    assert (
        "The datatype argument of zero should be of type "
        "psyir.symbols.ScalarType or psyir.symbols.ArrayType "
        "but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(NotImplementedError) as info:
        zero(CHARACTER_TYPE)
    assert (
        "Creating null Literals for types other than integer "
        "or real is not implemented yet." in str(info.value)
    )


def test_zero(fortran_writer):
    int_zero = zero(INTEGER_TYPE)
    real_zero = zero(REAL_TYPE)
    array_int_zero = zero(ArrayType(INTEGER_TYPE, [2]))
    array_real_zero = zero(ArrayType(REAL_TYPE, [2]))

    assert fortran_writer(int_zero) == "0"
    assert fortran_writer(real_zero) == "0.0"
    assert fortran_writer(array_int_zero) == "0"
    assert fortran_writer(array_real_zero) == "0.0"


def test_one_error():
    with pytest.raises(TypeError) as info:
        one(None)
    assert (
        "The datatype argument of one should be of type "
        "psyir.symbols.ScalarType or psyir.symbols.ArrayType "
        "but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(NotImplementedError) as info:
        one(CHARACTER_TYPE)
    assert (
        "Creating unitary Literals for types other than integer "
        "or real is not implemented yet." in str(info.value)
    )


def test_one(fortran_writer):
    int_one = one(INTEGER_TYPE)
    real_one = one(REAL_TYPE)
    array_int_one = one(ArrayType(INTEGER_TYPE, [2]))
    array_real_one = one(ArrayType(REAL_TYPE, [2]))

    assert fortran_writer(int_one) == "1"
    assert fortran_writer(real_one) == "1.0"
    assert fortran_writer(array_int_one) == "1"
    assert fortran_writer(array_real_one) == "1.0"


def test_minus_error():
    with pytest.raises(TypeError) as info:
        minus(None)
    assert (
        "Argument of minus must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_minus(fortran_writer):
    int_one = one(INTEGER_TYPE)
    var_sym = DataSymbol("var", REAL_TYPE)

    assert fortran_writer(minus(int_one)) == "-1"
    assert fortran_writer(minus(var_sym)) == "-var"


def test_inverse_error():
    with pytest.raises(TypeError) as info:
        inverse(None)
    assert (
        "Argument of inverse must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_inverse(fortran_writer):
    int_one = one(INTEGER_TYPE)
    var_sym = DataSymbol("var", REAL_TYPE)

    assert fortran_writer(inverse(int_one)) == "1.0 / 1"
    assert fortran_writer(inverse(var_sym)) == "1.0 / var"


def test_square_error():
    with pytest.raises(TypeError) as info:
        square(None)
    assert (
        "Argument of square must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_square(fortran_writer):
    int_one = one(INTEGER_TYPE)
    var_sym = DataSymbol("var", REAL_TYPE)

    assert fortran_writer(square(int_one)) == "1 ** 2"
    assert fortran_writer(square(var_sym)) == "var ** 2"


def test_log_error():
    with pytest.raises(TypeError) as info:
        log(None)
    assert (
        "Argument of log must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_log(fortran_writer):
    int_one = one(INTEGER_TYPE)
    var_sym = DataSymbol("var", REAL_TYPE)

    # NOTE: the IntrinsicCall nodes need to have a parent so that the backend
    # doesn't print eg. 'call LOG(1)\n' instead of 'LOG(1)'
    log1 = log(int_one)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, log1)
    logvar = log(var_sym)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, logvar)

    assert fortran_writer(log1) == "LOG(1)"
    assert fortran_writer(logvar) == "LOG(var)"


def test_exp_error():
    with pytest.raises(TypeError) as info:
        exp(None)
    assert (
        "Argument of exp must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_exp(fortran_writer):
    int_one = one(INTEGER_TYPE)
    var_sym = DataSymbol("var", REAL_TYPE)

    # NOTE: the IntrinsicCall nodes need to have a parent so that the backend
    # doesn't print eg. 'call LOG(1)\n' instead of 'LOG(1)'

    exp1 = exp(int_one)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, exp1)
    assert fortran_writer(exp1) == "EXP(1)"

    expvar = exp(var_sym)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, expvar)
    assert fortran_writer(expvar) == "EXP(var)"


def test_cos_error():
    with pytest.raises(TypeError) as info:
        cos(None)
    assert (
        "Argument of cos must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_cos(fortran_writer):
    int_one = one(INTEGER_TYPE)
    var_sym = DataSymbol("var", REAL_TYPE)

    # NOTE: the IntrinsicCall nodes need to have a parent so that the backend
    # doesn't print eg. 'call LOG(1)\n' instead of 'LOG(1)'

    cos1 = cos(int_one)
    cosvar = cos(var_sym)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, cos1)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, cosvar)

    assert fortran_writer(cos1) == "COS(1)"
    assert fortran_writer(cosvar) == "COS(var)"


def test_sin_error():
    with pytest.raises(TypeError) as info:
        sin(None)
    assert (
        "Argument of sin must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_sin(fortran_writer):
    int_one = one(INTEGER_TYPE)
    var_sym = DataSymbol("var", REAL_TYPE)

    # NOTE: the IntrinsicCall nodes need to have a parent so that the backend
    # doesn't print eg. 'call LOG(1)\n' instead of 'LOG(1)'

    sin1 = sin(int_one)
    sinvar = sin(var_sym)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, sin1)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, sinvar)

    assert fortran_writer(sin1) == "SIN(1)"
    assert fortran_writer(sinvar) == "SIN(var)"


def test_sqrt_error():
    with pytest.raises(TypeError) as info:
        sqrt(None)
    assert (
        "Argument of sqrt must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_sqrt(fortran_writer):
    int_one = one(INTEGER_TYPE)
    var_sym = DataSymbol("var", REAL_TYPE)

    # NOTE: the IntrinsicCall nodes need to have a parent so that the backend
    # doesn't print eg. 'call LOG(1)\n' instead of 'LOG(1)'

    sqrt1 = sqrt(int_one)
    sqrtvar = sqrt(var_sym)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, sqrt1)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, sqrtvar)

    assert fortran_writer(sqrt1) == "SQRT(1)"
    assert fortran_writer(sqrtvar) == "SQRT(var)"


def test_power_error():
    var_sym = DataSymbol("var", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        power(None, var_sym)
    assert (
        "lhs argument of power must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        power(var_sym, None)
    assert (
        "rhs argument of power must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_power(fortran_writer):
    lhs_sym = DataSymbol("lhs", REAL_TYPE)
    rhs_sym = DataSymbol("rhs", REAL_TYPE)

    assert fortran_writer(power(lhs_sym, rhs_sym)) == "lhs ** rhs"


def test_mul_error():
    var_sym = DataSymbol("var", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        mul(None, var_sym)
    assert (
        "lhs argument of mul must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        mul(var_sym, None)
    assert (
        "rhs argument of mul must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_mul(fortran_writer):
    lhs_sym = DataSymbol("lhs", REAL_TYPE)
    rhs_sym = DataSymbol("rhs", REAL_TYPE)

    assert fortran_writer(mul(lhs_sym, rhs_sym)) == "lhs * rhs"


def test_div_error():
    var_sym = DataSymbol("var", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        div(None, var_sym)
    assert (
        "lhs argument of div must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        div(var_sym, None)
    assert (
        "rhs argument of div must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_div(fortran_writer):
    lhs_sym = DataSymbol("lhs", REAL_TYPE)
    rhs_sym = DataSymbol("rhs", REAL_TYPE)

    assert fortran_writer(div(lhs_sym, rhs_sym)) == "lhs / rhs"


def test_sub_error():
    var_sym = DataSymbol("var", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        sub(None, var_sym)
    assert (
        "lhs argument of sub must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        sub(var_sym, None)
    assert (
        "rhs argument of sub must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_sub(fortran_writer):
    lhs_sym = DataSymbol("lhs", REAL_TYPE)
    rhs_sym = DataSymbol("rhs", REAL_TYPE)

    assert fortran_writer(sub(lhs_sym, rhs_sym)) == "lhs - rhs"


def test_add_error():
    var_sym = DataSymbol("var", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        add(None, var_sym)
    assert (
        "lhs argument of add must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        add(var_sym, None)
    assert (
        "rhs argument of add must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_add(fortran_writer):
    lhs_sym = DataSymbol("lhs", REAL_TYPE)
    rhs_sym = DataSymbol("rhs", REAL_TYPE)

    assert fortran_writer(add(lhs_sym, rhs_sym)) == "lhs + rhs"


def test_sign_error():
    var_sym = DataSymbol("var", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        sign(None, var_sym)
    assert (
        "lhs argument of sign must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        sign(var_sym, None)
    assert (
        "rhs argument of sign must be of type 'DataNode' or "
        "'DataSymbol' but found 'NoneType'." in str(info.value)
    )


def test_sign(fortran_writer):
    lhs_sym = DataSymbol("lhs", REAL_TYPE)
    rhs_sym = DataSymbol("rhs", REAL_TYPE)

    # NOTE: the IntrinsicCall nodes need to have a parent so that the backend
    # doesn't print eg. 'call LOG(1)\n' instead of 'LOG(1)'

    sign_call = sign(lhs_sym, rhs_sym)
    UnaryOperation.create(UnaryOperation.Operator.MINUS, sign_call)

    assert fortran_writer(sign_call) == "SIGN(lhs, rhs)"


def test_own_routine_symbol_error():
    with pytest.raises(TypeError) as info:
        own_routine_symbol(None)
    assert (
        "'routine' argument should be of "
        "type 'Routine' but found "
        "'NoneType'." in str(info.value)
    )


def test_own_routine_symbol():
    routine = Routine("routine_name")

    routine_symbol = own_routine_symbol(routine)

    assert isinstance(routine_symbol, RoutineSymbol)
    assert routine_symbol.name == "routine_name"
