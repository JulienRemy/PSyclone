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

"""A module to perform tests on the autodiff ADValueTape class.
"""

import pytest

from psyclone.autodiff.tapes import ADValueTape

from psyclone.psyir.symbols import (
    ArrayType,
    REAL_TYPE,
    INTEGER_TYPE,
    DataSymbol,
    SymbolTable,
)
from psyclone.psyir.nodes import Routine, Reference

TaP = ADValueTape._tape_prefix


def test_ad_value_tape_initialization():
    with pytest.raises(TypeError) as info:
        ADValueTape(None, REAL_TYPE)
    assert "'name' argument should be of type " "'str' but found 'NoneType'." in str(
        info.value
    )

    with pytest.raises(TypeError) as info:
        ADValueTape("name", None)
    assert (
        "'datatype' argument should be of type "
        "'ScalarType' or 'ArrayType' but found "
        "'NoneType'." in str(info.value)
    )

    array_type = ArrayType(REAL_TYPE, [2])
    with pytest.raises(NotImplementedError) as info:
        ADValueTape("name", array_type)
    assert (
        "Only ScalarType symbols can be value_taped "
        "for now but found DataType "
        "'Array<Scalar<REAL, UNDEFINED>, shape=[2]>' instead." in str(info.value)
    )

    # with pytest.raises(NotImplementedError) as info:
    #    ADValueTape("name", INTEGER_TYPE)
    # assert (
    #    "Only REAL symbols can be value_taped "
    #    "for now but found "
    #    "'Intrinsic.INTEGER' instead." in str(info.value)
    # )

    ad_value_tape = ADValueTape("name", REAL_TYPE)

    assert ad_value_tape.datatype == REAL_TYPE
    assert ad_value_tape.recorded_nodes == []
    assert ad_value_tape.length == 0

    assert isinstance(ad_value_tape.symbol, DataSymbol)
    assert ad_value_tape.symbol.name == f"{TaP}name"
    assert ad_value_tape.symbol.datatype == ArrayType(REAL_TYPE, [0])


def test_ad_value_tape_record(fortran_writer):
    ad_value_tape = ADValueTape("name", REAL_TYPE)

    assert ad_value_tape.name == f"{TaP}name"

    with pytest.raises(TypeError) as info:
        ad_value_tape.record(None)
    assert (
        "'reference' argument should be of type "
        "'Reference' but found "
        "'NoneType'." in str(info.value)
    )

    int_sym = DataSymbol("int_var", INTEGER_TYPE)
    int_ref = Reference(int_sym)
    with pytest.raises(TypeError) as info:
        ad_value_tape.record(int_ref)
    assert (
        "The intrinsic datatype of the 'reference' argument "
        "should be Intrinsic.REAL but found "
        "Intrinsic.INTEGER." in str(info.value)
    )

    array_sym = DataSymbol("array_var", ArrayType(REAL_TYPE, [2]))
    ref = Reference(array_sym)
    with pytest.raises(NotImplementedError) as info:
        ad_value_tape.record(ref)
    assert "Taping arrays is not implemented yet." in str(info.value)

    sym_1 = DataSymbol("var_1", REAL_TYPE)
    var_1 = Reference(sym_1)
    sym_2 = DataSymbol("var_2", REAL_TYPE)
    var_2 = Reference(sym_2)

    rec_1 = ad_value_tape.record(var_1)
    assert fortran_writer(rec_1) == f"{TaP}name(1) = var_1\n"
    assert ad_value_tape.length == 1
    assert ad_value_tape.recorded_nodes == [var_1]

    rec_2 = ad_value_tape.record(var_1)
    assert fortran_writer(rec_2) == f"{TaP}name(2) = var_1\n"
    assert ad_value_tape.length == 2
    assert ad_value_tape.recorded_nodes == [var_1, var_1]

    rec_3 = ad_value_tape.record(var_2)
    assert fortran_writer(rec_3) == f"{TaP}name(3) = var_2\n"
    assert ad_value_tape.length == 3
    assert ad_value_tape.recorded_nodes == [var_1, var_1, var_2]


def test_ad_value_tape__has_last():
    ad_value_tape = ADValueTape("name", REAL_TYPE)

    with pytest.raises(TypeError) as info:
        ad_value_tape._has_last(None)
    assert (
        "'node' argument should be of type among "
        "['Reference'] but found "
        "'NoneType'." in str(info.value)
    )

    var_1 = DataSymbol("var_1", REAL_TYPE)
    ref_1 = Reference(var_1)
    var_2 = DataSymbol("var_2", REAL_TYPE)
    ref_2 = Reference(var_2)

    ad_value_tape.record(ref_1)
    ad_value_tape._has_last(ref_1)

    with pytest.raises(ValueError) as info:
        ad_value_tape._has_last(ref_2)
    assert (
        "node argument named var_2 was not "
        "stored as last element of the value_tape." in str(info.value)
    )


def test_ad_value_tape_restore(fortran_writer):
    ad_value_tape = ADValueTape("name", REAL_TYPE)

    assert ad_value_tape.name == f"{TaP}name"

    with pytest.raises(TypeError) as info:
        ad_value_tape.restore(None)
    assert (
        "'reference' argument should be of type "
        "'Reference' but found "
        "'NoneType'." in str(info.value)
    )

    sym_1 = DataSymbol("var_1", REAL_TYPE)
    var_1 = Reference(sym_1)
    sym_2 = DataSymbol("var_2", REAL_TYPE)
    var_2 = Reference(sym_2)

    ad_value_tape.record(var_1)
    rest_1 = ad_value_tape.restore(var_1)
    assert ad_value_tape.length == 1
    assert fortran_writer(rest_1) == f"var_1 = {TaP}name(1)\n"

    ad_value_tape.record(var_1)
    rest_2 = ad_value_tape.restore(var_1)
    assert ad_value_tape.length == 2
    assert fortran_writer(rest_2) == f"var_1 = {TaP}name(2)\n"

    ad_value_tape.record(var_2)
    rest_3 = ad_value_tape.restore(var_2)
    assert ad_value_tape.length == 3
    assert fortran_writer(rest_3) == f"var_2 = {TaP}name(3)\n"


def test_ad_value_tape_reshape(fortran_writer):
    ad_value_tape = ADValueTape("name", REAL_TYPE)
    assert ad_value_tape.length == 0

    sym_1 = DataSymbol("var_1", REAL_TYPE)
    var_1 = Reference(sym_1)

    for i in range(1, 5):
        ad_value_tape.record(var_1)
        assert ad_value_tape.length == i == len(ad_value_tape.recorded_nodes)

        ad_value_tape.reshape()
        assert ad_value_tape.symbol.datatype == ArrayType(REAL_TYPE, [i])

        symbol_table = SymbolTable()
        symbol_table.add(ad_value_tape.symbol)
        routine = Routine.create("my_routine", symbol_table, [])
        src = f"real, dimension({i}) :: {TaP}name\n"

        assert src in fortran_writer(routine)


def test_ad_value_tape_extend():
    tape1 = ADValueTape("1", REAL_TYPE)
    tape2 = ADValueTape("2", REAL_TYPE)
    tape3 = ADValueTape("3", INTEGER_TYPE)

    with pytest.raises(TypeError) as info:
        tape1.extend(None)
    assert (
        "'tape' argument should be of type "
        "'ADValueTape' but found "
        "'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        tape1.extend(tape3)
    assert (
        "'tape' argument should have elements of datatype "
        "'Scalar<REAL, UNDEFINED>' but found "
        "'Scalar<INTEGER, UNDEFINED>'." in str(info.value)
    )

    syms1 = [DataSymbol("var" + str(i), REAL_TYPE) for i in range(10)]
    refs1 = [Reference(sym) for sym in syms1]
    syms2 = [DataSymbol("VAR" + str(i), REAL_TYPE) for i in range(15)]
    refs2 = [Reference(sym) for sym in syms2]

    for ref in refs1:
        tape1.record(ref)
    for ref in refs2:
        tape2.record(ref)

    tape1.extend(tape2)

    assert tape1.length == 10 + 15
    assert tape1.recorded_nodes == refs1 + refs2


def test_ad_value_tape_extend_and_slice(fortran_writer):
    def init_tapes():
        tape1 = ADValueTape("1", REAL_TYPE)
        tape2 = ADValueTape("2", REAL_TYPE)
        tape3 = ADValueTape("3", INTEGER_TYPE)
        return tape1, tape2, tape3

    tape1, tape2, tape3 = init_tapes()

    with pytest.raises(TypeError) as info:
        tape1.extend(None)
    assert (
        "'tape' argument should be of type "
        "'ADValueTape' but found "
        "'NoneType'." in str(info.value)
    )

    with pytest.raises(TypeError) as info:
        tape1.extend(tape3)
    assert (
        "'tape' argument should have elements of datatype "
        "'Scalar<REAL, UNDEFINED>' but found "
        "'Scalar<INTEGER, UNDEFINED>'." in str(info.value)
    )

    syms2 = [DataSymbol("VAR" + str(i), REAL_TYPE) for i in range(15)]
    refs2 = [Reference(sym) for sym in syms2]

    for ref in refs2:
        tape2.record(ref)

    tape_slice = tape1.extend_and_slice(tape2)
    assert tape1.length == 15
    assert tape1.recorded_nodes == refs2
    assert fortran_writer(tape_slice) == f"{TaP}1(:)"

    tape1.record(refs2[0])
    assert tape1.length == 16
    assert fortran_writer(tape_slice) == f"{TaP}1(:15)"

    #
    tape1, tape2, _ = init_tapes()
    syms1 = [DataSymbol("var" + str(i), REAL_TYPE) for i in range(10)]
    refs1 = [Reference(sym) for sym in syms1]
    syms2 = [DataSymbol("VAR" + str(i), REAL_TYPE) for i in range(15)]
    refs2 = [Reference(sym) for sym in syms2]
    for ref in refs1:
        tape1.record(ref)
    for ref in refs2:
        tape2.record(ref)

    tape_slice = tape1.extend_and_slice(tape2)
    assert tape1.length == 10 + 15
    assert tape1.recorded_nodes == refs1 + refs2
    assert fortran_writer(tape_slice) == f"{TaP}1(11:)"

    tape1.record(refs2[0])
    assert tape1.length == 10 + 15 + 1
    assert fortran_writer(tape_slice) == f"{TaP}1(11:25)"

if __name__ == "__main__":
    print("Testing ADValueTape")
    from psyclone.psyir.backend.fortran import FortranWriter

    fwriter = FortranWriter()

    test_ad_value_tape_initialization()
    test_ad_value_tape__has_last()
    test_ad_value_tape_record(fwriter)
    test_ad_value_tape_restore(fwriter)
    test_ad_value_tape_reshape(fwriter)
    test_ad_value_tape_extend()
    test_ad_value_tape_extend_and_slice(fwriter)

    print("passed")
