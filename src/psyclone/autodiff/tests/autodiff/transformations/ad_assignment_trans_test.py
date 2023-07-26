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
# LOSS OF USE, DATA, OR PROFITeS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
# Authors: J. Remy, Inria

"""A module to perform tests on the autodiff ADAssignmentTrans class.
"""

import pytest

from psyclone.autodiff.transformations import (
    ADAssignmentTrans,
    ADContainerTrans,
    ADRoutineTrans,
)
from psyclone.autodiff import assign, minus, add, ADSplitReversalSchedule

from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.symbols import (
    DataSymbol,
    REAL_TYPE,
)
from psyclone.psyir.nodes import Literal, Container
from psyclone.psyir.transformations import TransformationError

AP = ADRoutineTrans._adjoint_prefix
AS = ADRoutineTrans._adjoint_suffix
TeP = ADRoutineTrans._temp_name_prefix
TeS = ADRoutineTrans._temp_name_suffix


def compare(nodes, strings, fortran_writer):
    assert len(nodes) == len(strings)
    for node, expected_line in zip(nodes, strings):
        line = fortran_writer(node)
        assert line == expected_line


def initialize_transformations():
    freader = FortranReader()
    reversal_schedule = ADSplitReversalSchedule()

    src = """subroutine foo()
    end subroutine foo"""
    psy = freader.psyir_from_source(src)
    container = psy.walk(Container)[0]

    ad_container_trans = ADContainerTrans()
    ad_container_trans.apply(container, "foo", [], [], reversal_schedule)
    ad_routine_trans = ad_container_trans.routine_transformations[0]

    return ad_container_trans, ad_routine_trans, ad_routine_trans.assignment_trans


def test_ad_assignment_trans_initialization():
    with pytest.raises(TypeError) as info:
        ADAssignmentTrans(None)
    assert (
        "Argument should be of type 'ADRoutineTrans' "
        "but found 'NoneType'." in str(info.value)
    )

    _, ad_routine_trans, ad_assignment_trans = initialize_transformations()

    assert ad_assignment_trans.routine_trans == ad_routine_trans


def test_ad_assignment_trans_validate():
    _, _, ad_assignment_trans = initialize_transformations()

    sym = DataSymbol("var", REAL_TYPE)
    assignment = assign(sym, sym)

    with pytest.raises(TransformationError) as info:
        ad_assignment_trans.validate(None)
    assert (
        "'assignment' argument in ADAssignmentTrans.apply should be a "
        "PSyIR 'Assignment' but found 'NoneType'." in str(info.value)
    )

    ad_assignment_trans.validate(assignment)


def test_ad_assignment_trans_is_iterative():
    _, _, ad_assignment_trans = initialize_transformations()

    with pytest.raises(TypeError) as info:
        ad_assignment_trans.is_iterative(None)
    assert (
        "'assignment' argument in is_iterative should be a "
        "PSyIR 'Assignment' but found 'NoneType'." in str(info.value)
    )

    sym = DataSymbol("var", REAL_TYPE)
    sym2 = DataSymbol("var2", REAL_TYPE)

    iterative = assign(sym, sym)
    not_iterative = assign(sym, sym2)
    assert ad_assignment_trans.is_iterative(iterative)
    assert not ad_assignment_trans.is_iterative(not_iterative)

    iterative = assign(sym, minus(sym))
    not_iterative = assign(sym, minus(sym2))
    assert ad_assignment_trans.is_iterative(iterative)
    assert not ad_assignment_trans.is_iterative(not_iterative)

    iterative = assign(sym, add(sym, sym2))
    not_iterative = assign(sym, add(sym2, sym2))
    assert ad_assignment_trans.is_iterative(iterative)
    assert not ad_assignment_trans.is_iterative(not_iterative)

    iterative = assign(sym, minus(add(sym, sym2)))
    not_iterative = assign(sym, minus(add(sym2, sym2)))
    assert ad_assignment_trans.is_iterative(iterative)
    assert not ad_assignment_trans.is_iterative(not_iterative)


def test_ad_assignment_trans_apply(fortran_writer):
    def initialize():
        _, ad_routine_trans, ad_assignment_trans = initialize_transformations()

        sym = DataSymbol("var", REAL_TYPE)
        adj_sym = ad_routine_trans.create_adjoint_symbol(sym)
        assert adj_sym.name == f"{AP}var{AS}"

        sym2 = DataSymbol("var_2", REAL_TYPE)
        adj_sym2 = ad_routine_trans.create_adjoint_symbol(sym2)
        assert adj_sym2.name == f"{AP}var_2{AS}"

        return ad_assignment_trans, sym, adj_sym, sym2, adj_sym2

    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()

    ###############
    # Literal assignment
    one = Literal("1.0", REAL_TYPE)
    literal_assignment = assign(sym, one)

    # Literal assignment, with overwriting (taping)
    literal_overwriting = ad_assignment_trans.apply(literal_assignment)
    recording, returning = literal_overwriting
    assert isinstance(recording, list)
    assert isinstance(returning, list)
    assert len(recording) == 1
    assert len(returning) == 1
    expected_rec = ("var = 1.0\n",)
    expected_ret = (f"{AP}var{AS} = 0.0\n",)
    compare(recording, expected_rec, fortran_writer)
    compare(returning, expected_ret, fortran_writer)

    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    literal_assignment = assign(sym, one)

    ##################
    # Reference assignment
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    reference_assignment = assign(sym, sym2)

    # Reference assignment, without overwriting, without iterative assignment
    recording, returning = ad_assignment_trans.apply(
        reference_assignment
    )
    assert isinstance(recording, list)
    assert isinstance(returning, list)
    assert len(recording) == 1
    assert len(returning) == 2
    expected_rec = ("var = var_2\n",)
    expected_ret = (
        f"{AP}var_2{AS} = {AP}var_2{AS} + {AP}var{AS}\n",
        f"{AP}var{AS} = 0.0\n",
    )
    compare(recording, expected_rec, fortran_writer)
    compare(returning, expected_ret, fortran_writer)

    # Reference assignment, with iterative statement
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    reference_assignment = assign(sym, sym)

    recording, returning = ad_assignment_trans.apply(
        reference_assignment
    )
    assert isinstance(recording, list)
    assert isinstance(returning, list)
    assert len(recording) == 1
    assert len(returning) == 0
    expected_rec = ("var = var\n",)
    expected_ret = tuple()
    compare(recording, expected_rec, fortran_writer)
    compare(returning, expected_ret, fortran_writer)

    #######################
    # Operation assignment, not iterative
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    operation = minus(sym2)
    operation_assignment = assign(sym, operation)
    recording, returning = ad_assignment_trans.apply(
        operation_assignment,
    )
    assert isinstance(recording, list)
    assert isinstance(returning, list)
    assert len(recording) == 1
    assert len(returning) == 2
    expected_rec = ("var = -var_2\n",)
    expected_ret = (
        f"{AP}var_2{AS} = {AP}var_2{AS} + {AP}var{AS} * (-1)\n",
        f"{AP}var{AS} = 0.0\n",
    )
    compare(recording, expected_rec, fortran_writer)
    compare(returning, expected_ret, fortran_writer)

    # Operation assignment, iterative unary
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    operation = minus(sym)
    operation_assignment = assign(sym, operation)
    recording, returning = ad_assignment_trans.apply(
        operation_assignment
    )
    assert isinstance(recording, list)
    assert isinstance(returning, list)
    assert len(recording) == 1
    assert len(returning) == 1
    expected_rec = ("var = -var\n",)
    expected_ret = (
        f"{AP}var{AS} = {AP}var{AS} * (-1)\n",
    )
    compare(recording, expected_rec, fortran_writer)
    compare(returning, expected_ret, fortran_writer)

    # Operation assignment, iterative binary
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    operation = add(sym, sym2)
    operation_assignment = assign(sym, operation)
    recording, returning = ad_assignment_trans.apply(
        operation_assignment
    )
    assert isinstance(recording, list)
    assert isinstance(returning, list)
    assert len(recording) == 1
    assert len(returning) == 2
    expected_rec = ("var = var + var_2\n",)
    expected_ret = (
        f"{AP}var_2{AS} = {AP}var_2{AS} + {AP}var{AS} * 1\n",
        f"{AP}var{AS} = {AP}var{AS} * 1\n",
    )
    compare(recording, expected_rec, fortran_writer)
    compare(returning, expected_ret, fortran_writer)

    # Operation assignment, iterative binary, other way
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    operation = add(sym2, sym)
    operation_assignment = assign(sym, operation)
    recording, returning = ad_assignment_trans.apply(
        operation_assignment
    )
    assert isinstance(recording, list)
    assert isinstance(returning, list)
    assert len(recording) == 1
    assert len(returning) == 2
    expected_rec = ("var = var_2 + var\n",)
    expected_ret = (
        f"{AP}var_2{AS} = {AP}var_2{AS} + {AP}var{AS} * 1\n",
        f"{AP}var{AS} = {AP}var{AS} * 1\n",
    )
    compare(recording, expected_rec, fortran_writer)
    compare(returning, expected_ret, fortran_writer)

    # Operation assignment, iterative binary, both
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    operation = add(sym, sym)
    operation_assignment = assign(sym, operation)
    recording, returning = ad_assignment_trans.apply(
        operation_assignment
    )
    assert isinstance(recording, list)
    assert isinstance(returning, list)
    assert len(recording) == 1
    assert len(returning) == 1
    expected_rec = ("var = var + var\n",)
    expected_ret = (
        #f"{AP}var_2{AS} = {AP}var_2{AS} + {AP}var{AS} * 1\n",
        f"{AP}var{AS} = {AP}var{AS} * (1 + 1)\n",
        #f"{AP}var{AS} = {AP}var{AS} + {AP}var{AS} * 1\n",
    )
    compare(recording, expected_rec, fortran_writer)
    compare(returning, expected_ret, fortran_writer)


    #######################
    # Call assignment is not implemented yet


from psyclone.psyir.backend.fortran import FortranWriter

fwriter = FortranWriter()

test_ad_assignment_trans_initialization()
test_ad_assignment_trans_validate()
test_ad_assignment_trans_is_iterative()
test_ad_assignment_trans_apply(fwriter)
