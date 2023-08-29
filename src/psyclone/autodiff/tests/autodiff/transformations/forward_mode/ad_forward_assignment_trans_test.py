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

"""A module to perform tests on the autodiff ADForwardAssignmentTrans class.
"""

import pytest

from psyclone.autodiff.transformations import (
    ADForwardAssignmentTrans,
    ADForwardContainerTrans,
    ADForwardRoutineTrans,
)
from psyclone.autodiff import assign, minus, add

from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.symbols import (
    DataSymbol,
    REAL_TYPE,
)
from psyclone.psyir.nodes import Literal, Container
from psyclone.psyir.transformations import TransformationError

AP = ADForwardRoutineTrans._differential_prefix
AS = ADForwardRoutineTrans._differential_postfix
TeP = ADForwardRoutineTrans._temp_name_prefix
TeS = ADForwardRoutineTrans._temp_name_postfix


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

    return ad_container_trans, ad_routine_trans, ad_routine_trans.assignment_trans


def test_ad_assignment_trans_initialization():
    with pytest.raises(TypeError) as info:
        ADForwardAssignmentTrans(None)
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
        "'assignment' argument should be a "
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
        adj_sym = ad_routine_trans.create_differential_symbol(sym)
        assert adj_sym.name == f"{AP}var{AS}"

        sym2 = DataSymbol("var_2", REAL_TYPE)
        adj_sym2 = ad_routine_trans.create_differential_symbol(sym2)
        assert adj_sym2.name == f"{AP}var_2{AS}"

        return ad_assignment_trans, sym, adj_sym, sym2, adj_sym2

    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()

    ###############
    # Literal assignment
    one = Literal("1.0", REAL_TYPE)
    literal_assignment = assign(sym, one)

    transformed = ad_assignment_trans.apply(literal_assignment)
    assert isinstance(transformed, list)
    assert len(transformed) == 2

    expected = (f"{AP}var{AS} = 0.0\n", "var = 1.0\n")

    compare(transformed, expected, fortran_writer)

    ##################
    # Reference assignment
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    reference_assignment = assign(sym, sym2)

    transformed = ad_assignment_trans.apply(reference_assignment)
    assert isinstance(transformed, list)
    assert len(transformed) == 2

    expected = (f"{AP}var{AS} = {AP}var_2{AS}\n", "var = var_2\n")

    compare(transformed, expected, fortran_writer)

    #######################
    # Operation assignment, unary
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    operation = minus(sym2)
    operation_assignment = assign(sym, operation)
    transformed = ad_assignment_trans.apply(operation_assignment)

    assert isinstance(transformed, list)
    assert len(transformed) == 2

    expected = (f"{AP}var{AS} = -{AP}var_2{AS}\n", "var = -var_2\n")

    compare(transformed, expected, fortran_writer)

    # Operation assignment, binary
    ad_assignment_trans, sym, adj_sym, sym2, adj_sym2 = initialize()
    operation = add(sym, sym2)
    operation_assignment = assign(sym, operation)

    transformed = ad_assignment_trans.apply(operation_assignment)

    assert isinstance(transformed, list)
    assert len(transformed) == 2

    expected = (f"{AP}var{AS} = {AP}var{AS} + {AP}var_2{AS}\n", "var = var + var_2\n")

    compare(transformed, expected, fortran_writer)

    #######################
    # Call assignment is not implemented yet


if __name__ == "__main__":
    print("Testing ADForwardAssignmentTrans")
    from psyclone.psyir.backend.fortran import FortranWriter

    fwriter = FortranWriter()

    test_ad_assignment_trans_initialization()
    test_ad_assignment_trans_validate()
    test_ad_assignment_trans_is_iterative()
    test_ad_assignment_trans_apply(fwriter)

    print("passed")
