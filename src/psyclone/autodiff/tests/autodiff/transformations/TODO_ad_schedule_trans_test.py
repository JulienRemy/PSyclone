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
# Authors: J. Remy, Inria

"""A module to perform tests on the autodiff ADReverseScheduleTrans class.
#"""

import pytest
from itertools import product

from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.symbols import (
    DataSymbol,
    INTEGER_TYPE,
    REAL_TYPE,
    RoutineSymbol,
)
from psyclone.psyir.nodes import (
    Literal,
    UnaryOperation,
    BinaryOperation,
    NaryOperation,
    Call,
    Reference,
    Container,
    Schedule,
    Assignment,
    Routine,
)
from psyclone.psyir.transformations import TransformationError
from psyclone.autodiff.transformations import (
    ADReverseOperationTrans,
    ADReverseScheduleTrans,
    ADReverseContainerTrans,
    ADReverseCallTrans,
)
from psyclone.autodiff.tapes import ADValueTape
from psyclone.autodiff import (
    own_routine_symbol,
    assign,
    add,
    zero,
    one,
    ADSplitReversalSchedule,
    ADJointReversalSchedule,
)

AP = ADReverseScheduleTrans._adjoint_prefix
AS = ADReverseScheduleTrans._adjoint_suffix
OA = ADReverseScheduleTrans._operation_adjoint_name
TaP = ADValueTape._tape_prefix

SRC = """subroutine foo()
end subroutine foo
"""
def initialize_transformations():
    freader = FortranReader()
    psy = freader.psyir_from_source(SRC)
    container = psy.walk(Container)[0]

    ad_container_trans = ADReverseContainerTrans()
    ad_container_trans.apply(container, 'foo', [], [], ADJointReversalSchedule())
    ad_schedule_trans = ADReverseScheduleTrans(ad_container_trans)

    return ad_container_trans, ad_schedule_trans


def test_ad_schedule_trans_initialization():
    ad_container_trans, ad_schedule_trans = initialize_transformations()
    assert not ad_schedule_trans._was_applied 
    assert ad_schedule_trans.container_trans == ad_container_trans
    assert 