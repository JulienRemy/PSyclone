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

"""A module to perform tests on the autodiff ADReverseCallTrans class.
"""

import pytest
from itertools import product

from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.symbols import (
    DataSymbol,
    REAL_TYPE,
    RoutineSymbol,
)
from psyclone.psyir.nodes import (
    Call,
    Reference,
    Container,
    Schedule,
    Assignment,
    Routine,
)
from psyclone.psyir.transformations import TransformationError
from psyclone.autodiff.transformations import (
    ADReverseRoutineTrans,
    ADReverseContainerTrans,
    ADReverseCallTrans,
)
from psyclone.autodiff.tapes import ADValueTape
from psyclone.autodiff import (
    own_routine_symbol,
    add,
    zero,
    one,
    ADSplitReversalSchedule,
    ADJointReversalSchedule,
)

AP = ADReverseRoutineTrans._differential_prefix
AS = ADReverseRoutineTrans._differential_postfix
OA = ADReverseRoutineTrans._operation_adjoint_name
TaP = ADValueTape._tape_prefix

RECP = ADReverseRoutineTrans._recording_prefix
RECS = ADReverseRoutineTrans._recording_postfix
RETP = ADReverseRoutineTrans._returning_prefix
RETS = ADReverseRoutineTrans._returning_postfix
REVP = ADReverseRoutineTrans._reversing_prefix
REVS = ADReverseRoutineTrans._reversing_postfix

SRC = """subroutine foo()
end subroutine foo
subroutine bar()
end subroutine bar"""


def initialize_transformations(
    routine="foo", src=SRC, reversal_schedule=ADSplitReversalSchedule()
):
    freader = FortranReader()
    psy = freader.psyir_from_source(src)
    container = psy.walk(Container)[0]

    ad_container_trans = ADReverseContainerTrans()
    ad_container_trans.apply(container, routine, [], [], reversal_schedule)
    ad_routine_trans = ad_container_trans.routine_transformations[0]

    return ad_container_trans, ad_routine_trans, ad_routine_trans.call_trans


def test_ad_call_trans_initialization():
    with pytest.raises(TypeError) as info:
        ADReverseCallTrans(None)
    assert (
        "Argument should be of type 'ADRoutineTrans' "
        "but found 'NoneType'." in str(info.value)
    )

    _, ad_routine_trans, ad_call_trans = initialize_transformations()

    assert ad_call_trans.routine_trans == ad_routine_trans


def test_ad_call_trans_validate():
    ad_container_trans, _, ad_call_trans = initialize_transformations()

    with pytest.raises(TransformationError) as info:
        ad_call_trans.validate(None)
    assert (
        "'call' argument should be a "
        "PSyIR 'Call' but found 'NoneType'." in str(info.value)
    )

    routine_symbol = RoutineSymbol("routine")
    call = Call.create(routine_symbol, [one()])

    with pytest.raises(NotImplementedError) as info:
        ad_call_trans.validate(call)
    assert "Transforming function calls is not " "supported yet." in str(info.value)

    schedule = Schedule([call])
    call = schedule.children[0]

    with pytest.raises(TransformationError) as info:
        ad_call_trans.validate(call)
    assert (
        "Called routine named 'routine' "
        "can be found neither in the routine_map "
        "(already transformed) "
        "nor in the names of routines in the container "
        "(possible to transform)." in str(info.value)
    )

    # As if 'routine' was already transformed, should pass
    ad_container_trans.routine_map[routine_symbol] = ()
    ad_call_trans.validate(call)

    # 'bar' is a subroutine in the container, should pass
    call = Call.create(RoutineSymbol("bar"), [])
    schedule = Schedule([call])
    call = schedule.children[0]
    ad_call_trans.validate(call)

    # Named argument
    call = Call.create(
        RoutineSymbol("bar"),
        [
            ("arg", one()),
        ],
    )
    schedule = Schedule([call])
    call = schedule.children[0]
    with pytest.raises(NotImplementedError) as info:
        ad_call_trans.validate(call)
    assert "Transforming Call with named arguments is not implemented yet." in str(
        info.value
    )


def test_ad_call_trans_transform_literal_argument():
    _, ad_routine_trans, ad_call_trans = initialize_transformations()
    with pytest.raises(TypeError) as info:
        ad_call_trans.transform_literal_argument(None)
    assert (
        "'literal' argument should be of type 'Literal' "
        "but found 'NoneType'." in str(info.value)
    )

    literal = one()
    args, temp, adj = ad_call_trans.transform_literal_argument(literal)
    assert args == [literal, Reference(ad_routine_trans.temp_symbols[0])]

    assert len(temp) == 1
    assert isinstance(temp[0], Assignment)
    assert temp[0].lhs == Reference(ad_routine_trans.temp_symbols[0])
    assert temp[0].rhs == zero(ad_routine_trans._default_differential_datatype)

    assert len(adj) == 0


def test_ad_call_trans_transform_reference_argument():
    _, ad_routine_trans, ad_call_trans = initialize_transformations()
    with pytest.raises(TypeError) as info:
        ad_call_trans.transform_reference_argument(None)
    assert (
        "'reference' argument should be of type 'Reference' "
        "but found 'NoneType'." in str(info.value)
    )

    sym = DataSymbol("var", REAL_TYPE)
    adj_sym = ad_routine_trans.create_differential_symbol(sym)
    ref = Reference(sym)
    args, temp, adj = ad_call_trans.transform_reference_argument(ref)
    assert args == [ref, Reference(adj_sym)]

    assert len(temp) == 0

    assert len(adj) == 0


def test_ad_call_trans_transform_operation_argument():
    _, ad_routine_trans, ad_call_trans = initialize_transformations()
    with pytest.raises(TypeError) as info:
        ad_call_trans.transform_operation_argument(None)
    assert (
        "'operation' argument should be of type 'Operation' "
        "but found 'NoneType'." in str(info.value)
    )

    sym1 = DataSymbol("var1", REAL_TYPE)
    sym2 = DataSymbol("var2", REAL_TYPE)
    adj1 = ad_routine_trans.create_differential_symbol(sym1)
    adj2 = ad_routine_trans.create_differential_symbol(sym2)

    operation = add(sym1, sym2)

    args, temp, adj = ad_call_trans.transform_operation_argument(operation)
    ope_adj = ad_routine_trans.operation_adjoints[0]
    assert args == [operation, Reference(ope_adj)]

    assert len(temp) == 1
    assert isinstance(temp[0], Assignment)
    assert temp[0].lhs == Reference(ope_adj)
    assert temp[0].rhs == zero(ad_routine_trans._default_differential_datatype)

    #assert adj == ad_routine_trans.operation_trans.apply(operation, ope_adj)


def test_ad_call_trans_transform_call_arguments():
    _, ad_routine_trans, ad_call_trans = initialize_transformations()
    with pytest.raises(TypeError) as info:
        ad_call_trans.transform_call_arguments(None)
    assert "'call' argument should be of type 'Call' " "but found 'NoneType'." in str(
        info.value
    )

    lit = one()
    syms = [DataSymbol("var" + str(i), REAL_TYPE) for i in range(3)]
    adj_syms = [ad_routine_trans.create_differential_symbol(sym) for sym in syms]
    op = add(syms[1], syms[2])
    args = [lit, Reference(syms[0]), op]
    call = Call.create(RoutineSymbol("routine"), args)

    ret_args, temp, adj = ad_call_trans.transform_call_arguments(call)

    _, ad_routine_trans, ad_call_trans = initialize_transformations()
    adj_syms = [ad_routine_trans.create_differential_symbol(sym) for sym in syms]
    funcs = [
        ad_call_trans.transform_literal_argument,
        ad_call_trans.transform_reference_argument,
        ad_call_trans.transform_operation_argument,
    ]
    comp_arg = []
    comp_temp = []
    comp_adj = []
    for arg, func in zip(args, funcs):
        ret, te, ad = func(arg)
        comp_arg.extend(ret)
        comp_temp.extend(te)
        comp_adj.extend(ad)

    assert ret_args == comp_arg
    assert temp == comp_temp
    assert adj == comp_adj


def test_ad_call_trans_transform_called_routine():
    ad_container_trans, ad_routine_trans, ad_call_trans = initialize_transformations()

    with pytest.raises(TypeError) as info:
        ad_call_trans.transform_called_routine(None)
    assert (
        "'routine' argument should be of type 'Routine' "
        "but found 'NoneType'." in str(info.value)
    )

    routine = ad_container_trans.container.walk(Routine)[1]
    assert routine.name == "bar"
    routine_sym = own_routine_symbol(routine)
    rec_sym, ret_sym, rev_sym, tape = ad_call_trans.transform_called_routine(routine)

    assert isinstance(rec_sym, RoutineSymbol)
    assert isinstance(ret_sym, RoutineSymbol)
    assert isinstance(rev_sym, RoutineSymbol)
    assert isinstance(tape, ADValueTape)

    assert rec_sym.name == f"{RECP}bar{RECS}"
    assert ret_sym.name == f"{RETP}bar{RETS}"
    assert rev_sym.name == f"{REVP}bar{REVS}"

    assert tape.name == f"{TaP}bar"
    assert ad_container_trans.value_tape_map[routine_sym] == tape

    assert len(ad_container_trans.routine_transformations) == 2
    assert isinstance(ad_container_trans.routine_transformations[1], ADReverseRoutineTrans)
    assert ad_container_trans.routine_transformations[1].routine == routine

    assert ad_container_trans.routine_map[routine_sym] == [rec_sym, ret_sym, rev_sym]

    assert isinstance(ad_call_trans.called_routine_trans, ADReverseRoutineTrans)
    assert ad_call_trans.called_routine_trans != ad_routine_trans

    #
    rec_sym2, ret_sym2, rev_sym2, tape2 = ad_call_trans.transform_called_routine(
        routine
    )
    assert rec_sym != rec_sym2
    assert ret_sym != ret_sym2
    assert rev_sym != rev_sym2
    assert tape != tape2


def test_ad_call_trans_apply():
    src = """subroutine foo()
    end subroutine foo
    subroutine bar()
        call foo()
    end subroutine bar"""

    ad_container_trans, ad_routine_trans, ad_call_trans = initialize_transformations(
        routine="bar", src=src
    )

    with pytest.raises(TransformationError) as info:
        ad_call_trans.apply(None)
    assert (
        "'call' argument should be a "
        "PSyIR 'Call' but found 'NoneType'." in str(info.value)
    )

    call = ad_container_trans.container.walk(Call)[0]
    assert call.routine.name == "foo"
    call_sym = call.routine.name
    recording, returning = ad_call_trans.apply(call)
    recording2, returning2 = ad_call_trans.apply(call)
    assert recording == recording2
    assert returning == returning2

    # Split reversal
    ad_container_trans, ad_routine_trans, ad_call_trans = initialize_transformations(
        routine="bar", src=src, reversal_schedule=ADSplitReversalSchedule()
    )
    recording, returning = ad_call_trans.apply(call)
    assert ad_call_trans.routine_trans.container_trans == ad_container_trans
    assert len(recording) == 1
    assert len(returning) >= 1
    assert isinstance(recording[0], Call)
    assert any([isinstance(ret, Call) for ret in returning])
    rec_call = recording[0]
    routine_sym = ad_call_trans.routine_symbol
    # TODO: this is a problem from PSyclone
    assert routine_sym != call_sym
    # check that recording contains a call to the recording routine
    assert rec_call.routine == ad_container_trans.routine_map[routine_sym][0]
    # check that returning contains a call to the returning routine
    for ret in returning:
        if isinstance(ret, Call):
            assert ret.routine == ad_container_trans.routine_map[routine_sym][1]

    # Apply twice on different routines
    src2 = """subroutine foo()
    end subroutine foo
    subroutine foo2()
    end subroutine foo2
    subroutine bar()
        call foo()
        call foo2()
    end subroutine bar"""

    ad_container_trans, ad_routine_trans, ad_call_trans = initialize_transformations(
        routine="bar", src=src2, reversal_schedule=ADJointReversalSchedule()
    )
    call = ad_container_trans.container.walk(Call)[0]
    call2 = ad_container_trans.container.walk(Call)[1]

    recording, returning = ad_call_trans.apply(call)
    routines = ad_call_trans.transformed.copy()

    recording2, returning2 = ad_call_trans.apply(call2)
    routines2 = ad_call_trans.transformed.copy()

    assert [routine.name for routine in routines] == [
        f"{RECP}foo{RECS}",
        f"{RETP}foo{RETS}",
        f"{REVP}foo{REVS}",
    ]

    assert [routine.name for routine in routines2] == [
        f"{RECP}foo2{RECS}",
        f"{RETP}foo2{RETS}",
        f"{REVP}foo2{REVS}",
    ]

    # Joint reversal
    ad_container_trans, ad_routine_trans, ad_call_trans = initialize_transformations(
        routine="bar", src=src, reversal_schedule=ADJointReversalSchedule()
    )
    recording, returning = ad_call_trans.apply(call)
    assert len(recording) == 1
    assert len(returning) >= 1
    assert isinstance(recording[0], Call)
    assert any([isinstance(ret, Call) for ret in returning])
    rec_call = recording[0]
    routine_sym = ad_call_trans.routine_symbol
    # check that recording contains a call to the original routine
    assert rec_call.routine == routine_sym
    # check that returning contains a call to the reversing routine
    for ret in returning:
        if isinstance(ret, Call):
            assert ret.routine == ad_container_trans.routine_map[routine_sym][2]

    # Test the arguments
    sources = (
        """
    subroutine foo(x)
        implicit none
        real :: x
        x = 1.0
    end subroutine foo
    subroutine bar(a)
        implicit none
        real :: a
        call foo(a)
    end subroutine bar""",
        """subroutine foo(x,y)
    implicit none
        real :: x, y
        x = 1.0
        y = 2.0
    end subroutine foo
    subroutine bar(a, b)
        implicit none
        real :: a, b
        call foo(a,b)
    end subroutine bar""",
        """subroutine foo(x,y,z)
    implicit none
        real :: x, y, z
        x = 1.0
        y = 2.0
        z = 3.0
    end subroutine foo
    subroutine bar(a, b, c)
        implicit none
        real :: a,b,c
        call foo(a,b,c)
    end subroutine bar""",
        """subroutine foo(x,y,z)
        implicit none
        real :: x, y, z
        y = 2.0
        z = 3.0
        x = y
    end subroutine foo
    subroutine bar(a, b, c)
        implicit none
        real :: a,b,c
        call foo(a,b,c)
    end subroutine bar""",
        """subroutine foo(x,y,z)
        implicit none
        real :: x, y, z
        x = 1.0
        y = 2.0
        z = 3.0
        x = x + y
    end subroutine foo
    subroutine bar(a, b, c)
        implicit none
        real :: a,b,c
        call foo(a,b,c)
    end subroutine bar""",
    )

    schedules = (ADJointReversalSchedule(), ADSplitReversalSchedule())
    for src, sched in product(sources, schedules):
        (
            ad_container_trans,
            ad_routine_trans,
            ad_call_trans,
        ) = initialize_transformations(routine='bar', src=src, reversal_schedule=sched)
        call = ad_container_trans.container.walk(Call)[0]
        ad_call_trans.apply(call)

        _test_arguments(ad_call_trans)


def _test_arguments(applied_ad_call_trans):
    arg_list = applied_ad_call_trans.routine_table.argument_list
    recording_args = applied_ad_call_trans.recording.symbol_table.argument_list
    returning_args = applied_ad_call_trans.returning.symbol_table.argument_list
    reversing_args = applied_ad_call_trans.reversing.symbol_table.argument_list

    # Some symbols are copies so compare their names
    assert recording_args[: len(arg_list)] == arg_list
    for arg in arg_list:
        assert arg.name in [ret.name for ret in returning_args]
        assert arg in reversing_args
    for i, arg in enumerate(reversing_args[: len(returning_args)]):
        assert returning_args[i].name == arg.name 

    called_routine_trans = applied_ad_call_trans.called_routine_trans
    for arg in returning_args:
        # Get the non adjoints
        for sym in called_routine_trans.data_symbol_differential_map:
            if arg.name == sym.name:
                arg_adj = called_routine_trans.data_symbol_differential_map[sym]
                assert arg_adj in returning_args
                assert returning_args.index(arg_adj) == returning_args.index(arg) + 1
    
    # Tape
    if len(recording_args) > 0:
        # joint reversal : no tape
        if applied_ad_call_trans.reversal_schedule is ADJointReversalSchedule():
            for arg in recording_args:
                assert TaP not in arg.name
            for arg in returning_args:
                assert TaP not in arg.name
        # split reversal : tape in rec and ret, not in rev
        elif applied_ad_call_trans.reversal_schedule is ADSplitReversalSchedule():
            if TaP in recording_args[-1].name:
                assert recording_args[-1].name == returning_args[-1].name
            for arg in reversing_args:
                assert TaP not in arg.name


#################"
# TODO: still a lot to test here
# tape slices
# arguments
# temp and adjoint assignments?

if __name__ == "__main__":
    print("Testing ADReverseCallTrans")
    from psyclone.psyir.backend.fortran import FortranWriter

    fwriter = FortranWriter()

    test_ad_call_trans_initialization()
    test_ad_call_trans_validate()
    test_ad_call_trans_transform_literal_argument()
    test_ad_call_trans_transform_reference_argument()
    test_ad_call_trans_transform_operation_argument()
    test_ad_call_trans_transform_call_arguments()
    test_ad_call_trans_transform_called_routine()
    test_ad_call_trans_apply()
    print("passed")
