# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2022-2024, Science and Technology Facilities Council.
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
# Author: A. R. Porter, STFC Daresbury Lab
# Modified: R. W. Ford and S. Siso, STFC Daresbury Lab
#           J. Remy, Université Grenoble Alpes, Inria
# -----------------------------------------------------------------------------

'''
This module contains pytest tests for the IntrinsicCall node.

TODO #2341 - tests need to be added for all of the supported intrinsics.

'''

from itertools import product
import pytest
from psyclone.psyir.symbols.datatypes import INTEGER_SINGLE_TYPE

from psyclone.tests.utilities import Compile

from psyclone.psyir.nodes import (
    ArrayReference, Literal, IntrinsicCall, Reference, Schedule, Routine, Call,
    FileContainer, Assignment)
from psyclone.core import VariablesAccessInfo
from psyclone.psyir.nodes.intrinsic_call import IntrinsicCall, IAttr
from psyclone.psyir.symbols import (
    ArrayType, DataSymbol, INTEGER_TYPE, IntrinsicSymbol, REAL_TYPE,
    BOOLEAN_TYPE, CHARACTER_TYPE, ScalarType)
from psyclone.psyir.symbols.interfaces import ArgumentInterface, StaticInterface


def test_intrinsic_enum():
    '''Basic test for the IntrinsicCall.Intrinsic enum.'''
    assert isinstance(IntrinsicCall.Intrinsic.MINVAL, IAttr)
    assert hash(IntrinsicCall.Intrinsic.MINVAL) == hash("MINVAL")


def test_intrinsiccall_constructor():
    '''Tests that the class' constructor and its parent are called
    correctly.

    '''
    # Wrong type of routine argument.
    with pytest.raises(TypeError) as err:
        _ = IntrinsicCall(None)
    assert ("IntrinsicCall 'routine' argument should be an instance of "
            "IntrinsicCall.Intrinsic, but found 'NoneType'." in str(err.value))
    # Check that supplied intrinsic and optional parent node is stored
    # correctly.
    sched = Schedule()
    call = IntrinsicCall(IntrinsicCall.Intrinsic.MINVAL, parent=sched)
    assert call._intrinsic is IntrinsicCall.Intrinsic.MINVAL
    assert isinstance(call.routine, IntrinsicSymbol)
    assert call.routine.name == "MINVAL"
    assert call.parent is sched


def test_intrinsiccall_intrinsic():
    '''Tests the intrinsic property returns the type of intrinsics from
    the intrinsic property.

    '''
    call = IntrinsicCall(IntrinsicCall.Intrinsic.MAXVAL)
    assert call.intrinsic is IntrinsicCall.Intrinsic.MAXVAL


def test_intrinsiccall_is_elemental():
    '''Tests the is_elemental() method works as expected. There are
    currently no elemental intrinsics so we can only test for
    False.

    '''
    intrinsic = IntrinsicCall(IntrinsicCall.Intrinsic.SUM)
    assert intrinsic.is_elemental is False


def test_intrinsiccall_is_pure():
    '''Tests that the is_pure() method works as expected.'''
    intrinsic = IntrinsicCall(IntrinsicCall.Intrinsic.SUM)
    assert intrinsic.is_pure is True
    intrinsic = IntrinsicCall(IntrinsicCall.Intrinsic.ALLOCATE)
    assert intrinsic.is_pure is False


def test_intrinsiccall_is_inquiry():
    '''Test that the is_inquiry() method works as expected.'''
    intrinsic = IntrinsicCall(IntrinsicCall.Intrinsic.SUM)
    assert intrinsic.is_inquiry is False
    intrinsic = IntrinsicCall(IntrinsicCall.Intrinsic.ALLOCATED)
    assert intrinsic.is_inquiry is True


@pytest.mark.parametrize("intrinsic, result", [
                (IntrinsicCall.Intrinsic.ABS, True),
                (IntrinsicCall.Intrinsic.MIN, True),
                (IntrinsicCall.Intrinsic.MAX, True),
                (IntrinsicCall.Intrinsic.MAXVAL, False),
                (IntrinsicCall.Intrinsic.ALLOCATE, False),
                (IntrinsicCall.Intrinsic.MATMUL, False)])
def test_intrinsiccall_is_available_on_device(intrinsic, result):
    '''Tests that the is_available_on_device() method works as expected.'''
    intrinsic_call = IntrinsicCall(intrinsic)
    assert intrinsic_call.is_available_on_device() is result


def test_intrinsiccall_alloc_create():
    '''Tests the create() method supports various forms of 'allocate'.

    '''
    sym = DataSymbol("my_array", ArrayType(INTEGER_TYPE,
                                           [ArrayType.Extent.DEFERRED]))
    bsym = DataSymbol("my_array2", ArrayType(INTEGER_TYPE,
                                             [ArrayType.Extent.DEFERRED]))
    isym = DataSymbol("ierr", INTEGER_TYPE)
    csym = DataSymbol("msg", CHARACTER_TYPE)
    # Straightforward allocation of an array.
    alloc = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.ALLOCATE,
        [ArrayReference.create(sym, [Literal("20", INTEGER_TYPE)])])
    assert isinstance(alloc, IntrinsicCall)
    assert alloc.intrinsic is IntrinsicCall.Intrinsic.ALLOCATE
    assert isinstance(alloc.routine, IntrinsicSymbol)
    assert alloc.routine.name == "ALLOCATE"
    alloc = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.ALLOCATE,
        [Reference(sym), ("Mold", Reference(bsym))])
    assert isinstance(alloc, IntrinsicCall)
    assert alloc.argument_names == [None, "Mold"]
    alloc = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.ALLOCATE,
        [Reference(sym), ("Source", Reference(bsym)),
         ("stat", Reference(isym)), ("errmsg", Reference(csym))])
    assert alloc.argument_names == [None, "Source", "stat", "errmsg"]


def test_intrinsiccall_dealloc_create():
    '''Tests for the creation of a 'deallocate' call.

    '''
    sym = DataSymbol("my_array", ArrayType(INTEGER_TYPE,
                                           [ArrayType.Extent.DEFERRED]))
    ierr = DataSymbol("ierr", INTEGER_TYPE)
    dealloc = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.DEALLOCATE, [Reference(sym)])
    assert isinstance(dealloc, IntrinsicCall)
    assert dealloc.intrinsic is IntrinsicCall.Intrinsic.DEALLOCATE
    assert isinstance(dealloc.routine, IntrinsicSymbol)
    assert dealloc.routine.name == "DEALLOCATE"
    assert dealloc.children[0].symbol is sym
    # With 'stat' optional argument.
    dealloc = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.DEALLOCATE, [Reference(sym),
                                             ("Stat", Reference(ierr))])
    assert dealloc.argument_names == [None, "Stat"]


def test_intrinsiccall_random_create():
    '''Tests for the creation of a 'random' call.

    '''
    sym = DataSymbol("my_array", ArrayType(REAL_TYPE,
                                           [ArrayType.Extent.DEFERRED]))
    rand = IntrinsicCall.create(
        IntrinsicCall.Intrinsic.RANDOM_NUMBER, [Reference(sym)])
    assert isinstance(rand, IntrinsicCall)
    assert rand.intrinsic is IntrinsicCall.Intrinsic.RANDOM_NUMBER
    assert isinstance(rand.routine, IntrinsicSymbol)
    assert rand.routine.name == "RANDOM_NUMBER"
    assert rand.children[0].symbol is sym


@pytest.mark.parametrize("intrinsic_call", [
    IntrinsicCall.Intrinsic.MINVAL, IntrinsicCall.Intrinsic.MAXVAL,
    IntrinsicCall.Intrinsic.SUM])
def test_intrinsiccall_minmaxsum_create(intrinsic_call):
    '''Tests for the creation of the different argument options for
    'minval', 'maxval' and 'sum' IntrinsicCalls.

    '''
    array = DataSymbol(
        "my_array", ArrayType(REAL_TYPE, [ArrayType.Extent.DEFERRED]))
    dim = DataSymbol("dim", INTEGER_TYPE)
    mask = DataSymbol("mask", BOOLEAN_TYPE)

    # array only
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array)])
    assert isinstance(intrinsic, IntrinsicCall)
    assert intrinsic.intrinsic is intrinsic_call
    assert isinstance(intrinsic.routine, IntrinsicSymbol)
    intrinsic_name = intrinsic_call.name
    assert intrinsic.routine.name == intrinsic_name
    assert intrinsic.children[0].symbol is array
    # array and optional dim
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array), ("dim", Reference(dim))])
    assert intrinsic.argument_names == [None, "dim"]
    # array and optional mask
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array), ("mask", Reference(mask))])
    assert intrinsic.argument_names == [None, "mask"]
    # array and optional dim then optional mask
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array), ("dim", Reference(dim)),
                         ("mask", Reference(mask))])
    assert intrinsic.argument_names == [None, "dim", "mask"]
    # array and optional mask then optional dim
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array), ("mask", Reference(mask)),
                         ("dim", Reference(dim))])
    assert intrinsic.argument_names == [None, "mask", "dim"]
    # array and optional literal mask and optional literal dim
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [
            Reference(array),
            ("mask", Literal("1", INTEGER_TYPE)),
            ("dim", Literal("false", BOOLEAN_TYPE))])
    assert intrinsic.argument_names == [None, "mask", "dim"]


@pytest.mark.parametrize("intrinsic_call", [
    IntrinsicCall.Intrinsic.TINY, IntrinsicCall.Intrinsic.HUGE])
@pytest.mark.parametrize("form", ["array", "literal"])
def test_intrinsiccall_tinyhuge_create(intrinsic_call, form):
    '''Tests for the creation of the different argument options for
    'tiny' and 'huge' IntrinsicCalls.

    '''
    if form == "array":
        array = DataSymbol(
            "my_array", ArrayType(REAL_TYPE, [ArrayType.Extent.DEFERRED]))
        arg = Reference(array)
    else:  # "literal"
        arg = Literal("1.0", REAL_TYPE)
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [arg])
    assert isinstance(intrinsic, IntrinsicCall)
    assert intrinsic.intrinsic is intrinsic_call
    assert isinstance(intrinsic.routine, IntrinsicSymbol)
    intrinsic_name = intrinsic_call.name
    assert intrinsic.routine.name == intrinsic_name
    if form == "array":
        assert intrinsic.children[0].symbol is array
    else:  # "literal"
        assert intrinsic.children[0] is arg

def test_intrinsiccall_reshape_create():
    '''Tests for the creation of the different argument options for
    'reshape' IntrinsicCall.

    '''
    intrinsic_call = IntrinsicCall.Intrinsic.RESHAPE

    array = DataSymbol("array", ArrayType(INTEGER_TYPE, [4]))
    shape = DataSymbol("shape", ArrayType(INTEGER_TYPE, [2, 2]))
    pad = DataSymbol("array", ArrayType(INTEGER_TYPE, [1]))
    order = DataSymbol("shape", ArrayType(INTEGER_TYPE, [2, 2]))

    # array and shape
    intrinsic = IntrinsicCall.create(intrinsic_call,
                                       [Reference(array),
                                        Reference(shape)])
    assert isinstance(intrinsic, IntrinsicCall)
    assert intrinsic.intrinsic is intrinsic_call
    assert isinstance(intrinsic.routine, IntrinsicSymbol)
    intrinsic_name = intrinsic_call.name
    assert intrinsic.routine.name == intrinsic_name
    assert intrinsic.children[0].symbol is array
    assert intrinsic.children[1].symbol is shape
    # array, shape and optional pad
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array), Reference(shape),
                         ("pad", Reference(pad))])
    assert intrinsic.argument_names == [None, None, "pad"]
    # array, shape and optional order
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array), Reference(shape),
                         ("order", Reference(order))])
    assert intrinsic.argument_names == [None, None, "order"]
    # array, shape and optional pad then optional order
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array), Reference(shape),
                         ("pad", Reference(pad)),
                         ("order", Reference(order))])
    assert intrinsic.argument_names == [None, None, "pad", "order"]
    # array, shape and optional order then optional pad
    intrinsic = IntrinsicCall.create(
        intrinsic_call, [Reference(array), Reference(shape),
                         ("order", Reference(order)),
                         ("pad", Reference(pad))])
    assert intrinsic.argument_names == [None, None, "order", "pad"]

def test_intrinsiccall_create_errors():
    '''Checks for the validation/type checking in the create() method.

    '''
    sym = DataSymbol("my_array", ArrayType(INTEGER_TYPE,
                                           [ArrayType.Extent.DEFERRED]))
    aref = ArrayReference.create(sym, [Literal("20", INTEGER_TYPE)])
    with pytest.raises(TypeError) as err:
        IntrinsicCall.create("ALLOCATE", [Reference(sym)])
    assert ("'routine' argument should be an instance of "
            "IntrinsicCall.Intrinsic but found 'str'" in str(err.value))
    # Supplied arguments must be a list.
    with pytest.raises(TypeError) as err:
        IntrinsicCall.create(IntrinsicCall.Intrinsic.ALLOCATE, aref)
    assert ("IntrinsicCall.create() 'arguments' argument should be a list "
            "but found 'ArrayReference'" in str(err.value))
    # An allocate must have one or more References as argument.
    with pytest.raises(ValueError) as err:
        IntrinsicCall.create(IntrinsicCall.Intrinsic.ALLOCATE, [])
    assert ("The 'ALLOCATE' intrinsic requires at least 1 arguments but "
            "got 0" in str(err.value))
    # The random intrinsic only accepts one argument.
    with pytest.raises(ValueError) as err:
        IntrinsicCall.create(IntrinsicCall.Intrinsic.RANDOM_NUMBER,
                             [aref, aref.copy()])
    assert ("The 'RANDOM_NUMBER' intrinsic requires between 1 and 1 arguments "
            "but got 2" in str(err.value))
    # Wrong type for a positional argument.
    with pytest.raises(TypeError) as err:
        IntrinsicCall.create(IntrinsicCall.Intrinsic.ALLOCATE,
                             [sym])
    assert ("The 'ALLOCATE' intrinsic requires that positional arguments be "
            "of type " in str(err.value))
    assert "but got a 'DataSymbol'" in str(err.value)
    # Positional argument after named argument.
    with pytest.raises(ValueError) as err:
        IntrinsicCall.create(IntrinsicCall.Intrinsic.DEALLOCATE,
                             [Reference(sym), ("stat", aref), aref])
    assert ("Found a positional argument *after* a named argument ('stat'). "
            "This is invalid." in str(err.value))

    # TODO #2303: We can not enable the validation of positional parameters
    # unless we store their name, otherwise when we parse a positional argument
    # by name, which is valid fortran, it will fail.
    # (e.g. RANDOM_NUMBER(harvest=4)

    # with pytest.raises(ValueError) as err:
    #     IntrinsicCall.create(IntrinsicCall.Intrinsic.RANDOM_NUMBER,
    #                          [aref, ("willow", sym)])
    # assert ("The 'RANDOM_NUMBER' intrinsic does not support any optional "
    #         "arguments but got 'willow'" in str(err.value))
    # An allocate only supports the 'stat' and 'mold' arguments.
    # with pytest.raises(ValueError) as err:
    #     IntrinsicCall.create(IntrinsicCall.Intrinsic.ALLOCATE,
    #                          [aref, ("yacht", Reference(sym))])
    # assert ("The 'ALLOCATE' intrinsic supports the optional arguments "
    #         "['errmsg', 'mold', 'source', 'stat'] but got 'yacht'"
    #         in str(err.value))

    # Wrong type for the name of an optional argument.
    with pytest.raises(TypeError) as err:
        IntrinsicCall.create(IntrinsicCall.Intrinsic.ALLOCATE,
                             [aref, (sym, sym)])
    assert ("Optional arguments to an IntrinsicCall must be specified by a "
            "(str, Reference) tuple but got a DataSymbol instead of a str"
            in str(err.value))
    # Wrong type for an optional argument.
    with pytest.raises(TypeError) as err:
        IntrinsicCall.create(IntrinsicCall.Intrinsic.ALLOCATE,
                             [aref, ("stat", sym)])
    assert ("The optional argument 'stat' to intrinsic 'ALLOCATE' must be "
            "of type 'Reference' but got 'DataSymbol'" in str(err.value))

# Internal functions for testing datatypes

def _initialize_scalar_kinds():
    intrinsics = ['INTEGER', 'BOOLEAN', 'REAL']
    precisions = ['SINGLE', 'DOUBLE', 'UNDEFINED', 4, 8]

    integer_kinds = []
    boolean_kinds = []
    real_kinds = []
    for t in intrinsics:
        for k in precisions:
            if isinstance(k, str):
                scalar_type = ScalarType(ScalarType.Intrinsic[t], 
                                         ScalarType.Precision[k])
            else:
                scalar_type = ScalarType(ScalarType.Intrinsic[t], k)

            if t == 'INTEGER':
                integer_kinds.append(scalar_type)
            elif t == 'REAL':
                real_kinds.append(scalar_type)
            else:
                boolean_kinds.append(scalar_type)

    return integer_kinds, real_kinds, boolean_kinds

def _initialize_type_checkers(container, array_shape, arrays_only=False):
    integer_kinds, real_kinds, boolean_kinds = _initialize_scalar_kinds()
    scalar_datatypes = integer_kinds + real_kinds + boolean_kinds

    array_datatypes = [ArrayType(scalar_type, array_shape) 
                       for scalar_type in scalar_datatypes]

    routine_symbols = {}

    if arrays_only:
        datatypes = array_datatypes
    else:
        datatypes = scalar_datatypes + array_datatypes

    for datatype in datatypes:
        arg = DataSymbol("arg", datatype)
        arg.interface = ArgumentInterface(ArgumentInterface.Access.READ)

        if isinstance(datatype.precision, int):
            name = f"{datatype.intrinsic.name}_{datatype.precision}"
        else:
            name = f"{datatype.intrinsic.name}_{datatype.precision.name}"

        if datatype in array_datatypes:
            dim_str = f"{array_shape[0]}"
            if len(array_shape) == 2:
                dim_str += f"x{array_shape[1]}"
            name += "_" + dim_str

        routine = Routine(name)
        routine.symbol_table._argument_list.append(arg)
        routine.symbol_table.add(arg)

        container.addchild(routine)

        if datatype in scalar_datatypes:
            key = "scalar"
        else:
            key = dim_str
        routine_symbols[(datatype.intrinsic, datatype.precision, key)] \
            = routine.symbol_table.lookup_with_tag("own_routine_symbol")

    return container, routine_symbols

def _initialize_datasymbol(datatype):
    if datatype.intrinsic is ScalarType.Intrinsic.BOOLEAN:
        initial_value = True
    elif datatype.intrinsic is ScalarType.Intrinsic.INTEGER:
        initial_value = 1
    else:
        initial_value = 1.1

    if isinstance(datatype.precision, int):
        name = f"ref_{datatype.intrinsic.name}_{datatype.precision}"
    else:
        name = f"ref_{datatype.intrinsic.name}_{datatype.precision.name}"

    if isinstance(datatype, ArrayType):
        name += f"_{datatype.shape[0].upper.value}"
        if len(datatype.shape) == 2:
            name += f"x{datatype.shape[1].upper.value}"

    return DataSymbol(name, datatype, initial_value=initial_value,
                      interface=StaticInterface())

def test_intrinsiccall_datatypes(fortran_writer, tmpdir):
    """Test by compiling that the actual datatypes of written IntrinsicCall 
    nodes are those expected.
    """
    container = FileContainer("intrinsiccall")

    array_shape = [2,3]
    vector_shape = [3]

    container, routine_symbols = _initialize_type_checkers(container,
                                                           array_shape)
    (container, 
     other_routine_symbols) = _initialize_type_checkers(container,
                                                        vector_shape,
                                                        arrays_only=True)
    routine_symbols = {**routine_symbols, **other_routine_symbols}

    integer_kinds, real_kinds, boolean_kinds = _initialize_scalar_kinds()
    scalar_datatypes = integer_kinds + real_kinds

    array_mask_datatypes = [ArrayType(scalar_type, array_shape) 
                            for scalar_type in boolean_kinds]
    array_datatypes = [ArrayType(scalar_type, array_shape) 
                       for scalar_type in scalar_datatypes]
    vector_datatypes = [ArrayType(scalar_type, vector_shape) 
                        for scalar_type in scalar_datatypes]

    program = Routine("intrinsiccall", is_program=True)
    container.addchild(program)

    for datatype in scalar_datatypes + array_datatypes + vector_datatypes:
        sym = _initialize_datasymbol(datatype)

        program.symbol_table.add(sym)

        if datatype in scalar_datatypes:
            key = "scalar"
        elif datatype in array_datatypes:
            key = f"{array_shape[0]}x{array_shape[1]}"
        else:
            key = f"{vector_shape[0]}"

        # RANDOM_NUMBER, HUGE and TINY only take REAL args
        if datatype.intrinsic is ScalarType.Intrinsic.REAL \
            and datatype in scalar_datatypes:
            for func in (IntrinsicCall.Intrinsic.HUGE,
                         IntrinsicCall.Intrinsic.TINY):
                call = IntrinsicCall.create(func, [Reference(sym)])

                check = Call.create(routine_symbols[(call.datatype.intrinsic,
                                                     call.datatype.precision,
                                                     key)],
                                    [call])
                program.addchild(check)

        # MINVAL, MAXVAL, SUM : matrix/vector => scalar
        if datatype in array_datatypes + vector_datatypes:
            for func in (IntrinsicCall.Intrinsic.MINVAL,
                        IntrinsicCall.Intrinsic.MAXVAL,
                        IntrinsicCall.Intrinsic.SUM):
                call = IntrinsicCall.create(func, [Reference(sym)])

                assert isinstance(call.datatype, ScalarType)
                assert (call.datatype.intrinsic is datatype.intrinsic)
                assert (call.datatype.precision is datatype.precision)

                check = Call.create(routine_symbols[(call.datatype.intrinsic,
                                                        call.datatype.precision,
                                                        "scalar")], 
                                    [call])
                program.addchild(check)

        # MINVAL, MAXVAL, SUM : (vector, 1) => scalar
        if datatype in vector_datatypes:
            for func in (IntrinsicCall.Intrinsic.MINVAL,
                         IntrinsicCall.Intrinsic.MAXVAL,
                         IntrinsicCall.Intrinsic.SUM):
                call = IntrinsicCall.create(func, [Reference(sym),
                                                   ("dim",
                                                    Literal("1", 
                                                            INTEGER_TYPE))])

                assert isinstance(call.datatype, ScalarType)
                assert (call.datatype.intrinsic is datatype.intrinsic)
                assert (call.datatype.precision is datatype.precision)

                check = Call.create(routine_symbols[(call.datatype.intrinsic,
                                                     call.datatype.precision,
                                                     "scalar")], 
                                    [call])
                program.addchild(check)

        # MINVAL, MAXVAL, SUM : (matrix_ixj, 1) => vector_j
        if datatype in array_datatypes:
            for func in (IntrinsicCall.Intrinsic.MINVAL,
                         IntrinsicCall.Intrinsic.MAXVAL,
                         IntrinsicCall.Intrinsic.SUM):
                call = IntrinsicCall.create(func, [Reference(sym),
                                                   ("dim",
                                                    Literal("1", 
                                                            INTEGER_TYPE))])

                assert isinstance(call.datatype, ArrayType)
                assert (call.datatype.shape[0].upper.value
                        == str(array_shape[1]))
                assert (call.datatype.intrinsic is datatype.intrinsic)
                assert (call.datatype.precision is datatype.precision)

                check = Call.create(routine_symbols[(call.datatype.intrinsic,
                                                     call.datatype.precision,
                                                     "3")], 
                                    [call])
                program.addchild(check)

        # MINVAL, MAXVAL, SUM : (matrix_ixj, mask_ixj) => scalar
        if datatype in array_datatypes:
            new_sym = _initialize_datasymbol(array_mask_datatypes[0])
            mask = DataSymbol(new_sym.name + "_mask", INTEGER_TYPE)
            mask.copy_properties(new_sym)
            if mask.name not in program.symbol_table:
                program.symbol_table.add(mask)

            for func in (IntrinsicCall.Intrinsic.MINVAL,
                         IntrinsicCall.Intrinsic.MAXVAL,
                         IntrinsicCall.Intrinsic.SUM):
                call = IntrinsicCall.create(func, [Reference(sym),
                                                   ("mask", Reference(mask))])

                assert isinstance(call.datatype, ScalarType)
                assert (call.datatype.intrinsic is datatype.intrinsic)
                assert (call.datatype.precision is datatype.precision)

                check = Call.create(routine_symbols[(call.datatype.intrinsic,
                                                     call.datatype.precision,
                                                     "scalar")], 
                                    [call])
                program.addchild(check)

        # MINVAL, MAXVAL, SUM : (matrix_ixj, 1, mask_ixj) => vector_j
        if datatype in array_datatypes:
            new_sym = _initialize_datasymbol(array_mask_datatypes[0])
            mask = DataSymbol(new_sym.name + "_mask", INTEGER_TYPE)
            mask.copy_properties(new_sym)
            if mask.name not in program.symbol_table:
                program.symbol_table.add(mask)

            for func in (IntrinsicCall.Intrinsic.MINVAL,
                         IntrinsicCall.Intrinsic.MAXVAL,
                         IntrinsicCall.Intrinsic.SUM):
                call = IntrinsicCall.create(func, [Reference(sym),
                                                   ("dim",
                                                    Literal("1", INTEGER_TYPE)),
                                                   ("mask", Reference(mask))])

                assert isinstance(call.datatype, ArrayType)
                assert (call.datatype.shape[0].upper.value
                        == str(array_shape[1]))
                assert (call.datatype.intrinsic is datatype.intrinsic)
                assert (call.datatype.precision is datatype.precision)

                check = Call.create(routine_symbols[(call.datatype.intrinsic,
                                                     call.datatype.precision,
                                                     "3")], 
                                    [call])
                program.addchild(check)

        # CEIL, FLOOR, NINT : REAL => default INTEGER
        if datatype.intrinsic is ScalarType.Intrinsic.REAL:
            for op in (IntrinsicCall.Intrinsic.CEILING,
                       IntrinsicCall.Intrinsic.FLOOR,
                       IntrinsicCall.Intrinsic.NINT):

                operation = IntrinsicCall.create(op, [Reference(sym)])

                assert (operation.datatype.intrinsic
                        is ScalarType.Intrinsic.INTEGER)
                assert (
                    operation.datatype.precision
                    is ScalarType.Precision.UNDEFINED)
                assert (type(operation.datatype) is type(datatype))
                if datatype in array_datatypes:
                    assert (operation.datatype.shape == datatype.shape)

                check = Call.create(
                    routine_symbols[(operation.datatype.intrinsic,
                                     operation.datatype.precision,
                                     key)],
                    [operation])

                program.addchild(check)

        # INT :
        # - INTEGER, REAL => default INTEGER
        if datatype.intrinsic in (ScalarType.Intrinsic.INTEGER,
                                  ScalarType.Intrinsic.REAL):
            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.INT,
                                             [Reference(sym)])

            assert (operation.datatype.intrinsic is
                    ScalarType.Intrinsic.INTEGER)
            assert (operation.datatype.precision is
                    ScalarType.Precision.UNDEFINED)
            assert (type(operation.datatype) is type(datatype))
            if datatype in array_datatypes:
                assert (operation.datatype.shape == datatype.shape)

            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                 operation.datatype.precision,
                                 key)],
                [operation])
            program.addchild(check)

        # ABS:
        # - INTEGER*k => INTEGER*k
        # - REAL*k => REAL*k
        if datatype.intrinsic in (ScalarType.Intrinsic.INTEGER,
                                  ScalarType.Intrinsic.REAL):
            op = IntrinsicCall.Intrinsic.ABS

            operation = IntrinsicCall.create(op, [Reference(sym)])

            assert (operation.datatype.intrinsic is datatype.intrinsic)
            assert (operation.datatype.precision is datatype.precision)
            assert (type(operation.datatype) is type(datatype))
            if datatype in array_datatypes:
                assert (operation.datatype.shape == datatype.shape)

            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                    operation.datatype.precision,
                                    key)],
                [operation])

            program.addchild(check)

        # SQRT, EXP, LOG, LOG10, COS, SIN, TAN, ACOS, ASIN, ATAN :
        # REAL*k => REAL*k
        if datatype.intrinsic is ScalarType.Intrinsic.REAL:
            for op in (IntrinsicCall.Intrinsic.SQRT,
                       IntrinsicCall.Intrinsic.EXP,
                       IntrinsicCall.Intrinsic.LOG,
                       IntrinsicCall.Intrinsic.LOG10,
                       IntrinsicCall.Intrinsic.COS,
                       IntrinsicCall.Intrinsic.SIN,
                       IntrinsicCall.Intrinsic.TAN,
                       IntrinsicCall.Intrinsic.ACOS,
                       IntrinsicCall.Intrinsic.ASIN,
                       IntrinsicCall.Intrinsic.ATAN):

                operation = IntrinsicCall.create(op, [Reference(sym)])

                assert (operation.datatype.intrinsic is datatype.intrinsic)
                assert (operation.datatype.precision is datatype.precision)
                assert (type(operation.datatype) is type(datatype))
                if datatype in array_datatypes:
                    assert (operation.datatype.shape == datatype.shape)

                check = Call.create(
                    routine_symbols[(operation.datatype.intrinsic,
                                     operation.datatype.precision,
                                     key)],
                    [operation])

                program.addchild(check)

        # NOT : LOGICAL*K => LOGICAL*K
        if datatype.intrinsic is ScalarType.Intrinsic.BOOLEAN:
            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.NOT,
                                             [Reference(sym)])

            assert (operation.datatype.intrinsic is datatype.intrinsic)
            assert (operation.datatype.precision is datatype.precision)
            assert (type(operation.datatype) is type(datatype))
            if datatype in array_datatypes:
                assert (operation.datatype.shape == datatype.shape)

            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                 operation.datatype.precision,
                                 key)],
                [operation])

            program.addchild(check)

        # TRANSPOSE : type_kind_ixj => type_kind_jxi
        if datatype in array_datatypes:
            transpose_datatype = ArrayType(ScalarType(
                datatype.intrinsic, datatype.precision), datatype.shape[::-1])
            transpose_sym = _initialize_datasymbol(transpose_datatype)

            new_sym = DataSymbol(transpose_sym.name +
                                 "_T", INTEGER_SINGLE_TYPE)
            new_sym.copy_properties(transpose_sym)

            program.symbol_table.add(new_sym)

            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.TRANSPOSE,
                                             [Reference(new_sym)])

            assert (operation.datatype.intrinsic is datatype.intrinsic)
            assert (operation.datatype.precision is datatype.precision)
            assert (type(operation.datatype) is type(datatype))
            assert (operation.datatype.shape == datatype.shape)

            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                 operation.datatype.precision,
                                 key)],
                [operation])

            program.addchild(check)

    source = fortran_writer(container)
    Compile(tmpdir).string_compiles(source)

    container = FileContainer("binaryoperation")

    array_shape = [2, 3]
    vector_shape = [3]

    container, routine_symbols = _initialize_type_checkers(
        container, array_shape)
    # For 2x3 @ 3x2
    container, other_routine_symbols = _initialize_type_checkers(
        container, [array_shape[0], array_shape[0]], arrays_only=True)
    routine_symbols = {**routine_symbols, **other_routine_symbols}
    # For 2x3 @ 3
    container, other_routine_symbols = _initialize_type_checkers(
        container, [array_shape[0]], arrays_only=True)
    routine_symbols = {**routine_symbols, **other_routine_symbols}

    integer_kinds, real_kinds, boolean_kinds = _initialize_scalar_kinds()
    scalar_datatypes = integer_kinds + real_kinds + boolean_kinds

    array_datatypes1 = [ArrayType(scalar_type, array_shape)
                        for scalar_type in scalar_datatypes]
    vector_datatypes1 = [ArrayType(scalar_type, vector_shape)
                         for scalar_type in scalar_datatypes]
    # Transposed matrices
    array_datatypes2 = [ArrayType(scalar_type, array_shape[::-1])
                        for scalar_type in scalar_datatypes]
    vector_datatypes2 = [ArrayType(scalar_type, vector_shape)
                         for scalar_type in scalar_datatypes]

    datatypes1 = scalar_datatypes + array_datatypes1 + vector_datatypes1
    datatypes2 = scalar_datatypes + array_datatypes2 + vector_datatypes2

    program = Routine("binaryoperation", is_program=True)
    container.addchild(program)

    for datatype1, datatype2 in product(datatypes1, datatypes2):
        sym1 = _initialize_datasymbol(datatype1)
        sym2 = _initialize_datasymbol(datatype2)

        for sym in (sym1, sym2):
            if sym.name not in program.symbol_table:
                program.symbol_table.add(sym)

        if datatype1 in scalar_datatypes:
            key1 = "scalar"
        elif datatype1 in array_datatypes1:
            key1 = f"{array_shape[0]}x{array_shape[1]}"
        else:
            key1 = f"{vector_shape[0]}"

        if datatype2 in scalar_datatypes:
            key2 = "scalar"
        elif datatype2 in array_datatypes2:
            key2 = f"{array_shape[1]}x{array_shape[0]}"
        else:
            key2 = f"{vector_shape[0]}"

        # SIZE, LBOUND, UBOUND : (array, INTEGER) => default INTEGER
        if datatype1 in array_datatypes1 \
            and (datatype2 in scalar_datatypes
                 and datatype2.intrinsic is ScalarType.Intrinsic.INTEGER):
            for op in (IntrinsicCall.Intrinsic.SIZE,
                       IntrinsicCall.Intrinsic.LBOUND,
                       IntrinsicCall.Intrinsic.UBOUND):

                operation = IntrinsicCall.create(
                    op, [Reference(sym1), Reference(sym2)])

                assert (operation.datatype.intrinsic 
                        is ScalarType.Intrinsic.INTEGER)
                assert (
                    operation.datatype.precision 
                    is ScalarType.Precision.UNDEFINED)
                assert isinstance(operation.datatype, ScalarType)

                check = Call.create(
                    routine_symbols[(operation.datatype.intrinsic,
                                     operation.datatype.precision,
                                     "scalar")],
                    [operation])

                program.addchild(check)

        # CAST : (arg, scalar_type_kind) => scalar_type_kind
        #if datatype2 in scalar_datatypes:
        #    operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.CAST,
        #                                     [Reference(sym1), Reference(sym2)])
        #
        #    assert (operation.datatype.intrinsic is datatype2.intrinsic)
        #    assert (operation.datatype.precision is datatype2.precision)
        #    assert isinstance(operation.datatype, ScalarType)
        #
        #    check = Call.create(
        #        routine_symbols[(operation.datatype.intrinsic,
        #                         operation.datatype.precision,
        #                         "scalar")],
        #        [operation])
        #
        #    program.addchild(check)

        # MATMUL : (array_ixj, array_jxk) => array_ixk
        if datatype1 in array_datatypes1 and datatype2 in array_datatypes2 \
            and not ((datatype1.intrinsic is ScalarType.Intrinsic.BOOLEAN)
                     ^ (datatype2.intrinsic is ScalarType.Intrinsic.BOOLEAN)):
            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.MATMUL,
                                             [Reference(sym1), Reference(sym2)])

            assert isinstance(operation.datatype, ArrayType)
            assert (operation.datatype.shape == [
                    datatype1.shape[0], datatype2.shape[1]])

            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                 operation.datatype.precision,
                                 f"{operation.datatype.shape[0].upper.value}"
                                 f"x"
                                 f"{operation.datatype.shape[1].upper.value}")],
                [operation])

            program.addchild(check)

        # MATMUL : (array_ixj, vector_j) => vector_i
        if datatype1 in array_datatypes1 and datatype2 in vector_datatypes2 \
            and not ((datatype1.intrinsic is ScalarType.Intrinsic.BOOLEAN)
                     ^ (datatype2.intrinsic is ScalarType.Intrinsic.BOOLEAN)):
            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.MATMUL,
                                             [Reference(sym1), Reference(sym2)])

            assert isinstance(operation.datatype, ArrayType)
            assert (operation.datatype.shape == [datatype1.shape[0]])

            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                 operation.datatype.precision,
                                 f"{operation.datatype.shape[0].upper.value}")],
                [operation])

            program.addchild(check)

        # DOT_PRODUCT : (vector_i, vector_i) => scalar
        if datatype1 in vector_datatypes1 and datatype2 in vector_datatypes2 \
            and not ((datatype1.intrinsic is ScalarType.Intrinsic.BOOLEAN)
                     ^ (datatype2.intrinsic is ScalarType.Intrinsic.BOOLEAN)):
            operation = IntrinsicCall.create(
                                        IntrinsicCall.Intrinsic.DOT_PRODUCT,
                                        [Reference(sym1), Reference(sym2)])

            assert isinstance(operation.datatype, ScalarType)

            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                 operation.datatype.precision,
                                 "scalar")],
                [operation])

            program.addchild(check)

        # REAL, INT : (arg, k) => REAL*k/INT*k
        # NOTE: no typecheck functions for vectors
        if datatype1 in scalar_datatypes + array_datatypes1 \
                and (datatype1.intrinsic is not ScalarType.Intrinsic.BOOLEAN):
            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.REAL,
                                               [Reference(sym1),
                                                Literal("4", 
                                                       INTEGER_SINGLE_TYPE)])
            assert (operation.datatype.intrinsic is ScalarType.Intrinsic.REAL)
            assert (operation.datatype.precision == 4)
            assert (type(operation.datatype) is type(datatype1))
            if datatype1 in array_datatypes1:
                assert (operation.datatype.shape == datatype1.shape)
            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                operation.datatype.precision,
                                key1)],
                [operation])
            program.addchild(check)

            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.REAL,
                                               [Reference(sym1),
                                                Literal("8",
                                                       INTEGER_SINGLE_TYPE)])
            assert (operation.datatype.intrinsic is ScalarType.Intrinsic.REAL)
            assert (operation.datatype.precision == 8)
            assert (type(operation.datatype) is type(datatype1))
            if datatype1 in array_datatypes1:
                assert (operation.datatype.shape == datatype1.shape)
            check = Call.create(
                routine_symbols[(operation.datatype.intrinsic,
                                operation.datatype.precision,
                                key1)],
                [operation])
            program.addchild(check)

            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.INT,
                                               [Reference(sym1),
                                                Literal("4",
                                                       INTEGER_SINGLE_TYPE)])
            assert (operation.datatype.intrinsic
                    is ScalarType.Intrinsic.INTEGER)
            assert (operation.datatype.precision == 4)
            assert (type(operation.datatype) is type(datatype1))
            if datatype1 in array_datatypes1:
                assert (operation.datatype.shape == datatype1.shape)
            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                operation.datatype.precision,
                                                key1)],
                                [operation])
            program.addchild(check)

            operation = IntrinsicCall.create(IntrinsicCall.Intrinsic.INT,
                                               [Reference(sym1),
                                                Literal("8",
                                                       INTEGER_SINGLE_TYPE)])
            assert (operation.datatype.intrinsic
                    is ScalarType.Intrinsic.INTEGER)
            assert (operation.datatype.precision == 8)
            assert (type(operation.datatype) is type(datatype1))
            if datatype1 in array_datatypes1:
                assert (operation.datatype.shape == datatype1.shape)
            check = Call.create(routine_symbols[(operation.datatype.intrinsic,
                                                operation.datatype.precision,
                                                key1)],
                                [operation])
            program.addchild(check)

    source = fortran_writer(container)
    Compile(tmpdir).string_compiles(source)

    container = FileContainer("intrinsiccall")

    container, routine_symbols = _initialize_type_checkers(container, [2, 3])

    # No booleans
    integer_kinds, real_kinds, _ = _initialize_scalar_kinds()
    scalar_datatypes = integer_kinds + real_kinds

    datatypes1 = scalar_datatypes
    datatypes2 = scalar_datatypes

    program = Routine("naryoperation", is_program=True)
    container.addchild(program)

    for datatype1, datatype2 in product(datatypes1, datatypes2):
        sym1 = _initialize_datasymbol(datatype1)
        sym2 = _initialize_datasymbol(datatype2)

        for sym in (sym1, sym2):
            if sym.name not in program.symbol_table:
                program.symbol_table.add(sym)

        if datatype1.intrinsic is datatype2.intrinsic:
            for op in (IntrinsicCall.Intrinsic.MAX,
                       IntrinsicCall.Intrinsic.MIN):

                operation = IntrinsicCall.create(
                    op, [Reference(sym1), Reference(sym2)])

                check = Call.create(
                    routine_symbols[(operation.datatype.intrinsic,
                                     operation.datatype.precision,
                                     "scalar")],
                    [operation])

                program.addchild(check)

    source = fortran_writer(container)
    Compile(tmpdir).string_compiles(source)

if __name__ == "__main__":
    from psyclone.psyir.backend.fortran import FortranWriter
    fortran_writer = FortranWriter()
    Compile.TEST_COMPILE = True
    test_intrinsiccall_datatypes(fortran_writer, None)

def test_create_positional_arguments_with_names():
    ''' Test the create method when given named positional arguments.'''
    sym = DataSymbol("my_array",
                     ArrayType(INTEGER_TYPE, [ArrayType.Extent.DEFERRED]))
    aref = ArrayReference.create(sym, [Literal("20", INTEGER_TYPE)])
    bref = ArrayReference.create(sym, [Literal("20", INTEGER_TYPE)])

    # All of these are valid
    intr = IntrinsicCall.create(IntrinsicCall.Intrinsic.DOT_PRODUCT,
                                [aref.copy(), bref.copy()])
    assert isinstance(intr, IntrinsicCall)
    assert intr.children[0] == aref
    assert intr.children[1] == bref
    assert intr.argument_names == [None, None]

    intr = IntrinsicCall.create(IntrinsicCall.Intrinsic.DOT_PRODUCT,
                                [aref.copy(), ("vector_b", bref.copy())])
    assert isinstance(intr, IntrinsicCall)
    assert intr.children[0] == aref
    assert intr.children[1] == bref
    assert intr.argument_names == [None, "vector_b"]

    intr = IntrinsicCall.create(IntrinsicCall.Intrinsic.DOT_PRODUCT,
                                [("vector_a", aref.copy()),
                                 ("vector_b", bref.copy())])
    assert isinstance(intr, IntrinsicCall)
    assert intr.children[0] == aref
    assert intr.children[1] == bref
    assert intr.argument_names == ["vector_a", "vector_b"]

    intr = IntrinsicCall.create(IntrinsicCall.Intrinsic.DOT_PRODUCT,
                                [("vector_b", bref.copy()),
                                 ("vector_a", aref.copy())])
    assert isinstance(intr, IntrinsicCall)
    assert intr.children[0] == bref
    assert intr.children[1] == aref
    assert intr.argument_names == ["vector_b", "vector_a"]


@pytest.mark.parametrize("operator", ["lbound", "ubound", "size"])
def test_reference_accesses_bounds(operator, fortran_reader):
    '''Test that the reference_accesses method behaves as expected when
    the reference is the first argument to either the lbound or ubound
    intrinsic as that is simply looking up the array bounds (therefore
    var_access_info should be empty) and when the reference is the
    second argument of either the lbound or ubound intrinsic (in which
    case the access should be a read).

    '''
    code = f'''module test
        contains
        subroutine tmp()
          real, dimension(:,:), allocatable:: a, b
          integer :: n
          n = {operator}(a, b(1,1))
        end subroutine tmp
        end module test'''
    psyir = fortran_reader.psyir_from_source(code)
    schedule = psyir.walk(Assignment)[0]

    # By default, the access to 'a' should not be reported as read,
    # but the access to b must be reported:
    vai = VariablesAccessInfo(schedule)
    assert str(vai) == "b: READ, n: WRITE"

    # When explicitly requested, the access to 'a' should be reported:
    vai = VariablesAccessInfo(schedule,
                              options={"COLLECT-ARRAY-SHAPE-READS": True})
    assert str(vai) == "a: READ, b: READ, n: WRITE"


def test_enumerator_name_matches_name_field():
    '''
    Test that the name given to every IntrinsicCall matches the
    corresponding name field in the IAttr namedtuple.
    '''
    for intrinsic_entry in IntrinsicCall.Intrinsic:
        assert intrinsic_entry._name_ == intrinsic_entry.name


def test_allocate_intrinsic(fortran_reader, fortran_writer):
    '''
    Test the ALLOCATE 'intrinsic'.
    '''
    code = '''
program test_prog
  implicit none
  integer :: ierr
  character(len=128) :: msg
  real, allocatable, dimension(:) :: arr1, arr2
  allocate(arr1(10), stat=ierr)
  allocate(arr2, mold=arr1)
  allocate(arr2, source=arr1, errmsg=msg)
end program test_prog
'''
    psyir = fortran_reader.psyir_from_source(code)
    assert len(psyir.walk(IntrinsicCall)) == 3
    result = fortran_writer(psyir).lower()
    assert "allocate(arr1(1:10), stat=ierr)" in result
    assert "allocate(arr2, mold=arr1)" in result
    assert "allocate(arr2, source=arr1, errmsg=msg)" in result


def test_deallocate_intrinsic(fortran_reader, fortran_writer):
    '''
    Test the DEALLOCATE 'intrinsic'.
    '''
    code = '''
program test_prog
  implicit none
  integer :: ierr
  real, allocatable, dimension(:) :: arr1
  deallocate(arr1)
  deallocate(arr1, stat=ierr)
end program test_prog
'''
    psyir = fortran_reader.psyir_from_source(code)
    assert len(psyir.walk(IntrinsicCall)) == 2
    result = fortran_writer(psyir).lower()
    assert "deallocate(arr1)" in result
    assert "deallocate(arr1, stat=ierr)" in result


def test_index_intrinsic(fortran_reader, fortran_writer):
    '''
    Test the INDEX intrinsic.
    '''
    code = '''
program test_prog
  implicit none
  character(len=10) :: clname
  integer :: ind1, ind2

  ind1 = INDEX( clname, '_', back = .TRUE. ) + 1
  ind2 = INDEX( clname, '.') - 1
  ind2 = INDEX( clname, '.', kind=4) - 1

end program test_prog
'''
    psyir = fortran_reader.psyir_from_source(code)
    assert len(psyir.walk(IntrinsicCall)) == 3
    result = fortran_writer(psyir).lower()
    assert "ind1 = index(clname, '_', back=.true.) + 1" in result
    assert "ind2 = index(clname, '.') - 1" in result
    assert "ind2 = index(clname, '.', kind=4) - 1" in result


def test_verify_intrinsic(fortran_reader, fortran_writer):
    '''
    Test the VERIFY intrinsic.
    '''
    code = '''
program test_prog
  implicit none
  character(len=10) :: clname
  integer :: ind1, ind2, idom, jpdom_local

  ind1 = 2
  ind2 = 5
  IF( VERIFY( clname(ind1:ind2), '0123456789' ) == 0 ) idom = jpdom_local
  IF( VERIFY( clname(ind1:ind2), '0123456789', back=.true. ) == 0 ) &
idom = jpdom_local
  IF( VERIFY( clname(ind1:ind2), '0123456789', kind=kind(1) ) == 0 ) &
idom = jpdom_local
  IF( VERIFY( clname(ind1:ind2), '0123456789', kind=kind(1), back=.true. ) &
== 0 ) idom = jpdom_local

end program test_prog
'''
    psyir = fortran_reader.psyir_from_source(code)
    # Should have 4 VERIFY and 2 KIND
    assert len(psyir.walk(IntrinsicCall)) == 6
    result = fortran_writer(psyir).lower()
    assert "if (verify(clname(ind1:ind2), '0123456789') == 0) then" in result
    assert ("if (verify(clname(ind1:ind2), '0123456789', back=.true.) "
            "== 0) then" in result)
    assert ("if (verify(clname(ind1:ind2), '0123456789', kind=kind(1)) "
            "== 0) then" in result)
    assert ("if (verify(clname(ind1:ind2), '0123456789', kind=kind(1), "
            "back=.true.) == 0) then" in result)
