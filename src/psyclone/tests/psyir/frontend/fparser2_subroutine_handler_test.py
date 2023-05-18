# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2021-2023, Science and Technology Facilities Council.
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
# Authors: R. W. Ford, A. R. Porter, S. Siso and N. Nobre, STFC Daresbury Lab

'''Module containing pytest tests for the _subroutine_handler method
in the class Fparser2Reader. This handler deals with the translation
of the fparser2 Subroutine_Subprogram and Function_Subprogram constructs
to PSyIR.

'''
import pytest

from fparser.common.readfortran import FortranStringReader
from psyclone.errors import InternalError
from psyclone.psyir.frontend.fparser2 import (Fparser2Reader,
                                              TYPE_MAP_FROM_FORTRAN)
from psyclone.psyir.nodes import Container, Routine, CodeBlock, FileContainer
from psyclone.psyir.symbols import (DataSymbol, ScalarType, UnknownFortranType,
                                    RoutineSymbol)

IN_OUTS = []
# subroutine no declarations
IN_OUTS.append(
    (("subroutine sub1()\n"
      "end subroutine\n"),
     ("subroutine sub1()\n\n\n"
      "end subroutine sub1\n")))
# subroutine with symbols/declarations
IN_OUTS.append(
    (("subroutine sub1(a)\n"
      "real :: a\n"
      "end subroutine\n"),
     ("subroutine sub1(a)\n"
      "  real :: a\n\n\n"
      "end subroutine sub1\n")))
# subroutine with executable content
IN_OUTS.append(
    (("subroutine sub1()\n"
      "real :: a\n"
      "a=0.0\n"
      "end subroutine\n"),
     ("subroutine sub1()\n"
      "  real :: a\n\n"
      "  a = 0.0\n\n"
      "end subroutine sub1\n")))


@pytest.mark.parametrize("code,expected", IN_OUTS)
def test_subroutine_handler(parser, fortran_writer, code, expected):
    '''Test that subroutine_handler handles valid Fortran subroutines.'''

    processor = Fparser2Reader()
    reader = FortranStringReader(code)
    parse_tree = parser(reader)
    subroutine = parse_tree.children[0]
    psyir = processor._subroutine_handler(subroutine, None)
    # Check the expected PSyIR nodes are being created
    assert isinstance(psyir, Routine)
    assert psyir.parent is None
    result = fortran_writer(psyir)
    assert expected == result


def test_subroutine_implicit_args(parser):
    """Check that we raise the expected error when we encounter a
    subroutine argument without an explicit declaration.

    """
    code = '''
subroutine sub1(idx)
end subroutine'''
    processor = Fparser2Reader()
    reader = FortranStringReader(code)
    parse_tree = parser(reader)
    subroutine = parse_tree.children[0]
    with pytest.raises(InternalError) as err:
        _ = processor._subroutine_handler(subroutine, None)
    assert ("The argument list ['idx'] for routine 'sub1' does not match "
            "the variable declarations:" in str(err.value))


def test_subroutine_some_implicit_args(parser):
    """Check that we raise the expected error when we encounter a
    subroutine which has declarations but has omitted to declare one
    of its arguments.

    """
    code = '''
subroutine sub1(var, idx)
    real, intent(in) :: var
end subroutine'''
    processor = Fparser2Reader()
    reader = FortranStringReader(code)
    parse_tree = parser(reader)
    subroutine = parse_tree.children[0]
    with pytest.raises(InternalError) as err:
        _ = processor._subroutine_handler(subroutine, None)
    err_msg = str(err.value)
    assert "The argument list ['var', 'idx'] for routine 'sub1'" in err_msg
    assert "Could not find 'idx' in the Symbol Table" in err_msg


def test_function_handler(fortran_reader, fortran_writer):
    '''Test that subroutine_handler correctly handles a function defined
    within a module.

    '''
    code = (
        "module a\n"
        "contains\n"
        "  function my_func()\n"
        "    integer :: my_func\n"
        "    my_func = 1\n"
        "  end function my_func\n"
        "end module\n")
    expected = (
        "module a\n"
        "  implicit none\n"
        "  public\n\n"
        "  contains\n"
        "  function my_func()\n"
        "    integer :: my_func\n"
        "\n"
        "    my_func = 1\n"
        "\n"
        "  end function my_func\n"
        "\n"
        "end module a\n")
    psyir = fortran_reader.psyir_from_source(code)
    # Check PSyIR nodes are being created
    assert isinstance(psyir, FileContainer)
    container = psyir.children[0]
    assert isinstance(container, Container)
    # Check that an appropriate RoutineSymbol has been created.
    func_sym = container.symbol_table.lookup("my_func")
    assert isinstance(func_sym, RoutineSymbol)
    routines = psyir.walk(Routine)
    assert len(routines) == 1
    # Check that there's a DataSymbol of the same name inside the function.
    assert isinstance(routines[0].return_symbol, DataSymbol)
    assert routines[0].return_symbol.name == "my_func"
    assert (routines[0].return_symbol.datatype.intrinsic ==
            ScalarType.Intrinsic.INTEGER)
    assert (routines[0].symbol_table.lookup("my_func") is
            routines[0].return_symbol)
    assert psyir.parent is None
    result = fortran_writer(psyir)
    assert result == expected


@pytest.mark.parametrize("basic_type, rhs_val", [("real", "1.0"),
                                                 ("integer", "1"),
                                                 ("logical", ".false."),
                                                 ("character", "'b'")])
def test_function_type_prefix(fortran_reader, fortran_writer,
                              basic_type, rhs_val):
    '''
    Test the handler when the function definition has a type prefix but no
    result suffix. Includes test that handling is not case sensitive.

    '''
    code = (
        f"module a\n"
        f"contains\n"
        f"  {basic_type} function my_fUnc()\n"
        f"    my_Func = {rhs_val}\n"
        f"  end function my_func\n"
        f"end module\n")
    expected = (
        f"module a\n"
        f"  implicit none\n"
        f"  public\n\n"
        f"  contains\n"
        f"  function my_fUnc()\n"
        f"    {basic_type} :: my_fUnc\n"
        f"\n"
        f"    my_fUnc = {rhs_val}\n"
        f"\n"
        f"  end function my_fUnc\n"
        f"\n"
        f"end module a\n")
    psyir = fortran_reader.psyir_from_source(code)
    assert isinstance(psyir, FileContainer)
    module = psyir.children[0]
    assert isinstance(module, Container)
    routine = module.children[0]
    assert isinstance(routine, Routine)
    return_sym = routine.return_symbol
    assert isinstance(return_sym, DataSymbol)
    assert return_sym.datatype.intrinsic == TYPE_MAP_FROM_FORTRAN[basic_type]
    result = fortran_writer(psyir)
    assert result == expected
    # Also check that the "own_routine_symbol" tag is maintained
    assert routine.symbol_table.lookup_with_tag("own_routine_symbol") \
        is return_sym


FN1_IN = ("  function my_func() result(my_val)\n"
          "    real :: my_val\n"
          "    my_val = 1.0\n"
          "  end function my_func\n")
FN2_IN = ("  real function my_func() result(my_val)\n"
          "    my_val = 1.0\n"
          "  end function my_func\n")
FN3_IN = ("  real(wp) function my_func() result(my_val)\n"
          "    my_val = 1.0\n"
          "  end function my_func\n")
EXPECTED_FN_OUT = ("  function my_func() result(my_val)\n"
                   "    real :: my_val\n\n"
                   "    my_val = 1.0\n\n"
                   "  end function my_func\n")
EXPECTED_FN3_OUT = ("  function my_func() result(my_val)\n"
                    "    real(kind=wp) :: my_val\n\n"
                    "    my_val = 1.0\n\n"
                    "  end function my_func\n")


@pytest.mark.parametrize("code,expected",
                         [(FN1_IN, EXPECTED_FN_OUT),
                          (FN2_IN, EXPECTED_FN_OUT),
                          (FN3_IN, EXPECTED_FN3_OUT)])
def test_function_result_suffix(fortran_reader, fortran_writer,
                                code, expected):
    '''
    Test that we handle a Fortran function with the return value specified
    using the 'result()' suffix. We test when the type is specified by a
    declaration inside the function or by a type specifier in the function
    prefix.

    '''
    code = (
        f"module a\n"
        f"use kind_params, only: wp\n"
        f"contains\n"
        f"{code}end module\n")
    psyir = fortran_reader.psyir_from_source(code)
    # Check PSyIR nodes are being created
    assert isinstance(psyir, Container)
    routines = psyir.walk(Routine)
    assert len(routines) == 1
    assert (routines[0].return_symbol is
            routines[0].symbol_table.lookup("my_val"))
    result = fortran_writer(psyir)
    assert expected in result


def test_function_missing_return_type(fortran_reader):
    '''
    Test that we generate a CodeBlock for a Fortran function without an
    explicit declaration of its return type (i.e. if it's relying on Fortran's
    implicit typing).

    '''
    code = (
        "module a\n"
        "contains\n"
        "  function my_func()\n"
        "    my_func = 1.0\n"
        "  end function my_func\n"
        "end module\n")
    psyir = fortran_reader.psyir_from_source(code)
    assert isinstance(psyir.children[0].children[0], CodeBlock)


def test_function_unsupported_type(fortran_reader):
    ''' Test that the frontend handles a function when the return type is not
    supported in the PSyIR. '''
    code = (
        "module a\n"
        "contains\n"
        "  function my_func()\n"
        "    complex :: my_func\n"
        "    my_func = CMPLX(1.0, 1.0)\n"
        "  end function my_func\n"
        "end module\n")
    psyir = fortran_reader.psyir_from_source(code)
    routine = psyir.children[0].children[0]
    assert isinstance(routine, Routine)
    assert routine.return_symbol.name == "my_func"
    assert isinstance(routine.return_symbol.datatype, UnknownFortranType)
    assert (routine.return_symbol.datatype.declaration.lower() ==
            "complex :: my_func")


def test_function_unsupported_derived_type(fortran_reader):
    ''' Test that the frontend handles a function when the return type is a
    derived type that is not supported in the PSyIR. '''
    code = (
        "module a\n"
        "contains\n"
        "  function my_func()\n"
        "    type :: my_type\n"
        "      integer :: flag\n"
        "    end type my_type\n"
        "    type(my_type), pointer :: my_func, var1\n"
        "    my_func => null()\n"
        "  end function my_func\n"
        "end module a\n")
    psyir = fortran_reader.psyir_from_source(code)
    routine = psyir.children[0].children[0]
    assert isinstance(routine, Routine)
    assert routine.return_symbol.name == "my_func"
    assert isinstance(routine.return_symbol.datatype, UnknownFortranType)
    assert (routine.return_symbol.datatype.declaration.lower() ==
            "type(my_type), pointer :: my_func")
    sym = routine.symbol_table.lookup("var1")
    assert isinstance(sym.datatype, UnknownFortranType)
    assert sym.datatype.declaration.lower() == "type(my_type), pointer :: var1"


@pytest.mark.parametrize("routine_type", ["function", "subroutine"])
def test_elemental_prefix(fortran_reader, routine_type):
    '''Check that the frontend correctly handles a routine with the 'elemental'
    prefix.'''
    code = (
        f"module a\n"
        f"contains\n"
        f"  elemental {routine_type} my_func()\n"
        f"    my_func = 1.0\n"
        f"  end {routine_type} my_func\n"
        f"end module\n")
    psyir = fortran_reader.psyir_from_source(code)
    routine = psyir.walk(Routine)[0]
    assert routine.is_elemental is True


def test_pure_prefix(fortran_reader):
    '''Check that the frontend correctly handles a routine with the 'pure'
    prefix.'''
    assert 0


@pytest.mark.parametrize("routine_type", ["function", "subroutine"])
@pytest.mark.parametrize("fn_prefix", ["recursive", "module"])
def test_unsupported_routine_prefix(fortran_reader, fn_prefix, routine_type):
    ''' Check that we get a CodeBlock if a Fortran routine has an unsupported
    prefix. '''
    code = (
        f"module a\n"
        f"contains\n"
        f"  {fn_prefix} {routine_type} my_func()\n"
        f"    my_func = 1.0\n"
        f"  end {routine_type} my_func\n"
        f"end module\n")
    psyir = fortran_reader.psyir_from_source(code)
    assert isinstance(psyir.children[0].children[0], CodeBlock)


def test_unsupported_char_len_function(fortran_reader):
    ''' Check that we get a CodeBlock if a Fortran function is of character
    type with a specified length. '''
    code = ("module a\n"
            "contains\n"
            "  character(len=2) function my_func()\n"
            "    my_func = 'aa'\n"
            "  end function my_func\n"
            "end module\n")
    psyir = fortran_reader.psyir_from_source(code)
    cblock = psyir.children[0].children[0]
    assert isinstance(cblock, CodeBlock)
    assert "LEN = 2" in str(cblock.get_ast_nodes[0])
