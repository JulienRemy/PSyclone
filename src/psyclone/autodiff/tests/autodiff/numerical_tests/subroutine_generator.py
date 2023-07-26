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
# Author J. Remy, Inria

"""This file contains a generator of Fortran subroutines for automated numerical 
testing of PSyclone `autodiff`."""

from enum import Enum

from psyclone.psyir.nodes import Routine, Assignment, Reference, Literal, Call
from psyclone.psyir.symbols import DataSymbol, RoutineSymbol, DataType, REAL_DOUBLE_TYPE
from psyclone.psyir.symbols.interfaces import ArgumentInterface
from psyclone.psyir.backend.fortran import FortranWriter

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
    # assign_zero,
    sin,
    cos,
    square,
    div,
    exp,
    sign,
    own_routine_symbol,
)


class FortranSubroutineGenerator(object):
    def __init__(self, subroutine_name, default_datatype=REAL_DOUBLE_TYPE):
        self._check_str(subroutine_name)
        self._check_datatype(default_datatype)

        self._writer = FortranWriter()
        self._subroutine = Routine(subroutine_name, is_program=False)
        self._default_datatype = default_datatype

    @property
    def writer(self):
        return self._writer

    @property
    def subroutine(self):
        return self._subroutine

    @property
    def symbol_table(self):
        return self.subroutine.symbol_table

    @property
    def routine_symbol(self):
        return own_routine_symbol(self.subroutine)

    @property
    def arguments(self):
        return self.symbol_table.argument_list

    @property
    def default_datatype(self):
        return self._default_datatype

    def _check_str(self, string):
        if not isinstance(string, str):
            raise TypeError(
                f"'string' argument should be of type 'str' "
                f"but found '{type(string).__name__}'."
            )
        if string == "":
            raise ValueError("'string' argument should not be ''.")

    def _check_symbol_exists(self, name):
        self._check_str(name)

        try:
            self.subroutine.symbol_table.lookup(name)
        except KeyError:
            pass
        else:
            raise ValueError(
                f"Symbol table of subroutine named"
                f"'{self.subroutine.name}' already contains a "
                f"symbol named '{name}'."
            )

    def _check_datatype(self, datatype):
        if not isinstance(datatype, DataType):
            raise TypeError(
                f"'datatype' argument should be of type 'DataType' "
                f"but found '{type(datatype).__name__}'."
            )

    def _check_literal(self, literal):
        if not isinstance(literal, Literal):
            raise TypeError(
                f"'literal' argument should be of type 'Literal' "
                f"but found '{type(literal).__name__}'."
            )

    def new_variable(self, name, datatype=None, constant_value=None):
        self._check_symbol_exists(name)
        if datatype is not None:
            self._check_datatype(datatype)
        if constant_value is not None:
            self._check_literal(constant_value)

        if datatype:
            use_datatype = datatype
        else:
            use_datatype = self.default_datatype

        return self.symbol_table.new_symbol(
            name,
            symbol_type=DataSymbol,
            datatype=use_datatype,
            constant_value=constant_value,
        )

    def new_arg(self, name, datatype=None, access=ArgumentInterface.Access.UNKNOWN):
        if not isinstance(access, Enum) and access not in ArgumentInterface.Access:
            raise ValueError(
                f"'access' argument should be an "
                f"'ArgumentInterface.Access' but found '{access}' "
                f"of type '{type(access).__name__}'."
            )

        arg = self.new_variable(name, datatype)
        arg.interface = ArgumentInterface(access)
        self.symbol_table._argument_list.append(arg)
        return arg

    def new_in_arg(self, name, datatype=None):
        return self.new_arg(name, datatype, ArgumentInterface.Access.READ)

    def new_out_arg(self, name, datatype=None):
        return self.new_arg(name, datatype, ArgumentInterface.Access.WRITE)

    def new_inout_arg(self, name, datatype=None):
        return self.new_arg(name, datatype, ArgumentInterface.Access.READWRITE)

    def write(self):
        return self.writer(self.subroutine)
    
    def print(self):
        print(self.write())

    def _check_datasymbol(self, datasymbol):
        if not isinstance(datasymbol, DataSymbol):
            raise TypeError(
                f"'datasymbol' argument should be of type "
                f"'DataSymbol' but found {type(datasymbol).__name__}."
            )

    # def _check_routine_symbol(self, routine_symbol):
    #    if not isinstance(routine_symbol, RoutineSymbol):
    #        raise TypeError(
    #            f"'routine_symbol' argument should be of type "
    #            f"'RoutineSymbol' but found {type(routine_symbol).__name__}."
    #        )

    def new_assignment(self, lhs, rhs):
        #self._check_datasymbol(lhs)

        assignment = assign(lhs, rhs)
        self.subroutine.addchild(assignment)
        return assignment

    def new_call(self, subroutine_generator, arg_symbols):
        if not isinstance(subroutine_generator, FortranSubroutineGenerator):
            raise TypeError(
                f"'subroutine_generator' argument should be of type "
                f"'FortranSubroutineGenerator' but found "
                f"{type(subroutine_generator).__name__}."
            )
        if not isinstance(arg_symbols, list):
            raise TypeError(
                f"'arg_symbols' argument should be of type "
                f"'list' but found {type(arg_symbols).__name__}."
            )

        if len(arg_symbols) != len(subroutine_generator.arguments):
            raise ValueError(
                f"The length of the 'arg_symbols' argument doesn't "
                f"match that of the argument list of subroutine "
                f"{subroutine_generator.routine_symbol.name} "
                f"being generated by the 'subroutine_generator' "
                f"argument."
            )
        # TODO: this would need to be extended to Operation once its datatype is implemented
        for i, sym in enumerate(arg_symbols):
            if isinstance(sym, DataSymbol) and (
                sym.datatype is not subroutine_generator.arguments[i].datatype
            ):
                raise ValueError(
                    f"The datatype of {sym} doesn't match that "
                    f"of the argument in the called subroutine."
                )

        call_args = [datanode(sym) for sym in arg_symbols]

        call = Call.create(subroutine_generator.routine_symbol, call_args)
        self.subroutine.addchild(call)
        return call

if __name__ == "__main__":
    bar = FortranSubroutineGenerator("bar")
    a = bar.new_in_arg('a')

    foo = FortranSubroutineGenerator("foo")
    x = foo.new_in_arg("x")
    y = foo.new_out_arg("y")
    a = foo.new_variable("a")
    foo.new_assignment(y, x)
    foo.new_assignment(a, add(x, y))
    foo.new_call(bar, [x])
    foo.print()

