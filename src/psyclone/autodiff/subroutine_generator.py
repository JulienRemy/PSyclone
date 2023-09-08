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
# Author: J. Remy, Université Grenoble Alpes, Inria

"""This file contains a generator of Fortran subroutines for automated numerical 
testing of PSyclone `autodiff`.
"""

from enum import Enum

from psyclone.psyir.nodes import Routine, Literal, Call, Reference, DataNode
from psyclone.psyir.symbols import DataSymbol, DataType, REAL_TYPE
from psyclone.psyir.symbols.interfaces import ArgumentInterface
from psyclone.psyir.backend.fortran import FortranWriter

from psyclone.autodiff.utils import (
    datanode,
    assign,
    own_routine_symbol,
)


class SubroutineGenerator(object):
    """This class is a Fortran subroutine generator using PSyIR nodes.
    It allows declaring and defining variables, arguments (with intents), \
    statements, operations, calls, etc. in a shorthand way, using helper \
    functions defined in `psyclone.autodiff.utils`.

    :param subroutine_name: name of the subroutine to generate.
    :type subroutine_name: Str
    :param default_datatype: default datatype to use in creating new arguments
        and variables, defaults to REAL_TYPE
    :type default_datatype: Optional[\
                                :py:class:`psyclone.psyir.symbols.DataType`]
    """

    # pylint: disable=useless-object-inheritance

    def __init__(self, subroutine_name, default_datatype=REAL_TYPE):
        self._check_str(subroutine_name)
        self._check_datatype(default_datatype)

        self._writer = FortranWriter()
        self._subroutine = Routine(subroutine_name, is_program=False)
        self._default_datatype = default_datatype

    @property
    def writer(self):
        """
        :return: the Fortran writer backend.
        :rtype: :py:class:`psyclone.psyir.backend.fortran.FortranWriter`
        """
        return self._writer

    @property
    def subroutine(self):
        """
        :return: the subroutine being generated.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self._subroutine

    @property
    def symbol_table(self):
        """
        :return: the symbol table of the subroutine being generated.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.subroutine.symbol_table

    @property
    def routine_symbol(self):
        """
        :return: the symbol of the subroutine being generated.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return own_routine_symbol(self.subroutine)

    @property
    def arguments(self):
        """
        :return: the argument list of the subroutine being generated.
        :rtype: List[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        """
        return self.symbol_table.argument_list

    @property
    def default_datatype(self):
        """
        :return: the default datatype of new variables and arguments.
        :rtype: :py:class:`psyclone.psyir.symbols.DataType`
        """
        return self._default_datatype

    def _check_str(self, string):
        """Checks that 'string' is a string and non-empty.

        :param string: string to test.
        :type string: Str

        :raises TypeError: if string is of the wrong type.
        :raises ValueError: if string is empty.
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'string' argument should be of type 'str' "
                f"but found '{type(string).__name__}'."
            )
        if string == "":
            raise ValueError("'string' argument should not be ''.")

    def _check_symbol_exists(self, name):
        """Checks if symbol with name 'name' exists in the symbol table.

        :param name: name of the symbol to check for.
        :type name: Str

        :raises ValueError: if the symbol table already has a symbol of that \
                            name.
        """
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
        """Check that the argument is a PSyIR `DataType`.

        :param datatype: object to check.
        :type datatype: :py:class:`psyclone.psyir.symbols.DataType`

        :raises TypeError: if not a DataType.
        """
        if not isinstance(datatype, DataType):
            raise TypeError(
                f"'datatype' argument should be of type 'DataType' "
                f"but found '{type(datatype).__name__}'."
            )

    def _check_literal(self, literal):
        """Check that the argument is a PSyIR `Literal`.

        :param literal: object to check.
        :type literal: :py:class:`psyclone.psyir.nodes.Literal`

        :raises TypeError: if not a Literal.
        """
        if not isinstance(literal, Literal):
            raise TypeError(
                f"'literal' argument should be of type 'Literal' "
                f"but found '{type(literal).__name__}'."
            )

    def new_variable(self, name, datatype=None, initial_value=None):
        """Create a new variable DataSymbol with the given name, datatype \
        and constant value.

        :param name: name of the new variable.
        :type name: str
        :param datatype: datatype of the new variable, defaults to None.
        :type datatype: Optional[\
                            Union[`NoneType`, 
                                  :py:class:`psyclone.psyir.symbols.DataType`]\
                        ]
        :param initial_value: initial value of the new variable, \
                              defaults to None.
        :type initial_value: Optional[\
                                Union[`NoneType`, 
                                      :py:class:`psyclone.psyir.nodes.Literal`]\
                             ]

        :raises TypeError: if name is not a string.
        :raises ValueError: if name is an empty string.
        :raises ValueError: if a symbol with this name already exists in the \
                            symbol table.
        :raises TypeError: if datatype is not a DataType or None.
        :raises TypeError: if constant value is not a Literal or None.

        :return: the symbol of the new variable.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        self._check_symbol_exists(name)
        if datatype is not None:
            self._check_datatype(datatype)
        if initial_value is not None:
            self._check_literal(initial_value)

        if datatype:
            use_datatype = datatype
        else:
            use_datatype = self.default_datatype

        return self.symbol_table.new_symbol(
            name,
            symbol_type=DataSymbol,
            datatype=use_datatype,
            initial_value=initial_value,
        )

    def new_arg(
        self, name, datatype=None, access=ArgumentInterface.Access.UNKNOWN
    ):
        """Create a new argument DataSymbol with the given name, datatype \
        and access.
        Create the new DataSymbol in the symbol table with the right \
        ArgumentInterface access and appends it to the argument list.

        :param name: name of the new variable.
        :type name: Str
        :param datatype: datatype of the new variable, defaults to None.
        :type datatype: Optional[\
                            Union[`NoneType`, 
                                  :py:class:`psyclone.psyir.symbols.DataType`]\
                        ]
        :param access: access of the new argument, \
                       defaults to `ArgumentInterface.Access.UNKNOWN`.
        :type access: Optional[element from \
        :py:class:`psyclone.psyir.symbols.interfaces.ArgumentInterface.Access`]

        :raises TypeError: if name is not a string.
        :raises ValueError: if name is an empty string.
        :raises ValueError: if a symbol with this name already exists in the \
                            symbol table.
        :raises TypeError: if access is not in ArgumentInterface.Access.

        :return: the symbol of the new argument.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        # Accessing private SymbolTable._argument_list to avoid property check.
        # pylint: disable=protected-access

        if (
            not isinstance(access, Enum)
            and access not in ArgumentInterface.Access
        ):
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
        """Create a new argument DataSymbol with the given name, datatype \
        and a READ ie. intent(in) access.
        Create the new DataSymbol in the symbol table with the right \
        ArgumentInterface access and appends it to the argument list.

        :param name: name of the new variable.
        :type name: Str
        :param datatype: datatype of the new variable, defaults to None.
        :type datatype: Optional[\
                            Union[`NoneType`, 
                                  :py:class:`psyclone.psyir.symbols.DataType`]\
                        ]

        :raises TypeError: if name is not a string.
        :raises ValueError: if name is an empty string.
        :raises ValueError: if a symbol with this name already exists in the \
                            symbol table.

        :return: the symbol of the new argument.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        return self.new_arg(name, datatype, ArgumentInterface.Access.READ)

    def new_out_arg(self, name, datatype=None):
        """Create a new argument DataSymbol with the given name, datatype \
        and a WRITE ie. intent(out) access.
        Create the new DataSymbol in the symbol table with the right \
        ArgumentInterface access and appends it to the argument list.

        :param name: name of the new variable.
        :type name: Str
        :param datatype: datatype of the new variable, defaults to None.
        :type datatype: Optional[\
                            Union[`NoneType`, 
                                  :py:class:`psyclone.psyir.symbols.DataType`]\
                        ]

        :raises TypeError: if name is not a string.
        :raises ValueError: if name is an empty string.
        :raises ValueError: if a symbol with this name already exists in the \
                            symbol table.

        :return: the symbol of the new argument.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        return self.new_arg(name, datatype, ArgumentInterface.Access.WRITE)

    def new_inout_arg(self, name, datatype=None):
        """Create a new argument DataSymbol with the given name, datatype \
        and a READWRITE ie. intent(inout) access.
        Create the new DataSymbol in the symbol table with the right \
        ArgumentInterface access and appends it to the argument list.

        :param name: name of the new variable.
        :type name: Str
        :param datatype: datatype of the new variable, defaults to None.
        :type datatype: Optional[\
                            Union[`NoneType`, 
                                  :py:class:`psyclone.psyir.symbols.DataType`]\
                        ]

        :raises TypeError: if name is not a string.
        :raises ValueError: if name is an empty string.
        :raises ValueError: if a symbol with this name already exists in the \
                            symbol table.

        :return: the symbol of the new argument.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        return self.new_arg(name, datatype, ArgumentInterface.Access.READWRITE)

    def write(self):
        """Use the Fortran backend to write the generated subroutine.

        :return: string produced by the Fortran backend.
        :rtype: Str
        """
        return self.writer(self.subroutine)

    def print(self):
        """Print the Fortran code generated by using the Fortran backend \
        on the generated subroutine.
        """
        print(self.write())

    def _check_datasymbol(self, datasymbol):
        """Check that the argument is a DataSymbol.

        :param datasymbol: object to check.
        :type datasymbol: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :raises TypeError: if not a DataSymbol.
        """
        if not isinstance(datasymbol, DataSymbol):
            raise TypeError(
                f"'datasymbol' argument should be of type "
                f"'DataSymbol' but found {type(datasymbol).__name__}."
            )

    def new_assignment(self, lhs, rhs):
        """Create a new assignment node in the subroutine being generated, \
        between the lhs and rhs arguments.
        Accepts data nodes or data symbols as arguments.

        :param lhs: LHS of Assignment.
        :type lhs: Union[:py:class:`psyclone.psyir.nodes.Reference`, \
                            :py:class:`psyclone.psyir.symbols.DataSymbol`]
        :param rhs: RHS of Assignment.
        :type rhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                        :py:class:`psyclone.psyir.symbols.DataSymbol`]

        :raises TypeError: if lhs or rhs is of the wrong type.
        :raises KeyError: if lhs or rhs is or contains a symbol that is \
                          not in the symbol table.

        :return: assignment node `lhs = rhs`.
        :rtype: :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        if not isinstance(lhs, (Reference, DataSymbol)):
            raise TypeError(
                f"'lhs' argument should be of type 'Reference' or "
                f"'DataSymbol' but found '{type(lhs).__name__}'."
            )
        if not isinstance(rhs, (DataNode, DataSymbol)):
            raise TypeError(
                f"'rhs' argument should be of type 'DataNode' or "
                f"'DataSymbol' but found '{type(rhs).__name__}'."
            )

        # This will raise a KeyError if the symbol doesn't exist in the
        # symbol table.
        for arg in (lhs, rhs):
            if isinstance(arg, DataSymbol):
                self.subroutine.symbol_table.lookup(arg.name)
            else:
                for ref in arg.walk(Reference):
                    self.subroutine.symbol_table.lookup(ref.name)

        assignment = assign(lhs, rhs)
        self.subroutine.addchild(assignment)
        return assignment

    def new_call(self, subroutine_generator, arg_symbols):
        """Create a new Call statement node in the subroutine being generated.

        :param subroutine_generator: subroutine generator of the subroutine 
            to call.
        :type subroutine_generator: \
            :py:class:`psyclone.autodiff.SubroutineGenerator`
        :param arg_symbols: list of datasymbol arguments for the call.
        :type arg_symbols: list[:py:class:`psyclone.psyir.symbols.DataSymbol`]

        :raises TypeError: if subroutine_generator is of the wrong type.
        :raises TypeError: if arg_symbols if of the wrong type.
        :raises TypeError: if an element of arg_symbols if of the wrong type.
        :raises ValueError: if the length of arg_symbols doesn't match the \
                            number of arguments of the called subroutine.
        :raises ValueError: if the datatype of an argument symbol in \
                            arg_symbols doesn't match that of the argument of \
                            the called subroutine.

        :return: the Call node.
        :rtype: :py:class:`psyclone.psyir.nodes.Call`
        """
        if not isinstance(subroutine_generator, SubroutineGenerator):
            raise TypeError(
                f"'subroutine_generator' argument should be of type "
                f"'SubroutineGenerator' but found "
                f"{type(subroutine_generator).__name__}."
            )
        if not isinstance(arg_symbols, list):
            raise TypeError(
                f"'arg_symbols' argument should be of type "
                f"'list[DataSymbol]' but found {type(arg_symbols).__name__}."
            )
        for sym in arg_symbols:
            if not isinstance(sym, DataSymbol):
                raise TypeError(
                    f"'arg_symbols' argument should be of type "
                    f"'list[DataSymbol]' but found an element of type "
                    f"{type(sym).__name__}."
                )

        if len(arg_symbols) != len(subroutine_generator.arguments):
            raise ValueError(
                f"The length of the 'arg_symbols' argument doesn't "
                f"match that of the argument list of subroutine "
                f"{subroutine_generator.routine_symbol.name} "
                f"being generated by the 'subroutine_generator' "
                f"argument."
            )
        # TODO: this would need to be extended to Operation once
        # its datatype is implemented
        for i, sym in enumerate(arg_symbols):
            if isinstance(sym, DataSymbol) and (
                sym.datatype != subroutine_generator.arguments[i].datatype
            ):
                raise ValueError(
                    f"The datatype of {sym} doesn't match that "
                    f"of the argument in the called subroutine."
                )

        call_args = [datanode(sym) for sym in arg_symbols]

        call = Call.create(subroutine_generator.routine_symbol, call_args)
        self.subroutine.addchild(call)
        return call

