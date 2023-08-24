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

"""This module provides an abstract Transformation for automatic 
differentiation of PSyIR Routine nodes."""

from abc import ABCMeta, abstractmethod

from psyclone.psyir.nodes import (
    Routine,
    Call,
    Reference,
    ArrayReference,
    Literal,
)
from psyclone.psyir.symbols import (
    INTEGER_TYPE,
    SymbolTable,
    DataSymbol,
    ArrayType,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface, AutomaticInterface
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff import assign_zero, own_routine_symbol, assign, one
from psyclone.autodiff.transformations import ADContainerTrans, ADScopeTrans


class ADRoutineTrans(ADScopeTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of Routine nodes.
    Subclasses ADForwardRoutineTrans and ADReverseRoutineTrans are designed to use \
    respectively ADForwardScheduleTrans and ADReverseScheduleTrans internally.
    """

    _routine_prefixes = tuple()
    _routine_postfixes = tuple()

    _jacobian_prefix = ""
    _jacobian_postfix = "_jacobian"

    @property
    def routine(self):
        """Returns the routine node being transformed.

        :return: routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self._schedule

    @routine.setter
    def routine(self, routine):
        if not isinstance(routine, Routine):
            raise TypeError(
                f"'routine' argument should be of "
                f"type 'Routine' but found"
                f"'{type(routine).__name__}'."
            )

        self._schedule = routine

    @property
    def routine_table(self):
        """Returns the symbol table of the routine node being transformed.

        :return: symbol table.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.schedule_table

    @property
    def routine_symbol(self):
        """Returns the symbol of the routine node being transformed.

        :return: routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return self.routine_table.lookup_with_tag("own_routine_symbol")

    @property
    def transformed_symbols(self):
        """Returns the routine symbols of the  3 transformed routines as a \
        list, these being the recording routine, the returning routine and \
        the reversing routine.

        :return: list of transformed routine symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbol.RoutineSymbol`]
        """
        return [own_routine_symbol(routine) for routine in self.transformed]

    def validate(
        self, routine, dependent_vars, independent_vars, options=None
    ):
        """Validates the arguments of the `apply` method.

        :param routine: routine Node to the transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param dependent_vars: list of dependent variables names to be \
            differentiated.
        :type dependent_vars: `List[str]`
        :param independent_vars: list of independent variables names to \
            differentiate with respect to.
        :type independent_vars: `List[str]`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if routine is of the wrong type.
        :raises NotImplementedError: if routine is a program.
        :raises NotImplementedError: if routine is a function.
        :raises NotImplementedError: if routine contains a recursive call (to itself).
        :raises TransformationError: if the SymbolTable of routine doesn't \
            contain a symbol for each name in independent_var
        :raises TransformationError: if the SymbolTable of routine doesn't \
            contain a symbol for each name in dependent_var
        :raises TransformationError: if the argument list of routine doesn't \
            contain an argument of correct Access for each name in independent_var
        :raises TransformationError: if the argument list of routine doesn't \
            contain an argument of correct Access for each name in dependent_var
        """
        super().validate(routine, dependent_vars, independent_vars, options)

        if not isinstance(routine, Routine):
            raise TransformationError(
                f"'routine' argument should be of "
                f"type 'Routine' but found"
                f"'{type(routine).__name__}'."
            )
        # TODO: extend this to functions and programs
        # - functions won't be pure if modifying the value_tape!
        # - programs would only work for ONE dependent variable,
        #   and only by making a single program out of the recording and returning routines
        if routine.is_program:
            raise NotImplementedError(
                "'routine' argument is a program, "
                "this is not implemented yet. "
                "For now ADRoutineTrans only transforms "
                "Fortran subroutines."
            )
        if routine.return_symbol is not None:
            raise NotImplementedError(
                "'routine' argument is a function, "
                "this is not implemented yet."
                "For now ADRoutineTrans only transforms "
                "Fortran subroutines."
            )

        # Avoid dealing with recursive calls for now
        # TODO: these actually should work when using a joint reversal schedule
        # TODO: make the link between the routine and itself always be weak?
        for call in routine.walk(Call):
            if call.routine.name == routine.name:
                raise NotImplementedError(
                    f"Found a recursive Call inside "
                    f"Routine '{self.routine.name}'."
                    f"This is not implemented yet."
                )

        symbol_table = routine.symbol_table
        data_symbols = symbol_table.symbols
        symbol_names = [symbol.name for symbol in data_symbols]

        args = symbol_table.argument_list
        dependent_args_names = []
        independent_args_names = []
        for arg in args:
            if arg.interface.access == ArgumentInterface.Access.READ:
                independent_args_names.append(arg.name)
            elif arg.interface.access == ArgumentInterface.Access.WRITE:
                # This also includes routine.return_symbol
                # if it is a function
                dependent_args_names.append(arg.name)
            else:  # READWRITE or UNKNOWN
                independent_args_names.append(arg.name)
                dependent_args_names.append(arg.name)

        for var in dependent_vars:
            if var not in symbol_names:
                raise TransformationError(
                    f"Dependent variable name '{var}'"
                    f"was not found among the "
                    f"Symbol names in the Routine "
                    f"SymbolTable."
                )
            if var not in dependent_args_names:
                raise TransformationError(
                    f"Dependent variable name '{var}'"
                    f"was not found among the "
                    f"Routine arguments with "
                    f"ArgumentInterface.Access WRITE,"
                    f"READWRITE or UNKNOWN."
                )

        for var in independent_vars:
            if var not in symbol_names:
                raise TransformationError(
                    f"Independent variable name '{var}'"
                    f"was not found among the "
                    f"Symbol names in the Routine "
                    f"SymbolTable."
                )
            if var not in independent_args_names:
                raise TransformationError(
                    f"Inependent variable name '{var}'"
                    f"was not found among the "
                    f"Routine arguments with "
                    f"ArgumentInterface.Access READ,"
                    f"READWRITE or UNKNOWN."
                )

    def schedules_to_routines(self, schedules):
        """Creates Routines out of the transformed schedules, \
        by adding names to them.

        :param schedules: list of schedules.
        :type schedules: List[:py:class:`psyclone.psyir.nodes.Schedule`]

        :return: list of routines.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Routine`]
        """
        # Remove the 'own_routine_symbol' symbols from their tables
        tables = [schedule.symbol_table for schedule in schedules]
        for table in tables:
            table.remove(table.lookup_with_tag('own_routine_symbol'))

        # Generate the names of the routine using pre- and postfixes
        names = [
            prefix + self.routine.name + postfix
            for prefix, postfix in zip(self._routine_prefixes, self._routine_postfixes)
        ]

        # Create them, this adds a new RoutineSymbol correctly named
        # to their symbol tables
        routines = [
            Routine.create(
                name=name,
                symbol_table=schedule.symbol_table.detach(),
                children=[child.copy() for child in schedule.children],
                is_program=False,
                return_symbol_name=None,
            )
            for name, schedule in zip(names, schedules)
        ]

        return routines

    def jacobian_routine(self, mode, dependent_vars, independent_vars, options=None):
        """Creates the Jacobian routine using automatic \
        differentation for the transformed routine and lists of \
        dependent and independent variables names.
        Options:
        - bool 'verbose' : preceding comment for the routine.

        :param mode: mode to use. Can be either 'forward' or 'reverse'.
        :type mode: str
        :param dependent_vars: list of dependent variables names to be \
            differentiated.
        :type dependent_vars: `List[str]`
        :param independent_vars: list of independent variables names to \
            differentiate with respect to.
        :type independent_vars: `List[str]`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if mode is of the wrong type.
        :raises ValueError: if mode is neither 'forward' nor 'reverse'.
        :raises TypeError: if dependent_vars is of the wrong type.
        :raises TypeError: if at least one element of dependent_vars is \
            of the wrong type.
        :raises TypeError: if independent_vars is of the wrong type.
        :raises TypeError: if at least one element of independent_vars is \
            of the wrong type.
        :raises ValueError: if at least one element of dependent_vars is \
            not in self.dependent_variables, so was not used in transforming \
            the routine.
        :raises ValueError: if at least one element of independent_vars is \
            not in self.independent_variables, so was not used in transforming \
            the routine.

        :return: the routine computing the Jacobian.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        if not isinstance(mode, str):
            raise TypeError(
                f"'mode' argument should be of "
                f"type 'str' but found"
                f"'{type(mode).__name__}'."
            )
        if mode not in ("forward", "reverse"):
            raise ValueError(
                f"'mode' argument should either 'forward' or 'reverse' "
                f"but found '{mode}'."
            )
        if not isinstance(dependent_vars, list):
            raise TypeError(
                f"'dependent_vars' argument should be of "
                f"type 'list' but found"
                f"'{type(dependent_vars).__name__}'."
            )
        for var in dependent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'dependent_vars' argument should be of "
                    f"type 'List[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )
            if var not in self.dependent_variables:
                raise ValueError(
                    f"'dependent_vars' argument contains variable name {var} "
                    f"but it was not used as a dependent variable when "
                    f"transforming the routine."
                )

        if not isinstance(independent_vars, list):
            raise TypeError(
                f"'independent_vars' argument should be of "
                f"type 'list' but found"
                f"'{type(independent_vars).__name__}'."
            )
        for var in independent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'independent_vars' argument should be of "
                    f"type 'List[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )
            if var not in self.independent_variables:
                raise ValueError(
                    f"'independent_vars' argument contains variable name {var} "
                    f"but it was not used as a dependent variable when "
                    f"transforming the routine."
                )

        dependent_diff_symbols = []
        independent_diff_symbols = []

        if mode == "forward":
            transformed_table = self.transformed_tables[0]
            transformed_symbol = self.transformed_symbols[0]
        else:
            transformed_table = self.transformed_tables[2]  #reversing
            transformed_symbol = self.transformed_symbols[2]#reversing

        jacobian_routine = Routine(
            self._jacobian_prefix + self.routine.name + self._jacobian_postfix
        )
        symbol_table = jacobian_routine.symbol_table

        diff_map = dict()
        diff_symbol_names = []
        other_args = []

        # All differential variables (dependent & independent)
        # They are added as arguments of jacobian with the same intent
        for var in dependent_vars + independent_vars:
            # Get the symbol from the reversing routine
            sym = transformed_table.lookup(var)

            # Add it to the jacobian table and as argument (same intent)
            if sym not in symbol_table._argument_list:
                symbol_table.add(sym)
                symbol_table._argument_list.append(sym)

                # Get and copy the associated diff symbol
                diff_sym_copy = self.data_symbol_differential_map[sym].copy()
                # Switch it to non-argument interface
                diff_sym_copy.interface = AutomaticInterface()
                # Add to the value_tape
                symbol_table.add(diff_sym_copy)

                # Keep track of the diffs associated with the differential
                # variables names
                diff_map[var] = diff_sym_copy
                diff_symbol_names.extend([sym.name, diff_sym_copy.name])

        # Remaining arguments of the reversing routine need to be added too
        for sym in transformed_table.argument_list:
            if sym.name not in diff_symbol_names:
                symbol_table.add(sym)
                symbol_table._argument_list.append(sym)
                other_args.append(sym.name)

        # Lists of diff symbols to fill the jacobian
        for var in independent_vars:
            diff_sym = diff_map[var]
            independent_diff_symbols.append(diff_sym)
        for var in dependent_vars:
            diff_sym = diff_map[var]
            dependent_diff_symbols.append(diff_sym)

        # Some arguments of the jacobian routine with intent(inout) or unknown
        # in the reversing routine could be overwritten
        # Store and restore them as needed
        temp_assigns = []
        temp_restores = []
        for arg in symbol_table._argument_list:
            # Filter out the diffs
            # if arg not in self.data_symbol_differential_map.values():
            # (dependent_diff_symbols + independent_diff_symbols):
            if arg.interface.access in (
                ArgumentInterface.Access.READWRITE,
                ArgumentInterface.Access.UNKNOWN,
            ):
                temp = symbol_table.new_symbol(
                    "temp_" + arg.name, symbol_type=DataSymbol, datatype=arg.datatype
                )
                temp_assigns.append(assign(temp, arg))
                temp_restores.append(assign(arg, temp))
        self.add_children(jacobian_routine, temp_assigns)

        # Jacobian matrix symbol, with intent(out)
        rows = len(dependent_vars)
        cols = len(independent_vars)
        jacobian = symbol_table.new_symbol(
            "J_" + self.routine.name,
            symbol_type=DataSymbol,
            datatype=ArrayType(self._default_differential_datatype, [cols, rows]),
        )
        jacobian.interface = ArgumentInterface(ArgumentInterface.Access.WRITE)
        symbol_table._argument_list.append(jacobian)

        if mode == "forward":
            first_diffs = independent_diff_symbols
            second_diffs = dependent_diff_symbols
        else:
            first_diffs = dependent_diff_symbols
            second_diffs = independent_diff_symbols

        for first_dim, first_diff in enumerate(first_diffs):
            # Restore overwritten arguments of the jacobian routine
            # First first_dim => first call so no restores
            if first_dim != 0:
                self.add_children(
                    jacobian_routine, [rest.copy() for rest in temp_restores]
                )

            #first_dim + 1 to get the Fortran index
            first_dim_literal = Literal(str(first_dim + 1), INTEGER_TYPE)

            # Set the independent derivative/dependent adjoint for the row/column to 1.0
            jacobian_routine.addchild(assign(first_diff, one(first_diff.datatype)))

            # Set all other independent derivatives/dependent adjoints to 0.0
            for other_first_diff in first_diffs:
                if other_first_diff != first_diff:
                    jacobian_routine.addchild(assign_zero(other_first_diff))

            # Set all dependent derivatives/independent adjoints to 0.0
            # TODO: check the indep = dep case
            for second_diff in second_diffs:
                if second_diff != first_diff:
                    jacobian_routine.addchild(assign_zero(second_diff))

            # Create the argument list from the transformed one
            rev_args = [Reference(sym) for sym in transformed_table.argument_list]
            # Create the call, add it to the jacobian routine
            call = Call.create(transformed_symbol, rev_args)
            jacobian_routine.addchild(call)

            # Insert every independent diff at the right location
            # in the jacobian matrix
            for second_dim, second_diff in enumerate(second_diffs):
                # second_dim + 1 to get the Fortran index
                second_dim_literal = Literal(str(second_dim + 1), INTEGER_TYPE)

                if mode == "forward":
                    col_literal = first_dim_literal
                    row_literal = second_dim_literal
                else:
                    col_literal = second_dim_literal
                    row_literal = first_dim_literal

                jacobian_ref = ArrayReference.create(
                    jacobian, [col_literal.copy(), row_literal.copy()]
                )

                jacobian_routine.addchild(assign(jacobian_ref, second_diff))

        # Verbose description writes the dependent variables (columns),
        # the independent variables (rows), the other arguments to specify,
        # and the derivatives of the jacobian matrix as d_/d_
        verbose = self.unpack_option("verbose", options)

        if verbose:
            jacobian_routine.preceding_comment = (
                f"Independent variables as columns: {independent_vars}.\n! "
                + f"Dependent variables as rows: {dependent_vars}.\n! "
            )
            if len(other_args) != 0:
                jacobian_routine.preceding_comment += f"Also specify: {other_args}."
            for dep in dependent_vars:
                jacobian_routine.preceding_comment += "\n! "
                for indep in independent_vars:
                    jacobian_routine.preceding_comment += f"d{dep}/d{indep} "

        return jacobian_routine

    def add_to_argument_list(self, symbol_table, argument, after=None):
        """Adds the argument to the symbol table's argument list, if it has \
        the correct interface.
        The argument is added after another if 'after' is provided or \
        appended at the end otherwise.

        :param symbol_table: symbol table whose argument_list will be augmented.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        :param argument: argument symbol to add.
        :type argument: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param after: optional argument symbol after which to insert, defaults to None
        :type after: Union[:py:class:`psyclone.psyir.symbols.DataSymbol`, `NoneType`]
        :raises TypeError: if symbol_table is of the wrong type.
        :raises TypeError: if argument is of the wrong type.
        :raises TypeError: if argument's interface is not an ArgumentInterface.
        :raises TypeError: if after is of the wrong type.
        :raises ValueError: if after is not None and is not in the argument list.
        """
        if not isinstance(symbol_table, SymbolTable):
            raise TypeError(
                f"'symbol_table' argument should be of type "
                f"'SymbolTable' but found "
                f"'{type(symbol_table).__name__}'."
            )
        if not isinstance(argument, DataSymbol):
            raise TypeError(
                f"'argument' argument should be of type "
                f"'DataSymbol' but found "
                f"'{type(argument).__name__}'."
            )
        if not isinstance(argument.interface, ArgumentInterface):
            raise TypeError(
                f"'argument' argument's interface should be of type "
                f"'ArgumentInterface' but found "
                f"'{type(argument.interface).__name__}'."
            )
        if not isinstance(after, (DataSymbol, type(None))):
            raise TypeError(
                f"'after' argument should be of type "
                f"'DataSymbol' or 'NoneType' but found "
                f"'{type(after).__name__}'."
            )
        if (after is not None) and (after not in symbol_table._argument_list):
            raise ValueError(
                f"'after' argument DataSymbol named {after.name} "
                f"is not in the argument_list of symbol_table."
            )

        argument_list = symbol_table._argument_list

        if after is None:
            argument_list.append(argument)
        else:
            index = argument_list.index(after) + 1
            argument_list.insert(index, argument)
