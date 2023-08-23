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

"""This module provides a Transformation for reverse-mode automatic 
differentiation of PSyIR Routine nodes."""

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
from psyclone.autodiff.transformations import (
    ADReverseContainerTrans,
    ADReverseScheduleTrans,
)


class ADReverseRoutineTrans(ADReverseScheduleTrans):
    """A class for automatic differentation transformations of Routine nodes. 
    Requires an ADReverseContainerTrans instance as context, where the definitions of  \
    the routines called inside the one to be transformed can be found.
    Inherits from ADReverseScheduleTrans, which is used to transform the Schedule-like \
    part of the Routine.
    """

    _jacobian_prefix = ""
    _jacobian_suffix = "_jacobian"

    def __init__(self, container_trans):
        super().__init__(container_trans)

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
        return self.routine.symbol_table

    @property
    def routine_symbol(self):
        """Returns the symbol of the routine node being transformed.

        :return: routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return self.routine_table.lookup_with_tag("own_routine_symbol")

    @property
    def container_trans(self):
        """Returns the contextual ADReverseContainerTrans instance this \
        transformation was initialized with.

        :return: container transformation.
        :rtype: :py:class:`psyclone.autodiff.transformation.ADReverseContainerTrans`
        """
        return self._container_trans

    @container_trans.setter
    def container_trans(self, container_trans):
        if not isinstance(container_trans, ADReverseContainerTrans):
            raise TypeError(
                f"'container_trans' argument should be of "
                f"type 'ADReverseContainerTrans' but found"
                f"'{type(container_trans).__name__}'."
            )

        self._container_trans = container_trans

    @property
    def transformed_symbols(self):
        """Returns the routine symbols of the  3 transformed routines as a \
        list, these being the recording routine, the returning routine and \
        the reversing routine.

        :return: list of transformed routine symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbol.RoutineSymbol`]
        """
        return [own_routine_symbol(routine) for routine in self.transformed]

    @property
    def recording_symbol(self):
        """Returns the routine symbol of the recording routine being generated.

        :return: recording routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return own_routine_symbol(self.recording)

    @property
    def returning_symbol(self):
        """Returns the routine symbol of the returning routine being generated.

        :return: returning routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return own_routine_symbol(self.returning)

    @property
    def reversing_symbol(self):
        """Returns the routine symbol of the reversing routine being generated.

        :return: reversing routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return own_routine_symbol(self.reversing)

    def validate(
        self, routine, dependent_vars, independent_vars, value_tape=None, options=None
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
        :param value_tape: value tape to use to transform the schedule.
        :type value_tape: Optional[Union[NoneType, ADValueTape]]
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
        super().validate(routine, dependent_vars, independent_vars, value_tape, options)

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
                "For now ADReverseRoutineTrans only transforms "
                "Fortran subroutines."
            )
        if routine.return_symbol is not None:
            raise NotImplementedError(
                "'routine' argument is a function, "
                "this is not implemented yet."
                "For now ADReverseRoutineTrans only transforms "
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

    def apply(
        self, routine, dependent_vars, independent_vars, value_tape=None, options=None
    ):
        """Applies the transformation, generating the recording and returning \
        routines that correspond to automatic differentiation of this Routine \
        using reverse-mode.

        Options:
        - bool 'jacobian': whether to generate the Jacobian routine. Defaults \
            to False.
        - bool 'verbose' : toggles explanatory comments. Defaults to False.
        - bool 'simplify': True to apply simplifications after applying AD \
            transformations. Defaults to True.
        - int 'simplify_n_times': number of time to apply simplification \
            rules to BinaryOperation nodes. Defaults to 5.
        - bool 'inline_operation_adjoints': True to inline all possible \
            operation adjoints definitions. Defaults to True.

        :param routine: routine Node to the transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param dependent_vars: list of dependent variables names to be \
            differentiated.
        :type dependent_vars: `List[str]`
        :param independent_vars: list of independent variables names to \
            differentiate with respect to.
        :type independent_vars: `List[str]`
        :param value_tape: value tape to use to transform the schedule.
        :type value_tape: Optional[Union[NoneType, ADValueTape]]
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises NotImplementedError: if no transformation rule has yet been \
            implemented for one of the children of routine.

        :return: couple composed of the recording and returning Routines \
            that correspond to the transformation of this Routine.
        :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Routine`, \
                      :py:class:`psyclone.psyir.nodes.Routine`]
        """
        self.validate(routine, dependent_vars, independent_vars, value_tape, options)

        # Add this transformation to the container_trans map
        # Do it before apply below or ordering is not from outer to inner routines
        self.container_trans.add_routine_trans(self)

        # Apply the parent ADReverseScheduleTrans
        super().apply(routine, dependent_vars, independent_vars, value_tape, options)

        # Rename the tape
        # self.value_tape.name = self.routine.name

        # Raise the transformed schedules to routines
        self.transformed = self.raise_schedules_to_routines()

        # Add the transformed routines symbols to the container_trans map
        self.container_trans.add_transformed_routines(
            self.routine_symbol, self.transformed_symbols
        )

        # Tape all the values that are not written back to the parent schedule/scope
        self.value_tape_non_written_values(options)

        # add The value tape to the container_trans map
        self.container_trans.add_value_tape(self.routine_symbol, self.value_tape)

        # All dependent and independent variables names
        # list(set(...)) to avoid duplicates
        diff_variables = list(set(self.differential_variables))

        # Add the necessary adjoints as arguments of the returning routine
        self.add_adjoint_arguments(diff_variables, options)

        # Change the intents as needed
        # NOTE: this replaces the non-adjoint symbols in the returning symbol table
        # so it breaks the adjoint map...
        self.set_argument_accesses(options)

        # Add the value_tape as argument of both routines
        # iff it's actually used
        if self.value_tape.length != 0:
            self.add_value_tape_argument(options)

        # Add the assignments of 0 to other adjoints
        self.add_adjoint_assignments(options)

        # Combine the calls to recording and returning in reversing
        self.add_calls_to_reversing(options)

        # Add the three routines to the container
        for transformed in self.transformed:
            self.container_trans.container.addchild(transformed)

        jacobian = self.unpack_option("jacobian", options)

        if jacobian:
            jacobian_routine = self.jacobian_routine(
                dependent_vars, independent_vars, options
            )
            self.container_trans.container.addchild(jacobian_routine)

        return self.recording, self.returning, self.reversing

    def raise_schedules_to_routines(self):
        """Creates Routines out of the transformed schedules.

        :return: three transformed routines from schedules.
        :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Routine`]
        """
        # Remove the 'own_routine_symbol' symbols from their tables
        tables = self.transformed_tables
        for table in tables:
            table.remove(self.routine_symbol)

        # Generate the names of the routine using pre- and suffixes
        names = [
            prefix + self.routine.name + suffix
            for prefix, suffix in zip(self._schedule_prefixes, self._schedule_suffixes)
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
            for name, schedule in zip(names, self.transformed)
        ]

        return routines

#    def create_reversal_routines(self):
#        """Create the empty recording, returning and reversing Routines.
#
#        :return: all three routines as a list.
#        :rtype: List[:py:class:`psyclone.psyir.nodes.Routine`]
#        """
#        # Shallow copy the symbol table
#        tables = [self.routine_table.shallow_copy() for i in range(3)]
#        original_table = self.routine_table.shallow_copy().detach()
#        tables = [table.detach() for table in tables]
#        original_table.attach(self.routine)
#
#        # Generate the names of the routine using pre- and suffixes
#        names = [
#            prefix + self.routine.name + suffix
#            for prefix, suffix in zip(self._schedule_prefixes, self._schedule_suffixes)
#        ]
#
#        # Remove the 'own_routine_symbol' symbols from their tables
#        for table in tables:
#            table.remove(self.routine_symbol)
#
#        # Create them, this adds a new RoutineSymbol correctly named
#        # to their symbol tables
#        routines = [
#            Routine.create(
#                name=name,
#                symbol_table=table,
#                children=[],
#                is_program=False,
#                return_symbol_name=None,
#            )
#            for name, table in zip(names, tables)
#        ]
#
#        return routines

    def value_tape_non_written_values(self, options):
        """Record and restore the last values of non-argument variables \
        using the value_tape.
        Indeed these are not returned by the call but could affect the results.
        Consider eg.
        ```subroutine foo(a,b)
            implicit none
            double precision, intent(in) :: a
            double precision, intent(out) :: b
            double precision :: c

            c = 2
            ! here the prevalue 2 will be value_taped for c
            c = 4
            b = c * a 
            ! db/da is evidently 4, so the last value of c needs to be value_taped
        end subroutine foo```

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """
        # Values that are not written back to the calling routine
        # are all non-arguments variables.
        # NOTE: Arguments with intent(in) are not written back but cannot
        # be modified.
        for var in self.recording_table.datasymbols:
            if var not in self.recording_table.argument_list:
                # "fake" reference to value_tape the last value
                ref = Reference(var)

                # Record and restore
                value_tape_record = self.value_tape.record(ref)
                self.recording.addchild(value_tape_record)
                value_tape_restore = self.value_tape.restore(ref)
                self.returning.addchild(value_tape_restore, index=0)

    def set_argument_accesses(self, options):
        """Sets the intents of all non-adjoint arguments with original intents \
        different from intent(in) to intent(inout) in the returning routine.
        Indeed, all overwritable arguments are either recorded or returned by \
        the recording routine and restores or taken as argument by the \
        returning routine.
        Note that intent(in) in the returning routine would not allow value_tape \
        restores.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """
        # Every argument of the returning routine
        for arg in self.returning_table.argument_list:
            # Non-adjoint ones
            if arg in self.data_symbol_adjoint_map:
                # Only change intent(in)
                if arg.interface.access != ArgumentInterface.Access.READ:
                    returning_arg = DataSymbol(arg.name, arg.datatype)
                    # to intent(inout)
                    returning_arg.interface = ArgumentInterface(
                        ArgumentInterface.Access.READWRITE
                    )

                    # Swap in the returning table
                    # NOTE: SymbolTable.swap doesn't accept DataSymbols
                    name = self.returning_table._normalize(arg.name)
                    self.returning_table._symbols[name] = returning_arg

                    # Also swap in the returning argument list
                    index = self.returning_table._argument_list.index(arg)
                    self.returning_table._argument_list[index] = returning_arg

    def add_adjoint_arguments(self, diff_variables, options=None):
        """Add the adjoints of all differentiation variables \
        ie. dependent and independent ones \
        as intent(inout) arguments of the returning and reverting routines. \

        :param variables: list of (in)dependent variables names, unique.
        :type variables: List[str]
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if diff_variables is of the wrong type.
        """
        if not isinstance(diff_variables, list):
            raise TypeError(
                f"'diff_variables' argument should be of "
                f"type 'List[str]' but found"
                f"'{type(diff_variables).__name__}'."
            )
        for var in diff_variables:
            if not isinstance(var, str):
                raise TypeError(
                    f"'diff_variables' argument should be of "
                    f"type 'List[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )

        for var in diff_variables:
            # Get the symbol associated to the name, then the adjoint symbol
            symbol = self.returning_table.lookup(var, scope_limit=self.returning)

            # Use the original symbol (not the copy) to get its adjoint
            adjoint_symbol = self.data_symbol_adjoint_map[symbol]
            adjoint_symbol.interface = ArgumentInterface(
                ArgumentInterface.Access.READWRITE
            )

            # Insert the adjoint in the returning argument list
            # After the argument
            self.add_to_argument_list(
                self.returning_table, adjoint_symbol, after=symbol
            )

            # Insert the adjoint in the reverting argument list
            self.reversing_table.add(adjoint_symbol)
            self.add_to_argument_list(
                self.reversing_table, adjoint_symbol, after=symbol
            )

    def add_value_tape_argument(self, options=None):
        """Add the value_tape as argument of both the transformed routines.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """
        # Reshape the value_tape array to its correct number of elements
        # self.value_tape.reshape()

        # Get three symbols for the value_tape, one per routine
        symbols = [self.value_tape.symbol.copy() for i in range(3)]

        # Use the correct intents
        # intent(out) for the recording routine
        symbols[0].interface = ArgumentInterface(ArgumentInterface.Access.WRITE)
        # intent(in) for the returning routine
        symbols[1].interface = ArgumentInterface(ArgumentInterface.Access.READ)
        # The reversing routine declares the value_tape,
        # so the default AutomaticInterface is correct

        # Add the value_tape to all tables
        for table, symbol in zip(self.transformed_tables, symbols):
            table.add(symbol)

        # Append it to the arguments lists of the recording and returning routines only
        for table, symbol in zip(self.transformed_tables[:-1], symbols[:-1]):
            table._argument_list.append(symbol)
        # The value_tape is not an argument of the reversing routine

    def add_adjoint_assignments(self, options=None):
        """Assign the value 0 to every variable adjoint (ie. not temporary adjoint, \
        not operation adjoint) that is not an argument of the returning routine, \
        at its beginning.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """

        variables_adjoints = list(self.data_symbol_adjoint_map.values())
        # Reverse it so that the assignments are in variables appearance order
        # when inserting at index 0
        variables_adjoints.reverse()
        for adjoint_symbol in variables_adjoints:
            if adjoint_symbol not in self.returning_table._argument_list:
                assignment = assign_zero(adjoint_symbol)
                self.returning.addchild(assignment, index=0)

        ##################################
        # TODO: this should be optional
        # - it makes sense for the Routine being transformed,
        #       but not for those called inside
        # - when assigning 0, there is no point in adding the adjoint as argument
        ##################################
        # All independent variables adjoints
        # that are arguments of the returning routine
        # are assigned 0 at the beginning of the reversing routine
        # symbol_names = [sym.name for sym in list(self.data_symbol_adjoint_map.keys())]
        # for var in self.independent_vars:
        #    index = symbol_names.index(var)
        #    symbol = list(self.data_symbol_adjoint_map.keys())[index]
        #    adjoint_symbol = self.data_symbol_adjoint_map[symbol]
        #    if adjoint_symbol in self.returning_table._argument_list:
        #        assignment = assign_zero(adjoint_symbol)
        #        self.reversing.addchild(assignment)

    def add_calls_to_reversing(self, options=None):
        """Inserts two calls, to the recording and returning routines, in the \
        reversing routine.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """
        # Combine the recording and returning routines in the reversing routine
        call_rec = Call.create(
            own_routine_symbol(self.recording),
            [Reference(sym) for sym in self.recording_table.argument_list],
        )
        call_ret = Call.create(
            own_routine_symbol(self.returning),
            [Reference(sym) for sym in self.returning_table.argument_list],
        )
        self.reversing.addchild(call_rec)
        self.reversing.addchild(call_ret)

    # TODO: column major matrix filling would be better
    # TODO: test variables being both dependent and independent quite carefully...
    def jacobian_routine(self, dependent_vars, independent_vars, options=None):
        """Creates the Jacobian routine using reverse-mode automatic \
        differentation for the transformed routine and lists of \
        dependent and independent variables names.
        Options:
        - bool 'verbose' : preceding comment for the routine.

        :param dependent_vars: list of dependent variables names to be \
            differentiated.
        :type dependent_vars: `List[str]`
        :param independent_vars: list of independent variables names to \
            differentiate with respect to.
        :type independent_vars: `List[str]`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

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
        # :raises NotImplementedError: if dependent_vars and independent_vars \
        # intersect.
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

        # if len(set(dependent_vars + independent_vars)) \
        #    != len(dependent_vars + independent_vars):
        #    raise NotImplementedError("Generating the jacobian routine for "
        #                              "variables that are both dependent "
        #                              "and independent is not implemented yet.")

        dependent_adjoint_symbols = []
        independent_adjoint_symbols = []

        jacobian_routine = Routine(
            self._jacobian_prefix + self.routine.name + self._jacobian_suffix
        )
        symbol_table = jacobian_routine.symbol_table

        adjoint_map = dict()
        diff_symbol_names = []
        other_args = []

        # All differential variables (dependent & independent)
        # They are added as arguments of jacobian with the same intent
        for var in dependent_vars + independent_vars:
            # Get the symbol from the reversing routine
            sym = self.reversing_table.lookup(var)

            # Add it to the jacobian table and as argument (same intent)
            if sym not in symbol_table._argument_list:
                symbol_table.add(sym)
                symbol_table._argument_list.append(sym)

                # Get and copy the associated adjoint symbol
                adj_sym_copy = self.data_symbol_adjoint_map[sym].copy()
                # Switch it to non-argument interface
                adj_sym_copy.interface = AutomaticInterface()
                # Add to the value_tape
                symbol_table.add(adj_sym_copy)

                # Keep track of the adjoints associated with the differential
                # variables names
                adjoint_map[var] = adj_sym_copy
                diff_symbol_names.extend([sym.name, adj_sym_copy.name])

        # Remaining arguments of the reversing routine need to be added too
        for sym in self.reversing_table.argument_list:
            if sym.name not in diff_symbol_names:
                symbol_table.add(sym)
                symbol_table._argument_list.append(sym)
                other_args.append(sym.name)

        # Lists of adjoint symbols to fill the jacobian
        for var in independent_vars:
            adj_sym = adjoint_map[var]
            independent_adjoint_symbols.append(adj_sym)
        for var in dependent_vars:
            adj_sym = adjoint_map[var]
            dependent_adjoint_symbols.append(adj_sym)

        # Some arguments of the jacobian routine with intent(inout) or unknown
        # in the reversing routine could be overwritten
        # Store and restore them as needed
        temp_assigns = []
        temp_restores = []
        for arg in symbol_table._argument_list:
            # Filter out the adjoints
            # if arg not in self.data_symbol_adjoint_map.values():
            # (dependent_adjoint_symbols + independent_adjoint_symbols):
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
            datatype=ArrayType(self._default_adjoint_datatype, [cols, rows]),
        )
        jacobian.interface = ArgumentInterface(ArgumentInterface.Access.WRITE)
        symbol_table._argument_list.append(jacobian)

        for row, dep_adj in enumerate(dependent_adjoint_symbols):
            # Restore overwritten arguments of the jacobian routine
            # First row => first call so no restores
            if row != 0:
                self.add_children(
                    jacobian_routine, [rest.copy() for rest in temp_restores]
                )

            # row + 1 to get the Fortran index
            row_literal = Literal(str(row + 1), INTEGER_TYPE)

            # Set the dependent adjoint for the row to 1.0
            jacobian_routine.addchild(assign(dep_adj, one(dep_adj.datatype)))

            # Set all other dependent adjoints to 0.0
            for other_dep_adj in dependent_adjoint_symbols:
                if other_dep_adj != dep_adj:
                    jacobian_routine.addchild(assign_zero(other_dep_adj))

            # Set all independent adjoints to 0.0
            # TODO: check the indep = dep case
            for indep_adj in independent_adjoint_symbols:
                if indep_adj != dep_adj:
                    jacobian_routine.addchild(assign_zero(indep_adj))

            # Create the argument list from the reversing one
            rev_args = [Reference(sym) for sym in self.reversing_table.argument_list]
            # Create the call, add it to the jacobian routine
            call = Call.create(self.reversing_symbol, rev_args)
            jacobian_routine.addchild(call)

            # Insert every independent adjoint at the right location
            # in the jacobian matrix
            for col, indep_adj in enumerate(independent_adjoint_symbols):
                # col + 1 to get the Fortran index
                col_literal = Literal(str(col + 1), INTEGER_TYPE)
                jacobian_ref = ArrayReference.create(
                    jacobian, [col_literal, row_literal.copy()]
                )

                jacobian_routine.addchild(assign(jacobian_ref, indep_adj))

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
