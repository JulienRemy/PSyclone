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

from psyclone.core import VariablesAccessInfo
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
from psyclone.autodiff.tapes import ADValueTape
from psyclone.autodiff.transformations import (
    ADReverseContainerTrans,
    ADReverseScheduleTrans,
    ADRoutineTrans
)


class ADReverseRoutineTrans(ADRoutineTrans):
    """A class for automatic differentation transformations of Routine nodes. 
    Requires an ADReverseContainerTrans instance as context, where the definitions of  \
    the routines called inside the one to be transformed can be found.
    Uses an ADReverseScheduleTrans internally.
    """

    _recording_prefix = ""
    _recording_postfix = "_rec"

    _returning_prefix = ""
    _returning_postfix = "_ret"

    _reversing_prefix = ""
    _reversing_postfix = "_rev"

    _routine_prefixes = (_recording_prefix, _returning_prefix, _reversing_prefix)
    _routine_postfixes = (_recording_postfix, _returning_postfix, _reversing_postfix)

    _number_of_schedules = ADReverseScheduleTrans._number_of_schedules
    _differential_prefix = ADReverseScheduleTrans._differential_prefix
    _differential_postfix = ADReverseScheduleTrans._differential_postfix
    _differential_table_index = ADReverseScheduleTrans._differential_table_index
    _operation_adjoint_name = ADReverseScheduleTrans._operation_adjoint_name
    _default_value_tape_datatype = ADReverseScheduleTrans._default_value_tape_datatype

    def __init__(self, container_trans):
        super().__init__(container_trans)

        self.schedule_trans = ADReverseScheduleTrans(container_trans)
        self.assignment_trans = self.schedule_trans.assignment_trans
        self.assignment_trans.routine_trans = self
        self.operation_trans = self.schedule_trans.operation_trans
        self.operation_trans.routine_trans = self
        self.call_trans = self.schedule_trans.call_trans
        self.call_trans.routine_trans = self

    @property
    def recording(self):
        """Returns the recording routine being generated.

        :return: recording routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self.transformed[0]

    @property
    def returning(self):
        """Returns the returning routine being generated.

        :return: returning routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self.transformed[1]

    @property
    def reversing(self):
        """Returns the reversing routine being generated.

        :return: reversing routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self.transformed[2]

    @property
    def recording_table(self):
        """Returns the symbol table of the recording routine being generated.

        :return: recording routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.recording.symbol_table

    @property
    def returning_table(self):
        """Returns the symbol table of the returning routine being generated.

        :return: returning routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.returning.symbol_table

    @property
    def reversing_table(self):
        """Returns the symbol table of the reversing routine being generated.

        :return: reversing routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.reversing.symbol_table

    @property
    def value_tape(self):
        """Returns the value_tape used by the transformation.

        :return: value_tape.
        :rtype: :py:class:`psyclone.autodiff.ADValueTape`
        """
        return self._value_tape

    @value_tape.setter
    def value_tape(self, value_tape):
        if not isinstance(value_tape, ADValueTape):
            raise TypeError(
                f"'value_tape' argument should be of "
                f"type 'ADValueTape' but found "
                f"'{type(value_tape).__name__}'."
            )
        self._value_tape = value_tape

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
        """
        super().validate(routine, dependent_vars, independent_vars, options)
        self.schedule_trans.validate(routine, dependent_vars, independent_vars, value_tape, options)

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

        self.routine = routine
        self.dependent_variables = dependent_vars
        self.independent_variables = independent_vars
        
        # Get the variables access information (to determine overwrites and taping)
        self.variables_info = VariablesAccessInfo(routine)

        # Tape for the transformation
        # - none provided, create one
        if value_tape is None:
            name = routine.name
            self.value_tape = ADValueTape(name, self._default_value_tape_datatype)
        # - use the provided one
        else:
            self.value_tape = value_tape

        # Add this transformation to the container_trans map
        # Do it before apply below or ordering is not from outer to inner routines
        self.container_trans.add_routine_trans(self)

        # Apply the ADReverseScheduleTrans
        schedules = self.schedule_trans.apply(routine, dependent_vars, independent_vars, self.value_tape, options)

        # Raise the transformed schedules to routines
        self.transformed = self.schedules_to_routines(schedules)
        # Get relevant attributes from the ScheduleTrans
        self.data_symbol_differential_map = self.schedule_trans.data_symbol_differential_map
        self.temp_symbols = self.schedule_trans.temp_symbols
        self.value_tape = self.schedule_trans.value_tape

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
            if arg in self.data_symbol_differential_map:
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
            adjoint_symbol = self.data_symbol_differential_map[symbol]
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

        variables_adjoints = list(self.data_symbol_differential_map.values())
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
        # symbol_names = [sym.name for sym in list(self.data_symbol_differential_map.keys())]
        # for var in self.independent_vars:
        #    index = symbol_names.index(var)
        #    symbol = list(self.data_symbol_differential_map.keys())[index]
        #    adjoint_symbol = self.data_symbol_differential_map[symbol]
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
