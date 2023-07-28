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
differentiation of PSyIR Schedule nodes."""

from psyclone.psyir.nodes import (
    Routine,
    Assignment,
    Call,
    Reference,
    Node,
    Schedule,
)
from psyclone.psyir.symbols import (
    REAL_DOUBLE_TYPE,
    SymbolTable,
    DataSymbol,
    ScalarType,
    ArrayType,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface
from psyclone.core import VariablesAccessInfo
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff import simplify_node
from psyclone.autodiff.tapes import ADValueTape
from psyclone.autodiff.transformations import ADTrans, ADContainerTrans


class ADScheduleTrans(ADTrans):
    """A class for automatic differentation transformations of Schedule nodes.
    Requires an ADContainerTrans instance as context, where the definitions of
    the routines called inside the schedule to be transformed can be found.

    :param container_trans: ADContainerTrans context instance
    :type container_trans: :py:class:`psyclone.autodiff.transformations.ADContainerTrans`

    :raises TypeError: if the container_trans argument is of the wrong type.
    """

    _recording_prefix = ""
    _recording_suffix = "_rec"

    _returning_prefix = ""
    _returning_suffix = "_ret"

    _reversing_prefix = ""
    _reversing_suffix = "_rev"

    _schedule_prefixes = (_recording_prefix, _returning_prefix, _reversing_prefix)
    _schedule_suffixes = (_recording_suffix, _returning_suffix, _reversing_suffix)

    _adjoint_prefix = ""
    _adjoint_suffix = "_adj"

    _operation_adjoint_name = "op_adj"
    # _call_adjoint_name = "call_adj"

    _temp_name_prefix = "temp_"
    _temp_name_suffix = ""

    # TODO: correct datatype
    _default_value_tape_datatype = REAL_DOUBLE_TYPE

    # TODO: #001 use the dependent variable type and precision
    _default_adjoint_datatype = REAL_DOUBLE_TYPE

    def __init__(self, container_trans):
        if not isinstance(container_trans, ADContainerTrans):
            raise TypeError(
                f"'container_trans' argument should be of type "
                f"'ADContainerTrans' but found "
                f"'{type(container_trans).__name__}'."
            )
        # Transformation can only be applied once
        self._was_applied = False

        self.container_trans = container_trans

        # DataSymbol => adjoint DataSymbol
        self.data_symbol_adjoint_map = dict()

        # Lists of adjoint symbols for operations
        self.operation_adjoints = []

        # Symbols for temporary adjoints
        self.temp_symbols = []

        # Transformations need to know about the ADScheduleTrans calling them
        # to access the attributes defined above
        # Import here to avoid circular dependencies
        from psyclone.autodiff.transformations import (
            ADOperationTrans,
            ADAssignmentTrans,
            ADCallTrans,
        )

        # Initialize the sub transformations
        # self.adjoint_symbol_trans = ADAdjointSymbolTrans(self)
        self.assignment_trans = ADAssignmentTrans(self)
        self.operation_trans = ADOperationTrans(self)
        self.call_trans = ADCallTrans(self)

    # def new_call_trans(self):
    #    """Returns a new instance of ADCallTrans.
    #
    #    :return: _description_
    #    :rtype: _type_
    #    """
    #    from psyclone.autodiff.transformations import ADCallTrans
    #    return ADCallTrans(self)

    @property
    def schedule(self):
        """Returns the schedule node being transformed.

        :return: schedule.
        :rtype: :py:class:`psyclone.psyir.nodes.Schedule`
        """
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        if not isinstance(schedule, Schedule):
            raise TypeError(
                f"'schedule' argument should be of "
                f"type 'Schedule' but found"
                f"'{type(schedule).__name__}'."
            )

        self._schedule = schedule

    @property
    def dependent_variables(self):
        """Names of the dependent variables used in transforming this Schedule. \
        These are the variables being differentiated.

        :return: list of names.
        :rtype: `List[Str]`
        """
        return self._dependent_variables

    @dependent_variables.setter
    def dependent_variables(self, dependent_vars):
        if not isinstance(dependent_vars, list):
            raise TypeError(
                f"'dependent_vars' argument should be of "
                f"type 'List[Str]' but found "
                f"'{type(dependent_vars).__name__}'."
            )
        for var in dependent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'dependent_vars' argument should be of "
                    f"type 'List[Str]' but found "
                    f"an element of type "
                    f"'{type(var).__name__}'."
                )
        self._dependent_variables = dependent_vars

    @property
    def independent_variables(self):
        """Names of the independent variables used in transforming this Schedule. \
        These are the variables with respect to which we are differentiating.

        :return: list of names.
        :rtype: `List[Str]`
        """
        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, independent_vars):
        if not isinstance(independent_vars, list):
            raise TypeError(
                f"'independent_vars' argument should be of "
                f"type 'List[Str]' but found "
                f"'{type(independent_vars).__name__}'."
            )
        for var in independent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'independent_vars' argument should be of "
                    f"type 'List[Str]' but found "
                    f"an element of type "
                    f"'{type(var).__name__}'."
                )
        self._independent_variables = independent_vars

    @property
    def differential_variables(self):
        """Names of all differential variables, both dependent and independent.
        The list begins with independent variables. Names may not be unique in it.

        :return: list of all differential variables.
        :rtype: `List[Str]`
        """
        return self.dependent_variables + self.independent_variables

    @property
    def schedule_table(self):
        """Returns the symbol table of the schedule node being transformed.

        :return: symbol table.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.schedule.symbol_table

    @property
    def container_trans(self):
        """Returns the contextual ADContainerTrans instance this \
        transformation was initialized with.

        :return: container transformation.
        :rtype: :py:class:`psyclone.autodiff.transformation.ADContainerTrans`
        """
        return self._container_trans

    @container_trans.setter
    def container_trans(self, container_trans):
        if not isinstance(container_trans, ADContainerTrans):
            raise TypeError(
                f"'container_trans' argument should be of "
                f"type 'ADContainerTrans' but found"
                f"'{type(container_trans).__name__}'."
            )

        self._container_trans = container_trans

    @property
    def transformed(self):
        """Returns the 3 transformed schedules as a list, these being the \
        recording schedule, the returning schedule and the reversing schedule.

        :return: list of transformed schedules.
        :rtype: List[:py:class:`psyclone.psyir.node.Schedule`]
        """
        return self._transformed

    @transformed.setter
    def transformed(self, transformed):
        if not isinstance(transformed, list):
            raise TypeError(
                f"'transformed' argument should be of "
                f"type 'List[Schedule]' of length 3 but found "
                f"'{type(transformed).__name__}'."
            )
        if len(transformed) != 3:
            raise ValueError(
                f"'transformed' argument should be of "
                f"a list of length 3 but found length "
                f"{len(transformed)}."
            )
        for sym in transformed:
            if not isinstance(sym, Schedule):
                raise TypeError(
                    f"'transformed' argument should be of "
                    f"type 'List[Schedule]' of length 3 but found "
                    f"an element of type "
                    f"'{type(sym).__name__}'."
                )
        self._transformed = transformed

    @property
    def recording(self):
        """Returns the recording schedule being generated.

        :return: recording schedule.
        :rtype: :py:class:`psyclone.psyir.nodes.Schedule`
        """
        return self.transformed[0]

    @property
    def returning(self):
        """Returns the returning schedule being generated.

        :return: returning schedule.
        :rtype: :py:class:`psyclone.psyir.nodes.Schedule`
        """
        return self.transformed[1]

    @property
    def reversing(self):
        """Returns the reversing schedule being generated.

        :return: reversing schedule.
        :rtype: :py:class:`psyclone.psyir.nodes.Schedule`
        """
        return self.transformed[2]

    @property
    def transformed_tables(self):
        """Returns the symbol tables of the 3 transformed schedules as a \
        list, these being the recording schedule, the returning schedule and \
        the reversing schedule.

        :return: list of transformed schedule symbol tables.
        :rtype: List[:py:class:`psyclone.psyir.symbols.SymbolTable`]
        """
        return [schedule.symbol_table for schedule in self.transformed]

    @property
    def recording_table(self):
        """Returns the symbol table of the recording schedule being generated.

        :return: recording schedule symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.recording.symbol_table

    @property
    def returning_table(self):
        """Returns the symbol table of the returning schedule being generated.

        :return: returning schedule symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.returning.symbol_table

    @property
    def reversing_table(self):
        """Returns the symbol table of the reversing schedule being generated.

        :return: reversing schedule symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        return self.reversing.symbol_table

    @property
    def variables_info(self):
        """Returns the variables access information of the schedule being \
        transformed.

        :return: variables access information.
        :rtype: :py:class:`psyclone.core.VariablesAccessInfo`
        """
        return self._variables_info

    @variables_info.setter
    def variables_info(self, variables_info):
        if not isinstance(variables_info, VariablesAccessInfo):
            raise TypeError(
                f"'variables_info' argument should be of "
                f"type 'VariablesAccessInfo' but found "
                f"'{type(variables_info).__name__}'."
            )
        self._variables_info = variables_info

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

    def validate(
        self, schedule, dependent_vars, independent_vars, value_tape=None, options=None
    ):
        """Validates the arguments of the `apply` method.

        :param schedule: schedule Node to the transformed.
        :type schedule: :py:class:`psyclone.psyir.nodes.Schedule`
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

        :raises TransformationError: if the transformation has already been applied.
        :raises TransformationError: if schedule is of the wrong type.
        :raises TypeError: if dependent_vars is of the wrong type.
        :raises TypeError: if at least one element of dependent_vars is \
            of the wrong type.
        :raises TypeError: if independent_vars is of the wrong type.
        :raises TypeError: if at least one element of independent_vars is \
            of the wrong type.
        :raises TypeError: if value_tape is of the wrong type.
        """
        super().validate(schedule, options)

        if self._was_applied:
            raise TransformationError(
                "ADScheduleTrans instance can only be " "applied once."
            )

        if not isinstance(schedule, Schedule):
            raise TransformationError(
                f"'schedule' argument should be of "
                f"type 'Schedule' but found"
                f"'{type(schedule).__name__}'."
            )
        # TODO: extend this to functions and programs
        # - functions won't be pure if modifying the value_tape!
        # - programs would only work for ONE dependent variable,
        #   and only by making a single program out of the recording and returning schedules

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
        if not isinstance(value_tape, (ADValueTape, type(None))):
            raise TypeError(
                f"'value_tape' argument should be of "
                f"type 'ADValueTape' or 'NoneType' but found an element of type"
                f"'{type(value_tape).__name__}'."
            )

    def apply(
        self, schedule, dependent_vars, independent_vars, value_tape=None, options=None
    ):
        """Applies the transformation, generating the recording and returning \
        schedules that correspond to automatic differentiation of this Schedule \
        using reverse-mode.

        Options:
        - bool 'verbose' : toggles preceding comment before the Jacobian \
            routine definition.
        - bool 'simplify': True to apply simplifications after applying AD \
            transformations. Defaults to True.
        - int 'simplify_n_times': number of time to apply simplification \
            rules to BinaryOperation nodes. Defaults to 5.
        - bool 'inline_operation_adjoints': True to inline all possible \
            operation adjoints definitions. Defaults to True.

        :param schedule: schedule Node to the transformed.
        :type schedule: :py:class:`psyclone.psyir.nodes.Schedule`
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
            implemented for one of the children of schedule.

        :return: couple composed of the recording and returning Schedules \
            that correspond to the transformation of this Schedule.
        :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Schedule`, \
                      :py:class:`psyclone.psyir.nodes.Schedule`]
        """
        self.validate(schedule, dependent_vars, independent_vars, value_tape, options)

        self._was_applied = True

        self.schedule = schedule
        self.dependent_variables = dependent_vars
        self.independent_variables = independent_vars

        # Get the variables access information (to determine overwrites and taping)
        self.variables_info = VariablesAccessInfo(schedule)

        # Empty transformed schedules with symbol tables
        self.transformed = self.create_reversal_schedules()
        # Add the transformed schedules symbols to the container_trans map
        # self.container_trans.add_transformed(self.schedule_symbol,
        #                                              self.transformed_symbols)

        # Tape for the transformation
        # - none provided, create one
        if value_tape is None:
            # If this Schedule is truly a Routine, use its name for the tape
            if isinstance(schedule, Routine):
                name = schedule.name
            else:
                name = "schedule"
            self.value_tape = ADValueTape(name, self._default_value_tape_datatype)
        # - use the provided one
        else:
            self.value_tape = value_tape

        # Process all symbols in the table, generating adjoint symbols
        self.process_data_symbols(options)

        # Transform the statements found in the Schedule
        self.transform_children(options)

        # Inline the operation adjoints definitions
        # (rhs of Assignment nodes whose LHS is an operation adjoint)
        inline_operation_adjoints = self.unpack_option('inline_operation_adjoints',
                                                       options)
        if inline_operation_adjoints:
            self.inline_operation_adjoints(options)

        # Simplify the BinaryOperation and Assignment nodes
        # in the returning schedule
        simplify = self.unpack_option("simplify", options)
        if simplify:
            self.simplify(options)

        # Add the value_tape to the tables of both the recording
        # and returning routines iff it's actually used
        # if self.value_tape.length != 0:
        #    self.add_value_tape_to_table(options)

        # Tape all the values that are not written back to the parent schedule/scope
        # self.value_tape_non_written_values(options)

        return self.recording, self.returning, self.reversing

    def create_reversal_schedules(self):
        """Create the empty recording, returning and reversing Schedules.

        :return: all three schedules as a list.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Schedule`]
        """
        # Shallow copy the symbol table
        tables = [self.schedule_table.shallow_copy() for i in range(3)]
        original_table = self.schedule_table.shallow_copy().detach()
        tables = [table.detach() for table in tables]
        original_table.attach(self.schedule)

        # Create the schedules
        schedules = [Schedule(children=[], symbol_table=table) for table in tables]

        return schedules

    def process_data_symbols(self, options=None):
        """Process all the data symbols of the symbol table, \
        generating their adjoint symbols in the returning table \
        and adding them to the data_symbol_adjoint_map.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """

        for symbol in self.schedule_table.datasymbols:
            self.create_adjoint_symbol(symbol, options)

    def create_adjoint_symbol(self, datasymbol, options=None):
        """Create the adjoint symbol of the argument symbol in the returning \
        table.
        Note: these are manually added later to the revesing table.

        :param datasymbol: data symbol whose adjoint to create.
        :param datasymbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if datasymbol is of the wrong type.

        :return: the adjoint symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        if not isinstance(datasymbol, DataSymbol):
            raise TypeError(
                f"'datasymbol' argument should be of "
                f"type 'DataSymbol' but found"
                f"'{type(datasymbol).__name__}'."
            )
        if not datasymbol.is_scalar:
            raise NotImplementedError(
                "'datasymbol' is not a scalar. " "Arrays are not implemented yet."
            )

        # TODO: #001 use the dependent variable type and precision
        # TODO: this would depend on the result of activity analysis
        # Name using pre- and suffix
        adjoint_name = self._adjoint_prefix + datasymbol.name
        adjoint_name += self._adjoint_suffix
        # New adjoint symbol with unique name in the returning table
        adjoint = self.returning_table.new_symbol(
            adjoint_name,
            symbol_type=DataSymbol,
            datatype=self._default_adjoint_datatype,
        )

        # Add it to the map
        self.data_symbol_adjoint_map[datasymbol] = adjoint

        return adjoint

    # TODO: when using arrays, it may make sense to check indices?
    def is_written_before(self, reference):
        """Checks whether the reference was written before. \
        This only considers it appearing as lhs of assignments.

        :param reference: reference to check.
        :type reference: :py:class:`psyclone.psyir.node.Reference`

        :raises TypeError: if reference is of the wrong type.

        :return: True is there are writes before, False otherwise.
        :rtype: bool
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of "
                f"type 'Reference' but found"
                f"'{type(reference).__name__}'."
            )

        # Get the signature and indices of the assignment lhs
        sig, indices = reference.get_signature_and_indices()
        # Get the variable info
        info = self.variables_info[sig]

        # Check whether it was written before this
        return info.is_written_before(reference)

    # TODO: consider the intents
    def is_call_argument_before(self, reference):
        """Checks whether the reference appeared before as an argument \
        of a routine call.
        This does not consider the intents of the arguments in the called \
        routine.

        :param reference: reference to check.
        :type reference: :py:class:`psyclone.psyir.node.Reference`

        :raises TypeError: if reference is of the wrong type.

        :return: True if it appeared as a call argument before, \
            False otherwise.
        :rtype: bool
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of "
                f"type 'Reference' but found"
                f"'{type(reference).__name__}'."
            )

        # All preceding nodes
        preceding = reference.preceding(routine=True)
        # All preceding calls
        preceding_calls = [node for node in preceding if isinstance(node, Call)]

        # If the reference is in a call, get it
        # then remove it from the calls to consider.
        # Otherwise we'll always get True if reference is a call argument.
        parent_call = reference.ancestor(Call)
        if parent_call in preceding_calls:
            preceding_calls.remove(parent_call)

        # Check all arguments of the calls
        # if the same symbol as reference is present
        # then it was possibly written before as an argument
        for call in preceding_calls:
            refs = call.walk(Reference)
            syms = [ref.symbol for ref in refs]
            if reference.symbol in syms:
                return True

        return False

    def is_overwrite(self, reference, options=None):
        """Checks whether a reference was written before in the Schedule
        being transformed or if appeared as a call argument before. 
        Used to determine whether to value_tape its prevalue or not.

        :param reference: reference to check.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if reference is of the wrong type.

        :return: True if this reference is overwriting a prevalue, \
            False otherwise.
        :rtype: bool
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of "
                f"type 'Reference' but found"
                f"'{type(reference).__name__}'."
            )

        # Arguments with intent(in) cannot be assigned to/modified in calls
        if reference.symbol in self.recording_table.argument_list:
            if reference.symbol.interface.access == ArgumentInterface.Access.READ:
                return False

        # Check whether the reference is written before
        # this only considers assignments lhs it seems
        overwriting = self.is_written_before(reference)

        # Also check if it appears as argument in a call
        overwriting = overwriting or self.is_call_argument_before(reference)

        # Check whether it is an argument and has intent other than out
        variable_is_in_arg = (
            reference.symbol in self.schedule_table.argument_list
            and reference.symbol.interface.access != ArgumentInterface.Access.WRITE
        )

        return overwriting or variable_is_in_arg

    def transform_assignment(self, assignment, options=None):
        """Transforms an Assignment child of the schedule and adds the \
        statements to the recording and returning schedules.

        :param assignment: assignment to transform.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if assignment is of the wrong type.

        """
        if not isinstance(assignment, Assignment):
            raise TypeError(
                f"'assignment' argument should be of "
                f"type 'Assignment' but found"
                f"'{type(assignment).__name__}'."
            )

        overwriting = self.is_overwrite(assignment.lhs)

        returning = []

        # Tape record and restore first
        if overwriting:
            value_tape_record = self.value_tape.record(assignment.lhs)
            self.recording.addchild(value_tape_record)

            value_tape_restore = self.value_tape.restore(assignment.lhs)
            returning.append(value_tape_restore)

            # verbose_comment += ", overwrite"

        # Apply the transformation
        recording, ret = self.assignment_trans.apply(assignment, options)
        returning.extend(ret)

        # Insert in the recording schedule
        self.add_children(self.recording, recording)

        # Insert in the returning schedule
        self.add_children(self.returning, returning, reverse=True)

    def transform_call(self, call, options=None):
        """Transforms a Call child of the schedule and adds the \
        statements to the recording and returning schedules.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if call is of the wrong type.
        :raises NotImplementedError: if a recursive call is detected.
        """
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of "
                f"type 'Call' but found"
                f"'{type(call).__name__}'."
            )

        # == Tape record/restore the Reference arguments of the Call
        # Symbols already value_taped due to this call
        # to avoid taping multiple times if it appears as multiple
        # arguments of the call
        value_taped_symbols = []
        # accumulate the restores for now
        value_tape_restores = []
        for arg in call.children:
            if isinstance(arg, Reference):
                # Check whether the lhs variable was written before
                # or if it is an argument of a call before
                overwriting = self.is_overwrite(arg)
                # TODO: this doesn't deal with the intents
                # of the routine being called for now
                # ie. intent(in) doesn't need to be value_taped?
                # see self.is_call_argument_before

                if overwriting:
                    # Symbol wasn't value_taped yet
                    if arg.symbol not in value_taped_symbols:
                        # Tape record in the recording schedule
                        value_tape_record = self.value_tape.record(arg)
                        self.recording.addchild(value_tape_record)

                        # Associated value_tape restore in the returning schedule
                        value_tape_restore = self.value_tape.restore(arg)
                        value_tape_restores.append(value_tape_restore)

                        # Don't value_tape the same symbol again in this call
                        value_taped_symbols.append(arg.symbol)

        # Apply an ADCallTrans, this creates a new one
        recording, returning = self.call_trans.apply(call, options)

        # Add the statements to the recording schedule
        self.add_children(self.recording, recording)

        # Add the statements to the returning schedule
        # Reverse to insert always at index 0
        self.add_children(self.returning, returning, reverse=True)

        # Add the value_tape restores before the call
        self.add_children(self.returning, value_tape_restores, reverse=True)

    def transform_children(self, options=None):
        """Transforms all the children of the schedule being transformed \
        and adds the statements to the recording and returning schedules.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises NotImplementedError: if a child is a recursive Call to the \
            Schedule being transformed.
        :raises NotImplementedError: if the child transformation is not \
            implemented yet. For now only those for Assignment and Call are.
        """
        # Go line by line through the Schedule
        # Note that this creates the symbols for operation adjoints and temporaries
        for child in self.schedule.children:
            if isinstance(child, Assignment):
                self.transform_assignment(child, options)
            elif isinstance(child, Call):
                self.transform_call(child, options)
            else:
                raise NotImplementedError(
                    f"Transforming a Schedule child of "
                    f"type '{type(child).__name__}' is "
                    f"not implemented yet."
                )

    # TODO: does this only tape the locally used variables when nested scopes are present?
    def value_tape_non_written_values(self, options):
        # TODO: doc this
        """

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """
        # Values that are not written back to the calling routine
        # are all non-arguments variables.
        # NOTE: Arguments with intent(in) are not written back but cannot
        # be modified.
        for var in self.recording_table.datasymbols:
            # "fake" reference to value_tape the last value
            ref = Reference(var)

            # Record and restore
            value_tape_record = self.value_tape.record(ref)
            self.recording.addchild(value_tape_record)
            value_tape_restore = self.value_tape.restore(ref)
            self.returning.addchild(value_tape_restore, index=0)

    def new_operation_adjoint(self, datatype):
        """Creates a new adjoint symbol for an Operation node in the \
        returning table. Also appends it to the operation_adjoints list.

        :param datatype: datatype of the adjoint symbol
        :type datatype: Union[:py:class:`psyclone.psyir.symbols.ScalarType`,
                              :py:class:`psyclone.psyir.symbols.ArrayType`]

        :raises TypeError: if datatype is of the wrong type.

        :return: the adjoint symbol generated.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        if not isinstance(datatype, (ScalarType, ArrayType)):
            raise TypeError(
                f"'datatype' argument should be of type "
                f"'ScalarType' or 'ArrayType' but found "
                f"'{type(datatype).__name__}'."
            )

        adjoint = self.returning_table.new_symbol(
            self._operation_adjoint_name, symbol_type=DataSymbol, datatype=datatype
        )
        self.operation_adjoints.append(adjoint)

        return adjoint

    @property
    def adjoint_symbols(self):
        """Returns all the adjoint symbols used in transforming the Schedule,
            ie. adjoints of data symbols, operations and all temporary symbols.

        :return: list of all adjoint symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        """
        symbols = self.operation_adjoints  # + self.function_call_adjoints
        symbols += self.temp_symbols + list(self.data_symbol_adjoint_map.values())
        return symbols

    def new_temp_symbol(self, symbol, symbol_table):
        """Creates a new temporary symbol for the symbol argument.
        Uses the name of the symbol.
        Inserts it in symbol_table with an unused name and in the temp_symbol \
        list.

        :param symbol: symbol for which a temporary symbol should be created.
        :type symbol: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param symbol_table: symbol table in which to insert it.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :raises TypeError: if symbol is of the wrong type.
        :raises TypeError: if symbol_table is of the wrong type.

        :return: temporary symbol
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        if not isinstance(symbol, DataSymbol):
            raise TypeError(
                f"'symbol' argument should be of type "
                f"'DataSymbol' but found "
                f"'{type(symbol).__name__}'."
            )

        if not isinstance(symbol_table, SymbolTable):
            raise TypeError(
                f"'symbol_table' argument should be of type "
                f"'SymbolTable' but found "
                f"'{type(symbol_table).__name__}'."
            )

        name = self._temp_name_prefix + symbol.name + self._temp_name_suffix

        temp_symbol = symbol_table.new_symbol(
            name, symbol_type=DataSymbol, datatype=symbol.datatype
        )
        self.temp_symbols.append(temp_symbol)

        return temp_symbol

    def add_children(self, schedule, children, reverse=False):
        """Adds the children from a list to a schedule.
        Inserts them in the order of the list if reverse is False, \
        in the reversed order and at index 0 otherwise.

        :param schedule: schedule to add children to.
        :type schedule: :py:class:`psyclone.psyir.nodes.Schedule`
        :param children: list of children to add.
        :type children: List[:py:class:`psyclone.psyir.nodes.Schedule`]
        :param reverse: whether to reverse and add at index 0, \
            defaults to False..
        :type reverse: bool, optional

        :raises TypeError: if schedule is of the wrong type.
        :raises TypeError: if children is of the wrong type.
        :raises TypeError: if some child is of the wrong type.
        :raises TypeError: if reverse is of the wrong type.
        """
        if not isinstance(schedule, Schedule):
            raise TypeError(
                f"'schedule' argument should be of type "
                f"'Schedule' but found "
                f"'{type(schedule).__name__}'."
            )
        if not isinstance(children, list):
            raise TypeError(
                f"'children' argument should be of type "
                f"'list' but found "
                f"'{type(children).__name__}'."
            )
        for child in children:
            if not isinstance(child, Node):
                raise TypeError(
                    f"Elements of 'children' argument list "
                    f"should be of type 'Node' but found "
                    f"'{type(child).__name__}'."
                )
        if not isinstance(reverse, bool):
            raise TypeError(
                f"'reverse' argument should be of type "
                f"'bool' but found "
                f"'{type(reverse).__name__}'."
            )

        # Leave the argument list unchanged
        children_copy = children.copy()

        index = None
        if reverse:
            children_copy.reverse()
            index = 0

        for child in children_copy:
            schedule.addchild(child, index)

    def simplify(self, options=None):
        """Apply simplifications to the BinaryOperation and Assignment nodes
        of the returning schedule.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """
        simplify_n_times = self.unpack_option('simplify_n_times', options)
        for i in range(simplify_n_times):
            # Reverse the walk result to apply from deepest operations to shallowest
            all_nodes = self.returning.walk(Node)[::-1]
            for i, node in enumerate(all_nodes):
                simplified_node = simplify_node(node)
                if simplified_node is None:
                    node.detach()
                    all_nodes.pop(i)
                else:
                    if simplified_node is not node:
                        node.replace_with(simplified_node)
                        all_nodes[i] = simplified_node

    def inline_operation_adjoints(self, options=None):
        """Inline the definitions of operations adjoints, ie. the RHS of 
        Assignment nodes with LHS being an operation adjoint, 
        everywhere it's possible in the returning schedule, ie. except 
        for those used as Call arguments.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """

        # NOTE: must NOT be done for independent adjoints...
        all_assignments = self.returning.walk(Assignment)
        all_calls = self.returning.walk(Call)
        # Only assignments to operation adjoints
        op_adj_assignments = [
            assignment
            for assignment in all_assignments
            if assignment.lhs.name.startswith(self._operation_adjoint_name)
        ]
        for assignment in op_adj_assignments:
            call_args = []
            for call in all_calls:
                call_args.extend(call.children)
            if assignment.lhs in call_args:
                continue

            # Used to look at other assignments after this one only
            i = all_assignments.index(assignment)
            # Get the occurences of this operation adjoint on the rhs of
            # other assignments
            rhs_occurences = []
            for other_assignment in all_assignments[i + 1 :]:
                refs_in_rhs = other_assignment.rhs.walk(Reference)
                for ref in refs_in_rhs:
                    if ref == assignment.lhs:
                        rhs_occurences.append(ref)
                        # If already 1 occurence, we won't inline unless rhs is a Reference
                        # so stop there
                        if (not isinstance(assignment.rhs, Reference)) and (
                            len(rhs_occurences) == 2
                        ):
                            break

            if len(rhs_occurences) == 1:
                substitute = assignment.rhs.detach()
                assignment.detach()
                all_assignments.remove(assignment)
                # TODO: this might not be right for vectors...
                self.returning_table._symbols.pop(assignment.lhs.name)
                for rhs_occurence in rhs_occurences:
                    rhs_occurence.replace_with(substitute.copy())