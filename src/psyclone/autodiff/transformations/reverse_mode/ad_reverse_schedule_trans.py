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
    Schedule,
)
from psyclone.psyir.symbols import (
    REAL_DOUBLE_TYPE,
    DataSymbol,
    ScalarType,
    ArrayType,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface
from psyclone.core import VariablesAccessInfo

from psyclone.autodiff.tapes import ADValueTape
from psyclone.autodiff.transformations import ADScheduleTrans


class ADReverseScheduleTrans(ADScheduleTrans):
    """A class for automatic differentation transformations of Schedule nodes.
    Requires an ADReverseContainerTrans instance as context, where the definitions of
    the routines called inside the schedule to be transformed can be found.

    :param container_trans: ADReverseContainerTrans context instance
    :type container_trans: :py:class:`psyclone.autodiff.transformations.ADReverseContainerTrans`

    :raises TypeError: if the container_trans argument is of the wrong type.
    """

    _number_of_schedules = 3        # Recording, returning, reversing
    _differential_prefix = ""
    _differential_postfix = "_adj"
    _differential_table_index = 1   # Adjoints are created in the returning table

    _operation_adjoint_name = "op_adj"
    # _call_adjoint_name = "call_adj"

    # TODO: correct datatype
    _default_value_tape_datatype = REAL_DOUBLE_TYPE

    def __init__(self, container_trans):
        super().__init__(container_trans)

        # DataSymbol => adjoint DataSymbol
        self.data_symbol_differential_map = dict()

        # Lists of adjoint symbols for operations
        self.operation_adjoints = []

        # Transformations need to know about the ADReverseScheduleTrans calling them
        # to access the attributes defined above
        # Import here to avoid circular dependencies
        from psyclone.autodiff.transformations import (
            ADReverseOperationTrans,
            ADReverseAssignmentTrans,
            ADReverseCallTrans,
        )

        # Initialize the sub transformations
        # self.adjoint_symbol_trans = ADAdjointSymbolTrans(self)
        self.assignment_trans = ADReverseAssignmentTrans(self)
        self.operation_trans = ADReverseOperationTrans(self)
        self.call_trans = ADReverseCallTrans(self)

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
        super().validate(schedule, dependent_vars, independent_vars, options)

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

        :return: list of the three transformed schedules \
        (recording, returning, reversing).
        :rtype: List[:py:class:`psyclone.psyir.nodes.Schedule`]
        """
        self.validate(schedule, dependent_vars, independent_vars, value_tape, options)

        self._was_applied = True

        self.schedule = schedule
        self.dependent_variables = dependent_vars
        self.independent_variables = independent_vars

        # Get the variables access information (to determine overwrites and taping)
        self.variables_info = VariablesAccessInfo(schedule)

        # Empty transformed schedules with symbol tables
        self.transformed = self.create_transformed_schedules()
        # Add the transformed schedules symbols to the container_trans map
        # self.container_trans.add_transformed(self.schedule_symbol,
        #                                              self.transformed_symbols)

        # Tape for the transformation
        # - none provided, create one
        if value_tape is None:
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
        inline_operation_adjoints = self.unpack_option(
            "inline_operation_adjoints", options
        )
        if inline_operation_adjoints:
            self.inline_operation_adjoints(options)

        # Simplify the BinaryOperation and Assignment nodes
        # in the returning schedule
        simplify = self.unpack_option("simplify", options)
        if simplify:
            self.simplify(self.returning, options)

        return self.transformed

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
        super().transform_assignment(assignment, options)

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
        super().transform_call(call, options)

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

        # Apply an ADReverseCallTrans
        recording, returning = self.call_trans.apply(call, options)

        # Add the statements to the recording schedule
        self.add_children(self.recording, recording)

        # Add the statements to the returning schedule
        # Reverse to insert always at index 0
        self.add_children(self.returning, returning, reverse=True)

        # Add the value_tape restores before the call
        self.add_children(self.returning, value_tape_restores, reverse=True)

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
        symbols += self.temp_symbols + list(self.data_symbol_differential_map.values())
        return symbols

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
