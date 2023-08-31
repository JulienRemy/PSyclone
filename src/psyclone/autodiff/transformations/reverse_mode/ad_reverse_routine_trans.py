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

"""This module provides a Transformation for reverse-mode automatic 
differentiation of PSyIR Routine nodes."""

from psyclone.core import VariablesAccessInfo
from psyclone.psyir.nodes import (
    Call,
    Reference,
    Assignment,
)
from psyclone.psyir.symbols import (
    REAL_DOUBLE_TYPE,
    DataSymbol,
    ScalarType,
    ArrayType,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface

from psyclone.autodiff import assign_zero, own_routine_symbol
from psyclone.autodiff.tapes import ADValueTape
from psyclone.autodiff.transformations import ADRoutineTrans


class ADReverseRoutineTrans(ADRoutineTrans):
    """A class for automatic differentation transformations of Routine nodes. 
    Requires an ADReverseContainerTrans instance as context, where the \
    definitions of the routines called inside the one to be transformed can be \
    found.

    :param container_trans: ADReverseContainerTrans context instance
    :type container_trans: 
           :py:class:`psyclone.autodiff.transformations.ADReverseContainerTrans`

    :raises TypeError: if the container_trans argument is of the wrong type.
    """

    # Pre- and postfix of the three transformed routines
    _recording_prefix = ""
    _recording_postfix = "_rec"

    _returning_prefix = ""
    _returning_postfix = "_ret"

    _reversing_prefix = ""
    _reversing_postfix = "_rev"

    _routine_prefixes = (
        _recording_prefix,
        _returning_prefix,
        _reversing_prefix,
    )
    _routine_postfixes = (
        _recording_postfix,
        _returning_postfix,
        _reversing_postfix,
    )

    # Redefining parent class attributes
    _number_of_routines = 3  # Recording, returning, reversing
    _differential_prefix = ""
    _differential_postfix = "_adj"
    _differential_table_index = (
        1  # Adjoints are created in the returning table
    )

    _operation_adjoint_name = "op_adj"
    # _call_adjoint_name = "call_adj"

    # TODO: correct datatype
    _default_value_tape_datatype = REAL_DOUBLE_TYPE

    def __init__(self, container_trans):
        super().__init__()

        # Contextual container trans
        self.container_trans = container_trans

        # DataSymbol => adjoint DataSymbol
        self.data_symbol_differential_map = dict()

        # Lists of adjoint symbols for operations
        self.operation_adjoints = []

        # Transformations need to know about the ADReverseRoutineTrans
        # calling them
        # to access the attributes defined above
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import (
            ADReverseOperationTrans,
            ADReverseAssignmentTrans,
            ADReverseCallTrans,
        )

        # Initialize the sub transformations
        self.assignment_trans = ADReverseAssignmentTrans(self)
        self.operation_trans = ADReverseOperationTrans(self)
        self.call_trans = ADReverseCallTrans(self)

    @property
    def container_trans(self):
        """Returns the ADReverseContainerTrans this instance uses.

        :return: container transformation, reverse-mode.
        :rtype: \
           :py:class:`psyclone.autodiff.transformations.ADReverseContainerTrans`
        """
        return self._container_trans

    @container_trans.setter
    def container_trans(self, container_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseContainerTrans

        if not isinstance(container_trans, ADReverseContainerTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseContainerTrans' "
                f"but found '{type(container_trans)}.__name__'."
            )

        self._container_trans = container_trans

    @property
    def assignment_trans(self):
        """Returns the ADReverseAssignmentTrans this instance uses.

        :return: assignment transformation, reverse-mode.
        :rtype: \
          :py:class:`psyclone.autodiff.transformations.ADReverseAssignmentTrans`
        """
        return self._assignment_trans

    @assignment_trans.setter
    def assignment_trans(self, assignment_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseAssignmentTrans

        if not isinstance(assignment_trans, ADReverseAssignmentTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseAssignmentTrans' "
                f"but found '{type(assignment_trans)}.__name__'."
            )

        self._assignment_trans = assignment_trans

    @property
    def operation_trans(self):
        """Returns the ADReverseOperationTrans this instance uses.

        :return: operation transformation, reverse-mode.
        :rtype: \
           :py:class:`psyclone.autodiff.transformations.ADReverseOperationTrans`
        """
        return self._operation_trans

    @operation_trans.setter
    def operation_trans(self, operation_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseOperationTrans

        if not isinstance(operation_trans, ADReverseOperationTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseOperationTrans' "
                f"but found '{type(operation_trans)}.__name__'."
            )

        self._operation_trans = operation_trans

    @property
    def call_trans(self):
        """Returns the ADReverseCallTrans this instance uses.

        :return: call transformation, reverse-mode.
        :rtype: :py:class:`psyclone.autodiff.transformations.ADReverseCallTrans`
        """
        return self._call_trans

    @call_trans.setter
    def call_trans(self, call_trans):
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseCallTrans

        if not isinstance(call_trans, ADReverseCallTrans):
            raise TypeError(
                f"Argument should be an 'ADReverseCallTrans' "
                f"but found '{type(call_trans)}.__name__'."
            )

        self._call_trans = call_trans

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
    def adjoint_symbols(self):
        """Returns all the adjoint symbols used in transforming the Routine,
            ie. adjoints of data symbols, operations and all temporary symbols.

        :return: list of all adjoint symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        """
        symbols = self.operation_adjoints  # + self.function_call_adjoints
        symbols += self.temp_symbols + list(
            self.data_symbol_differential_map.values()
        )
        return symbols

    def validate(
        self,
        routine,
        dependent_vars,
        independent_vars,
        value_tape=None,
        options=None,
    ):
        """Validates the arguments of the `apply` method.

        :param routine: routine Node to the transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param dependent_vars: list of dependent variables names to be \
                               differentiated.
        :type dependent_vars: `List[Str]`
        :param independent_vars: list of independent variables names to \
                                 differentiate with respect to.
        :type independent_vars: `List[Str]`
        :param value_tape: value tape to use to transform the routine.
        :type value_tape: Optional[Union[NoneType, ADValueTape]]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if value_tape is of the wrong type.
        """
        # pylint: disable=arguments-renamed, too-many-arguments

        super().validate(routine, dependent_vars, independent_vars, options)

        if not isinstance(value_tape, (ADValueTape, type(None))):
            raise TypeError(
                f"'value_tape' argument should be of "
                f"type 'ADValueTape' or 'NoneType' but found an element of type"
                f"'{type(value_tape).__name__}'."
            )

    def apply(
        self,
        routine,
        dependent_vars,
        independent_vars,
        value_tape=None,
        options=None,
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
                                            operation adjoints definitions. \
                                            Defaults to True.

        :param routine: routine Node to the transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param dependent_vars: list of dependent variables names to be \
                               differentiated.
        :type dependent_vars: `List[Str]`
        :param independent_vars: list of independent variables names to \
                                 differentiate with respect to.
        :type independent_vars: `List[Str]`
        :param value_tape: value tape to use to transform the routine.
        :type value_tape: Optional[\
                            Union[NoneType, 
                                  :py:class:`psyclone.autodiff.ADValueTape`]\
                          ]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises NotImplementedError: if no transformation rule has yet been \
                                     implemented for one of the children of \
                                     routine.

        :return: couple composed of the recording and returning Routines \
                 that correspond to the transformation of this Routine.
        :rtype: Tuple[:py:class:`psyclone.psyir.nodes.Routine`, \
                      :py:class:`psyclone.psyir.nodes.Routine`]
        """
        # pylint: disable=arguments-renamed, too-many-arguments

        self.validate(
            routine, dependent_vars, independent_vars, value_tape, options
        )

        # Transformation can only be applied once
        self._was_applied = True

        self.routine = routine
        self.dependent_variables = dependent_vars
        self.independent_variables = independent_vars

        # Get the variables access information (to determine overwrites
        # and taping)
        self.variables_info = VariablesAccessInfo(routine)

        # Tape for the transformation
        # - none provided, create one
        if value_tape is None:
            name = routine.name
            self.value_tape = ADValueTape(
                name, self._default_value_tape_datatype
            )
        # - use the provided one
        else:
            self.value_tape = value_tape

        # Add this transformation to the container_trans map
        # Do it before apply below or ordering is not from outer to
        # inner routines
        self.container_trans.add_routine_trans(self)

        # Empty transformed routines with symbol tables
        self.transformed = self.create_transformed_routines()

        # Process all symbols in the table, generating adjoint symbols
        self.process_data_symbols(options)

        # Transform the statements found in the Routine
        self.transform_children(options)

        # Inline the operation adjoints definitions
        # (rhs of Assignment nodes whose LHS is an operation adjoint)
        inline_operation_adjoints = self.unpack_option(
            "inline_operation_adjoints", options
        )
        if inline_operation_adjoints:
            self.inline_operation_adjoints(options)

        # Simplify the BinaryOperation and Assignment nodes
        # in the returning routine
        simplify = self.unpack_option("simplify", options)
        if simplify:
            self.simplify(self.returning, options)

        # Add the transformed routines symbols to the container_trans map
        self.container_trans.add_transformed_routines(
            self.routine_symbol, self.transformed_symbols
        )

        # Tape all the values that are not written back to the parent
        # routine/scope
        self.value_tape_non_written_values(options)

        # add The value tape to the container_trans map
        self.container_trans.add_value_tape(
            self.routine_symbol, self.value_tape
        )

        # All dependent and independent variables names
        # list(set(...)) to avoid duplicates
        diff_variables = list(set(self.differential_variables))

        # Add the necessary adjoints as arguments of the returning routine
        self.add_adjoint_arguments(diff_variables, options)

        # Change the intents as needed
        # NOTE: this replaces the non-adjoint symbols in the returning
        # symbol table
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
                "reverse", dependent_vars, independent_vars, options
            )
            self.container_trans.container.addchild(jacobian_routine)

        return self.recording, self.returning, self.reversing

    # TODO: when using arrays, it may make sense to check indices?
    def is_written_before(self, reference):
        """Checks whether the reference was written before. \
        This only considers it appearing as lhs of assignments.

        :param reference: reference to check.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if reference is of the wrong type.

        :return: True is there are writes before, False otherwise.
        :rtype: Bool
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of "
                f"type 'Reference' but found"
                f"'{type(reference).__name__}'."
            )

        # Get the signature and indices of the assignment lhs
        sig, _ = reference.get_signature_and_indices()
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
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if reference is of the wrong type.

        :return: True if it appeared as a call argument before, False otherwise.
        :rtype: Bool
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
        preceding_calls = [
            node for node in preceding if isinstance(node, Call)
        ]

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
        """Checks whether a reference was written before in the Routine
        being transformed or if appeared as a call argument before. 
        Used to determine whether to value_tape its prevalue or not.

        :param reference: reference to check.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`
        :param options: a dictionary with options for transformations, \
                       defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if reference is of the wrong type.

        :return: True if this reference is overwriting a prevalue, \
                 False otherwise.
        :rtype: Bool
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of "
                f"type 'Reference' but found"
                f"'{type(reference).__name__}'."
            )

        # Arguments with intent(in) cannot be assigned to/modified in calls
        if reference.symbol in self.recording_table.argument_list:
            if (
                reference.symbol.interface.access
                == ArgumentInterface.Access.READ
            ):
                return False

        # Check whether the reference is written before
        # this only considers assignments lhs it seems
        overwriting = self.is_written_before(reference)

        # Also check if it appears as argument in a call
        overwriting = overwriting or self.is_call_argument_before(reference)

        # Check whether it is an argument and has intent other than out
        variable_is_in_arg = (
            reference.symbol in self.routine_table.argument_list
            and reference.symbol.interface.access
            != ArgumentInterface.Access.WRITE
        )

        return overwriting or variable_is_in_arg

    def transform_assignment(self, assignment, options=None):
        """Transforms an Assignment child of the routine and adds the \
        statements to the recording and returning routines.

        :param assignment: assignment to transform.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

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

        # Insert in the recording routine
        self.add_children(self.recording, recording)

        # Insert in the returning routine
        self.add_children(self.returning, returning, reverse=True)

    def transform_call(self, call, options=None):
        """Transforms a Call child of the routine and adds the \
        statements to the recording and returning routines.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if call is of the wrong type.
        """
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of "
                f"type 'Call' but found"
                f"'{type(call).__name__}'."
            )

        # Tape record/restore the Reference arguments of the Call
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
                        # Tape record in the recording routine
                        value_tape_record = self.value_tape.record(arg)
                        self.recording.addchild(value_tape_record)

                        # Associated value_tape restore in the returning routine
                        value_tape_restore = self.value_tape.restore(arg)
                        value_tape_restores.append(value_tape_restore)

                        # Don't value_tape the same symbol again in this call
                        value_taped_symbols.append(arg.symbol)

        # Apply an ADReverseCallTrans
        recording, returning = self.call_trans.apply(call, options)

        # Add the statements to the recording routine
        self.add_children(self.recording, recording)

        # Add the statements to the returning routine
        # Reverse to insert always at index 0
        self.add_children(self.returning, returning, reverse=True)

        # Add the value_tape restores before the call
        self.add_children(self.returning, value_tape_restores, reverse=True)

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
            self._operation_adjoint_name,
            symbol_type=DataSymbol,
            datatype=datatype,
        )
        self.operation_adjoints.append(adjoint)

        return adjoint

    def inline_operation_adjoints(self, options=None):
        """Inline the definitions of operations adjoints, ie. the RHS of 
        Assignment nodes with LHS being an operation adjoint, 
        everywhere it's possible in the returning routine, ie. except 
        for those used as Call arguments.

        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=protected-access

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
                        # If already 1 occurence, we won't inline unless rhs
                        # is a Reference
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
            ! db/da is evidently 4, so the last value of c needs to be taped
        end subroutine foo```

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=protected-access

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
        Note that intent(in) in the returning routine would not allow tape \
        restores.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=protected-access

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
        :type options: Optional[Dict[Str, Any]]

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
            symbol = self.returning_table.lookup(
                var, scope_limit=self.returning
            )

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
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=protected-access

        # Reshape the value_tape array to its correct number of elements
        # self.value_tape.reshape()

        # Get three symbols for the value_tape, one per routine
        symbols = [self.value_tape.symbol.copy() for i in range(3)]

        # Use the correct intents
        # intent(out) for the recording routine
        symbols[0].interface = ArgumentInterface(
            ArgumentInterface.Access.WRITE
        )
        # intent(in) for the returning routine
        symbols[1].interface = ArgumentInterface(ArgumentInterface.Access.READ)
        # The reversing routine declares the value_tape,
        # so the default AutomaticInterface is correct

        # Add the value_tape to all tables
        for table, symbol in zip(self.transformed_tables, symbols):
            table.add(symbol)

        # Append it to the arguments lists of the recording and returning
        # routines only
        for table, symbol in zip(self.transformed_tables[:-1], symbols[:-1]):
            table._argument_list.append(symbol)
        # The value_tape is not an argument of the reversing routine

    def add_adjoint_assignments(self, options=None):
        """Assign the value 0 to every variable adjoint (ie. not temporary \
        adjoint nor operation adjoint) that is not an argument of the \
        returning routine, at its beginning.

        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=protected-access

        variables_adjoints = list(self.data_symbol_differential_map.values())
        # Reverse it so that the assignments are in variables appearance order
        # when inserting at index 0
        variables_adjoints.reverse()
        for adjoint_symbol in variables_adjoints:
            if adjoint_symbol not in self.returning_table._argument_list:
                assignment = assign_zero(adjoint_symbol)
                self.returning.addchild(assignment, index=0)

    def add_calls_to_reversing(self, options=None):
        """Inserts two calls, to the recording and returning routines, in the \
        reversing routine.

        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]
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
