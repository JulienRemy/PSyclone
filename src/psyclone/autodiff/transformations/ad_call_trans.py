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
differentiation of PSyIR Call nodes."""

# TODO: this needs to deal with reversal schedules...
# TODO: for now this is "joint reversal" everywhere

from psyclone.psyir.nodes import (
    Schedule,
    Call,
    Reference,
    Operation,
    Literal,
    Routine,
    ArrayReference,
    Range,
)
from psyclone.psyir.transformations import TransformationError
from psyclone.psyir.symbols import DataSymbol, RoutineSymbol, INTEGER_TYPE
from psyclone.psyir.symbols.interfaces import ArgumentInterface

from psyclone.autodiff.transformations import ADElementTrans, ADRoutineTrans
from psyclone.autodiff import own_routine_symbol, assign_zero
from psyclone.autodiff.tapes import ADValueTape


class ADCallTrans(ADElementTrans):
    """A class for automatic differentation transformations of Call nodes.
    Requires an ADRoutineTrans instance as context, where the adjoint symbols
    can be found.
    Applying it generates the calls to the recording and returning routines and returns
    both motions.

    :param container_trans: ADRoutineTrans context instance
    :type routine_trans: :py:class:`psyclone.autodiff.transformations.ADRoutineTrans`

    :raises TypeError: if the routine_trans argument is of the wrong type
    """

    # TODO: this only works for subroutines call for now

    @property
    def called_routine_trans(self):
        """Instance of ADRoutineTrans that transforms the called routine.

        :return: transformation of the called routine.
        :rtype: :py:class:`psyclone.autodiff.transformations.ADRoutineTrans`
        """
        return self._called_routine_trans

    @called_routine_trans.setter
    def called_routine_trans(self, called_routine_trans):
        from psyclone.autodiff.transformations import ADRoutineTrans

        if not isinstance(called_routine_trans, ADRoutineTrans):
            raise TypeError(
                f"Argument should be of type 'ADRoutineTrans' "
                f"but found '{type(called_routine_trans).__name__}'."
            )
        self._called_routine_trans = called_routine_trans

    @property
    def value_tape(self):
        """Value tape used by the transformation of the called routine.

        :return: value tape.
        :rtype: :py:class:`psyclone.autodiff.tapes.ADValueTape`
        """
        return self.called_routine_trans.value_tape

    @property
    def routine(self):
        """Returns the routine node being transformed.

        :return: routine.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        return self._routine

    @routine.setter
    def routine(self, routine):
        if not isinstance(routine, Routine):
            raise TypeError(
                f"'routine' argument should be of "
                f"type 'Routine' but found"
                f"'{type(routine).__name__}'."
            )

        self._routine = routine

    @property
    def reversal_schedule(self):
        return self.routine_trans.container_trans.reversal_schedule

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
    def transformed(self):
        """Returns the 3 transformed routines as a list, these being the \
        recording routine, the returning routine and the reversing routine.

        :return: list of transformed routines.
        :rtype: List[:py:class:`psyclone.psyir.node.Routine`]
        """
        return self.called_routine_trans.transformed

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
    def transformed_symbols(self):
        """Returns the routine symbols of the  3 transformed routines as a \
        list, these being the recording routine, the returning routine and \
        the reversing routine.

        :return: list of transformed routine symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbol.RoutineSymbol`]
        """
        return self.called_routine_trans.transformed_symbols

    @property
    def recording_symbol(self):
        """Returns the routine symbol of the recording routine being generated.

        :return: recording routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return self.transformed_symbols[0]

    @property
    def returning_symbol(self):
        """Returns the routine symbol of the returning routine being generated.

        :return: returning routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return self.transformed_symbols[1]

    @property
    def reversing_symbol(self):
        """Returns the routine symbol of the reversing routine being generated.

        :return: reversing routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return self.transformed_symbols[2]

    def validate(self, call, options=None):
        """Validates the arguments of the `apply` method.

        :param assignment: node to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if call is of the wrong type.
        :raises NotImplementedError: if the parent node of Call is not a Schedule \
            ie. this is not called as a subroutine.
        :raises NotImplementedError: if the call uses named arguments.
        """
        super().validate(call, options)

        if not isinstance(call, Call):
            raise TransformationError(
                f"'call' argument in ADCallTrans should be a "
                f"PSyIR 'Call' but found '{type(call).__name__}'."
            )

        if not isinstance(call.parent, Schedule):
            raise NotImplementedError(
                "Transforming function calls is not " "supported yet."
            )

        for name in call.argument_names:
            if name is not None:
                raise NotImplementedError(
                    "Transforming Call with named " "arguments is not implemented yet."
                )

        # Call RoutineSymbol
        call_symbol = call.routine
        already_transformed_names = [
            symbol.name for symbol in self.routine_trans.container_trans.routine_map
        ]

        if (
            call_symbol.name not in already_transformed_names
            and call_symbol.name
            not in [
                routine.name
                for routine in self.routine_trans.container_trans.container.walk(
                    Routine
                )
            ]
        ):
            raise TransformationError(
                f"Called routine named '{call_symbol.name}' "
                f"can be found neither in the routine_map "
                f"(already transformed) "
                f"nor in the names of routines in the container "
                f"(possible to transform)."
            )

    def apply(self, call, options=None):
        """Applies the transformation, generating the recording and returning \
        motions obtained by applying reverse-mode automatic differentiation \
        to the call argument.

        Options:
        - bool 'jacobian': whether to generate the Jacobian routine. Defaults \
            to False.
        - bool 'verbose' : toggles preceding comment before the Jacobian \
            routine definition. Defaults to False.
        - bool 'simplify': True to apply simplifications after applying AD \
            transformations. Defaults to True.
        - int 'simplify_n_times': number of time to apply simplification \
            rules to BinaryOperation nodes. Defaults to 5.
        - bool 'inline_operation_adjoints': True to inline all possible \
            operation adjoints definitions. Defaults to True.

        :param assignment: node to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :return: couple composed of the recording and returning motions \
            that correspond to the transformation of this Call.
        :rtype: Tuple[List[:py:class:`psyclone.psyir.nodes.Node`], \
                      List[:py:class:`psyclone.psyir.nodes.Node`]]
        """
        self.validate(call, options)

        verbose = self.unpack_option("verbose", options)

        # List for both motions
        recording = []
        returning = []

        # Call RoutineSymbol
        call_symbol = call.routine
        # Routine
        routine = self.routine_trans.container_trans.routine_from_symbol(call_symbol)
        self.routine = routine

        # check if the link between the parent and called routines is strong or weak
        # strong => split reversal
        # weak => joint reversal
        split = self.reversal_schedule.is_strong_link(
            self.routine_trans.routine.name, self.routine.name
        )

        # TODO: recursive calls seem to work in joint reversal?
        # so if self.routine_trans.routine.name == routine.name
        # force split = False

        # If routine was already transformed, get the different motions
        if self.routine_symbol in self.routine_trans.container_trans.routine_map:
            for trans in self.routine_trans.container_trans.routine_transformations:
                if trans.routine == self.routine:
                    self.called_routine_trans = trans

            # (
            #    recording_symbol,
            #    returning_symbol,
            #    reversing_symbol,
            #    value_tape,
            # ) = (self.transformed_symbols, self.value_tape)

            # Symbol of the transformed routines
            # self.transformed_symbols = self.routine_trans.container_trans.routine_map[
            #    self.routine_symbol
            # ]
            # recording_symbol, returning_symbol, reversing_symbol = transformed_symbols

            # Its value_tape
            # value_tape = self.routine_trans.container_trans.value_tape_map[
            #    self.routine_symbol
            # ]
        else:  # Otherwise, transform it
            # (
            #    recording_symbol,
            #    returning_symbol,
            #    reversing_symbol,
            #    value_tape,
            # ) = self.transform_called_routine(routine, options)
            self.transform_called_routine(routine, options)

        # Generate the arguments to use in the calls to the reversing routine
        # as well as the assignments to insert before/after the calls in returning motion.
        # NOTE: tape is not included, as it it not used in reversing
        # the returning arguments are the same, plus the tape at the end
        (
            reversing_args,
            temp_assignments,
            adjoint_assignments,
        ) = self.transform_call_arguments(call, options)

        # Returning arguments, copy
        returning_args = reversing_args.copy()

        # In any case the temp assignements go before the call in the returning motion
        returning.extend(temp_assignments)

        # If this is SPLIT REVERSAL, call the recording routine when
        # recording the parent routine
        # This passes a slice of the parent value_tape as value_tape argument of the
        # called routine.
        if split:
            # Recording arguments are the same as in the original
            # plus the value_tape if it's used
            recording_args = [arg.copy() for arg in call.children]

            # TODO: arguments names
            # recording_args = [arg for arg in zip(child.argument_names,
            #                                     child.children)]

            # If the value_tape has null length, it's unused
            if self.value_tape.length > 0:
                # Extend the calling routine value tape by the called routine one
                # and get the corresponding slice of the first
                value_tape_slice = self.routine_trans.value_tape.extend_and_slice(
                    self.value_tape
                )

                # Will be used as last argument of the recording and returning calls
                recording_args.append(value_tape_slice)
                returning_args.append(value_tape_slice.copy())

            # Call to the recording routine
            recording_call = Call.create(self.recording_symbol, recording_args)
            recording.append(recording_call)

            # Call to returning routine
            returning_call = Call.create(self.returning_symbol, returning_args)
            returning.append(returning_call)

        # Otherwise this is JOINT REVERSAL
        # so call the advancing routine (original call) while recording
        # and the reversing routine while returning
        else:
            # NOTE : this is a Call using the actual RoutineSymbol of the Routine
            # rather than the original RoutineSymbol of the Call node (not the same)
            advancing_call = Call.create(
                self.routine_symbol,
                [arg.copy() for arg in call.children],
            )
            recording.append(advancing_call)

            reversing_call = Call.create(self.reversing_symbol, reversing_args)
            returning.append(reversing_call)


        ######################################
        ######################################
        ######################################
        ######################################
        # NOTE: this was wrong. 
        # adjoints are intent(inout) and only need to be set to 0 inside
        # the called routine.

        # For all written out arguments,
        # set the adjoint to 0 RIGHT AFTER the recording or reverting call,
        # as for assignments
        # Right after since the argument may also be in an operation input
        # (should not be input and output as per Fortran aliasing convention)
        #for index, arg in enumerate(reversing_args):
        #    # Only if the argument is a Reference or Call (to a function),
        #    # else the adjoint is not important (Literal or Operation)
        #    if not isinstance(arg, (Reference, Call)):
        #        continue
        #    # We only want the non-adjoint arguments to check their interface
        #    # in the called routine
        #    if arg.symbol in self.routine_trans.adjoint_symbols:
        #        continue
        #    # Get the symbol of the argument slot in the called reversing routine
        #    routine_arg_symbol = self.reversing.symbol_table.argument_list[index]
        #    # If the intent is other than intent(in) then the argument is
        #    # written out of the subroutine
        #    if isinstance(routine_arg_symbol.interface, ArgumentInterface) and (
        #        routine_arg_symbol.interface.access is not ArgumentInterface.Access.READ
        #    ):
        #        # Adjoint symbol to set to 0
        #        adjoint_sym = self.routine_trans.data_symbol_adjoint_map[arg.symbol]
        #        
        #        # Assignment to 0
        #        adj_zero = assign_zero(adjoint_sym)
        #        if verbose:
        #            adj_zero.preceding_comment = f"{arg.name} is output so overwritten"
        #
        #        ### Add right after the call, as first adjoint assignment
        #        adjoint_assignments = [adj_zero] + adjoint_assignments

        ######################################
        ######################################
        ######################################
        ######################################

        # In any case the adjoints assignments go after the call in the returning motion
        returning.extend(adjoint_assignments)

        if verbose and len(returning) != 0:
            from psyclone.psyir.backend.fortran import FortranWriter

            fwriter = FortranWriter()
            src = fwriter(call.copy())
            returning[0].preceding_comment = f"Adjoining {src}"
            returning[-1].inline_comment = f"Finished adjoining {src}"

        return recording, returning

    def transform_literal_argument(self, literal, options=None):
        """Transforms a Literal argument of the Call.
        Returns the associated arguments to use in the reversing/returning call, \
        as well as the temporary assignments (before the call) and adjoint  \
        assignments (after the call).
        Creates a dummy adjoint set to 0.0 to pass as adjoint argument.
        For a Literal, the temp assignment is `dummy_adj = 0.0`, there are no \
        adjoint assignments and the arguments are the literal follow by a \
        reference to the dummy adjoint.

        :param literal: literal argument to transform.
        :type literal: :py:class:`psyclone.psyir.nodes.Literal`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if literal is of the wrong type.

        :return: list of returning arguments, list of temporary assignments, \
            list of adjoint assignments.
        :rtype: Tuple[List[Union[:py:class:`psyclone.psyir.nodes.Literal`,
                                 :py:class:`psyclone.psyir.nodes.Reference`],
                                 :py:class:`psyclone.psyir.nodes.Assignment`]]
        """
        if not isinstance(literal, Literal):
            raise TypeError(
                f"'literal' argument should be of type "
                f"'Literal' but found "
                f"'{type(literal).__name__}'."
            )
        
        verbose = self.unpack_option("verbose", options)

        # We can't just use 0.0 as adjoint because it will cause problems
        #    with intent(inout) of the returning routine
        # So we use a dummy adjoint set to 0.0
        # eg. call foo(x,1.0,f)
        # gives returning motion:
        # temp_dummy_adj = 0.0
        # call foo_rev(x, x_adj, 1.0, temp_dummy_adj, f, f_adj)

        # Data symbol is only here to use new_temp_symbol method
        sym = DataSymbol("dummy_adj", self.routine_trans._default_adjoint_datatype)
        # Temporary dummy adjoint
        dummy_adj = self.routine_trans.new_temp_symbol(
            sym, self.routine_trans.returning_table
        )
        # dummy_adj = 0.0
        dummy_adj_zero = assign_zero(dummy_adj)
        if verbose:
            dummy_adj_zero.preceding_comment = f"Dummy adjoint for literal {literal.value}"
        # Will be done before the call
        temp_assignments = [dummy_adj_zero]
        # Add (literal, dummy_adj) to the args
        returning_args = [literal.copy(), Reference(dummy_adj)]
        # There are no adjoint assignments
        adjoint_assignments = []

        return returning_args, temp_assignments, adjoint_assignments

    def transform_reference_argument(self, reference, options=None):
        """Transforms a Reference argument of the Call.
        Returns the associated arguments to use in the reversing/returning call, \
        as well as the temporary assignments (before the call) and adjoint  \
        assignments (after the call).
        For a Reference, there are no assignments of either type \
        and the arguments are the reference follow by a reference to the adjoint.

        :param reference: reference argument to transform.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if reference is of the wrong type.

        :return: list of returning arguments, list of temporary assignments, \
            list of adjoint assignments.
        :rtype: Tuple[List[Union[:py:class:`psyclone.psyir.nodes.Reference`],
                                 :py:class:`psyclone.psyir.nodes.Assignment`]]
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of type "
                f"'Reference' but found "
                f"'{type(reference).__name__}'."
            )

        # Symbol and adjoint symbol of the argument
        symbol = reference.symbol
        adjoint_symbol = self.routine_trans.data_symbol_adjoint_map[symbol]

        # Add (var, var_adj) as arguments of the returning/reversing routines
        returning_args = [Reference(symbol), Reference(adjoint_symbol)]
        # No temporary assignments
        # TODO: functions would potentially have some eg y = f(y)
        temp_assignments = []

        # NOTE: Set to 0 the adjoints of
        # all returned argument variables
        # same as with the lhs of an assignment
        # but this is due to the ArgumentInterface.access
        # attribute of the corresponding symbol of the
        # called routine
        # So it is not done here.
        adjoint_assignments = []

        #####################################
        # This is not needed given Fortran argument aliasing rules
        # TODO: make very sure of this
        # If it's needed, then we need to sum the aliased adjoints,
        # >>>>NOT<<<< AS DONE BELOW
        #####################################
        # else:
        #    # To make sure no aliasing is problematic, given some call
        #    #   call sr_ret(x, x_adj, f, f_adj, ...)
        #    # we use temporaries as:
        #    #   temp_x_adj = x_adj
        #    #   x_adj = 0
        #    #   temp_f_adj = f_adj
        #    #   f_adj = 0
        #    #   ...
        #    #   call sr_ret(x, temp_x_adj, f, temp_f_adj, ...)
        #    #   x_adj = temp_x_adj
        #    #   f_adj = temp_f_adj
        #    #   ...
        #    temp_adjoint_symbol = self.new_temp_symbol(adjoint_symbol,
        #                                                self.returning_table)
        #    # temp = adj
        #    # adj = 0
        #    temp_assign = assign(temp_adjoint_symbol, adjoint_symbol)
        #    adjoint_zero = assign_zero(adjoint_symbol)
        #    temp_assignments.extend([temp_assign, adjoint_zero])
        #    # adj = temp
        #    adjoint_assign = assign(adjoint_symbol, temp_adjoint_symbol)
        #    adjoints_assignments.append(adjoint_assign)
        #
        #    # call arguments as (x, temp_x_adj, f, temp_f_adj, etc.)
        #    returning_args.extend([Reference(symbol),
        #                            Reference(temp_adjoint_symbol)])

        return returning_args, temp_assignments, adjoint_assignments

    def transform_operation_argument(self, operation, options=None):
        """Transforms an Operation argument of the Call.
        Returns the associated arguments to use in the reversing/returning call, \
        as well as the temporary assignments (before the call) and adjoint  \
        assignments (after the call).
        Creates an operation adjoint.
        For a Literal, the temp assignment is `temp_[var]_adj = 0.0`, the \
        adjoint assignments are those obtained by transforming the Operation \
        and the arguments are the Operation follow by a reference to its adjoint.

        :param operation: operation argument to transform.
        :type operation: :py:class:`psyclone.psyir.nodes.Operation`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if operation is of the wrong type.

        :return: list of returning arguments, list of temporary assignments, \
            list of adjoint assignments.
        :rtype: Tuple[List[Union[:py:class:`psyclone.psyir.nodes.Operation`,
                                 :py:class:`psyclone.psyir.nodes.Reference`],
                                 :py:class:`psyclone.psyir.nodes.Assignment`]]
        """
        if not isinstance(operation, Operation):
            raise TypeError(
                f"'operation' argument should be of type "
                f"'Operation' but found "
                f"'{type(operation).__name__}'."
            )
        # if the argument of the called routine is an operation
        # eg. call foo(x+y,f)
        # - create an adjoint for the operation
        # - assign 0 to it
        # - use it as argument adjoint in the call
        # - increment the adjoints of the operands using ADOperationTrans
        # eg.
        # op_adj = 0.0
        # call foo_ret(x+y, op_adj, f, f_adj)
        # x_adj = x_adj + op_adj * 1
        # y _adj = y_adj + op_adj * 1

        verbose = self.unpack_option("verbose", options)

        # TODO: correct datatype
        # TODO: test this !
        # New adjoint for the operation
        op_adj = self.routine_trans.new_operation_adjoint(
            self.routine_trans._default_adjoint_datatype
        )
        # op_adj = 0.0
        op_adj_zero = assign_zero(op_adj)

        # Will be written before the call
        temp_assignments = [op_adj_zero]
        # Add (operation, op_adj) to the arguments
        returning_args = [operation.copy(), Reference(op_adj)]
        # Incrementing the adjoints of the operation is done after the call
        adjoints_assignments = self.routine_trans.operation_trans.apply(
            operation, op_adj, options
        )[0] # NOTE: [0] since a Call to a subroutine cannot be iterative

        return returning_args, temp_assignments, adjoints_assignments

    def transform_call_arguments(self, call, options=None):
        """Transforms an all arguments of the Call.
        Returns the associated arguments to use in the reversing/returning call, \
        as well as the temporary assignments (before the call) and adjoint  \
        assignments (after the call).
        This method calls the relevant transform_[node type]_argument method \
        and appends their results.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if call is of the wrong type.

        :return: list of returning arguments, list of temporary assignments, \
            list of adjoint assignments.
        :rtype: Tuple[List[Union[:py:class:`psyclone.psyir.nodes.Operation`,
                                 :py:class:`psyclone.psyir.nodes.Literal`,
                                 :py:class:`psyclone.psyir.nodes.Reference`],
                                 :py:class:`psyclone.psyir.nodes.Assignment`]]
        """
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of type "
                f"'Call' but found "
                f"'{type(call).__name__}'."
            )

        # Determine the arguments of the returning/reversing calls
        # Include the adjoints of the arguments
        # the value_tape if used
        # Arguments to be passed in the returning/reversing calls
        returning_args = []
        # Assignments (to temporaries) to be done before the call
        temp_assignments = []
        # Assignments (to adjoints) to be done after the call
        adjoints_assignments = []

        # Process all arguments of the call
        for arg in call.children:
            if isinstance(arg, Literal):
                ret_args, temp_asgs, adjoint_asgs = self.transform_literal_argument(
                    arg, options
                )
            elif isinstance(arg, Reference):
                ret_args, temp_asgs, adjoint_asgs = self.transform_reference_argument(
                    arg, options
                )
            elif isinstance(arg, Operation):
                ret_args, temp_asgs, adjoint_asgs = self.transform_operation_argument(
                    arg, options
                )
            else:
                raise NotImplementedError(
                    f"Transforming Call with  "
                    f"arguments of type other than "
                    f"Reference is not "
                    f"implemented yet but found an "
                    f"argument of type "
                    f"'{type(arg).__name__}'."
                )

            temp_assignments.extend(temp_asgs)
            returning_args.extend(ret_args)
            adjoints_assignments.extend(adjoint_asgs)

        return returning_args, temp_assignments, adjoints_assignments

    # TODO: this should depend on activity analysis
    def transform_called_routine(self, routine, options=None):
        """Transforms the routine found in a Call, returning its recording, \
        returning and reversing routines as well as its value_tape.
        **Important**: for now it treats all arguments with intent other than \
        intent(out) as independent variables and all arguments with intent \
        other than intent(in) as dependent variables, with possible overlaps. \

        :param routine: routine to be transformed.
        :type routine: :py:class:`psyclone.psyir.node.Routine`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TypeError: if routine is of the wrong type.

        :return: the three transformed routines and the value_tape
        :rtype: tuple[:py:class:`psyclone.psyir.node.Routine`, \
                      :py:class:`psyclone.psyir.node.Routine`, \
                      :py:class:`psyclone.psyir.node.Routine`, \
                      :py:class:`psyclone.autodiff.ADValueTape`]
        """
        if not isinstance(routine, Routine):
            raise TypeError(
                f"'routine' argument should be of type "
                f"'Routine' but found "
                f"'{type(routine).__name__}'."
            )

        routine_symbol = own_routine_symbol(routine)

        ##############################
        # for now this treats
        # - all intent(in), intent(inout) or default intents
        #       as independent variables
        # - all intent(out), intent(inout) or default intents
        #       as dependent variables

        # Get the arguments of the routine and sort them by intent
        args = routine.symbol_table.argument_list
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

        # Transformation
        self.called_routine_trans = ADRoutineTrans(self.routine_trans.container_trans)

        # Apply it
        self.called_routine_trans.apply(
            routine,
            dependent_args_names,
            independent_args_names,
            value_tape=None,
            options=options,
        )

        # Get the value_tape
        value_tape = self.called_routine_trans.value_tape
        ## Add it to the container_trans map
        # self.routine_trans.container_trans.add_value_tape(routine_symbol, value_tape)

        # Get the routines symbols
        transformed_symbols = self.called_routine_trans.transformed_symbols

        return *transformed_symbols, value_tape
