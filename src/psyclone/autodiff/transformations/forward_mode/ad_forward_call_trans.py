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

"""This module provides a Transformation for forward-mode automatic 
differentiation of PSyIR Call nodes."""

from psyclone.psyir.nodes import (
    Call,
    Reference,
    Operation,
    Literal,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface

from psyclone.autodiff.transformations import (
    ADCallTrans,
    ADForwardRoutineTrans,
)
from psyclone.autodiff import zero


class ADForwardCallTrans(ADCallTrans):
    """A class for automatic differentation transformations of Call nodes \
    in forward-mode.
    Requires an ADForwardRoutineTrans instance as context, where the \
    derivative symbols can be found.
    Applying it generates the calls to the transformed routine.
    """

    # TODO: this only works for subroutines call for now

    @property
    def transformed(self):
        """Returns the transformed routine as a list.

        :return: list of transformed routines.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Routine`]
        """
        return self.called_routine_trans.transformed

    @property
    def transformed_symbol(self):
        """Returns the routine symbol transformed routine.

        :return: transformed routine symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        return self.called_routine_trans.transformed_symbols[0]

    def apply(self, call, options=None):
        """Applies the transformation, generating the transformed call \
        obtained by applying forward-mode automatic differentiation \
        to the call arguments.

        | Options:
        | - bool 'jacobian': whether to generate the Jacobian routine. \
                           Defaults to False.
        | - bool 'verbose' : toggles explanatory comments. Defaults to False.
        | - bool 'simplify': True to apply simplifications after applying AD \
                           transformations. Defaults to True.
        | - int 'simplify_n_times': number of time to apply simplification \
                                  rules to BinaryOperation nodes. Defaults to 5.

        :param assignment: assignment to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: transformed call.
        :rtype: :py:class:`psyclone.psyir.nodes.Call`
        """
        self.validate(call, options)

        verbose = self.unpack_option("verbose", options)

        # Call RoutineSymbol
        call_symbol = call.routine
        # Routine
        routine = self.routine_trans.container_trans.routine_from_symbol(
            call_symbol
        )
        self.routine = routine

        # If routine was already transformed, get the transformation
        if (
            self.routine_symbol
            in self.routine_trans.container_trans.routine_map
        ):
            for (
                trans
            ) in self.routine_trans.container_trans.routine_transformations:
                if trans.routine == self.routine:
                    self.called_routine_trans = trans

        else:  # Otherwise, transform it
            self.transform_called_routine(routine, options)

        # Generate the arguments to use in the calls to the transformed routine
        transformed_args = self.transform_call_arguments(call, options)

        # Call to the transformed routine
        transformed_call = Call.create(
            self.transformed_symbol, transformed_args
        )

        if verbose:
            # TODO: writer should be an attribute of the (container?) trans
            from psyclone.psyir.backend.fortran import FortranWriter

            fwriter = FortranWriter()
            src = fwriter(call.copy())
            transformed_call.preceding_comment = f"Derivating {src}"

        return transformed_call

    def transform_literal_argument(self, literal, options=None):
        """Transforms a Literal argument of the Call.
        Passes 0 as the associated derivative.

        :param literal: literal argument to transform.
        :type literal: :py:class:`psyclone.psyir.nodes.Literal`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if literal is of the wrong type.

        :return: list of transformed arguments.
        :rtype: List[Union[:py:class:`psyclone.psyir.nodes.Literal`,
                           :py:class:`psyclone.psyir.nodes.Reference`]
        """
        super().transform_literal_argument(literal, options)

        # Add (literal, 0) to the args
        return [literal.copy(), zero(literal.datatype)]

    def transform_reference_argument(self, reference, options=None):
        """Transforms a Reference argument of the Call.
        Returns the associated arguments to use in the transformed call.
        For a Reference, the arguments are the reference followed by \
        a reference to the derivative.

        :param reference: reference argument to transform.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if reference is of the wrong type.

        :return: list of transformed arguments.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Reference`]
        """
        super().transform_reference_argument(reference, options)

        # Symbol and derivative symbol of the argument
        symbol = reference.symbol
        derivative_symbol = self.routine_trans.data_symbol_differential_map[
            symbol
        ]

        # Add (var, var_d) as arguments of the transformed routine
        return [Reference(symbol), Reference(derivative_symbol)]

    def transform_operation_argument(self, operation, options=None):
        """Transforms an Operation argument of the Call.
        Returns the associated arguments to use in the transformed call.
        For an Operation, the arguments are the operation followed by \
        its derivative (as an operation).

        :param operation: operation argument to transform.
        :type operation: :py:class:`psyclone.psyir.nodes.Operation`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if operation is of the wrong type.

        :return: list of transformed arguments.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Operation`]
        """
        super().transform_operation_argument(operation, options)

        derivative = self.routine_trans.operation_trans.apply(
            operation, options
        )
        # Add (operation, derivative_operation) to the arguments
        return [operation.copy(), derivative]

    def transform_call_arguments(self, call, options=None):
        """Transforms all arguments of the Call.
        Returns the associated arguments to use in the transformed call.
        This method calls the relevant transform_[node type]_argument method \
        and appends their results.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if call is of the wrong type.

        :return: list of transformed arguments.
        :rtype: List[Union[:py:class:`psyclone.psyir.nodes.Operation`,
                           :py:class:`psyclone.psyir.nodes.Literal`,
                           :py:class:`psyclone.psyir.nodes.Reference`]]
        """
        super().transform_call_arguments(call, options)

        # Arguments to be passed in the transformed calls
        transformed_args = []

        # Process all arguments of the call
        for arg in call.children:
            if isinstance(arg, Literal):
                args = self.transform_literal_argument(arg, options)
            elif isinstance(arg, Reference):
                args = self.transform_reference_argument(arg, options)
            elif isinstance(arg, Operation):
                args = self.transform_operation_argument(arg, options)
            else:
                raise NotImplementedError(
                    f"Transforming Call with  "
                    f"arguments of type other than "
                    f"Reference is not "
                    f"implemented yet but found an "
                    f"argument of type "
                    f"'{type(arg).__name__}'."
                )

            transformed_args.extend(args)

        return transformed_args

    # TODO: this should depend on activity analysis
    def transform_called_routine(self, routine, options=None):
        """Transforms the routine found in a Call.
        **Important**: for now it treats all arguments with intent other than \
        intent(out) as independent variables and all arguments with intent \
        other than intent(in) as dependent variables, with possible overlaps. \

        :param routine: routine to be transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if routine is of the wrong type.

        :return: the transformed routine symbol
        :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        """
        super().transform_called_routine(routine, options)

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
        self.called_routine_trans = ADForwardRoutineTrans(
            self.routine_trans.container_trans
        )

        # Apply it
        self.called_routine_trans.apply(
            routine,
            dependent_args_names,
            independent_args_names,
            options=options,
        )

        return self.called_routine_trans.transformed_symbols[0]
