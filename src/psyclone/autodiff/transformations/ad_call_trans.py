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

"""This module provides an abstract Transformation for automatic differentiation
of PSyIR Call nodes.
"""

from abc import ABCMeta, abstractmethod

from psyclone.psyir.nodes import (
    Schedule,
    Call,
    Reference,
    Operation,
    Literal,
    Routine,
)
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADElementTrans


class ADCallTrans(ADElementTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of Call
    nodes.
    """

    @property
    def called_routine_trans(self):
        """Instance of a subclass of ADRoutineTrans that transforms the called \
        routine.

        :return: transformation of the called routine.
        :rtype: :py:class:`psyclone.autodiff.transformations.ADRoutineTrans`
        """
        return self._called_routine_trans

    @called_routine_trans.setter
    def called_routine_trans(self, called_routine_trans):
        # Avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADRoutineTrans

        if not isinstance(called_routine_trans, ADRoutineTrans):
            raise TypeError(
                f"Argument should be of type 'ADForwardRoutineTrans' "
                f"or 'ADReverseRoutineTrans' "
                f"but found '{type(called_routine_trans).__name__}'."
            )
        self._called_routine_trans = called_routine_trans

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
                f"'routine' argument should be of type 'Routine' but found "
                f"'{type(routine).__name__}'."
            )

        self._routine = routine

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
        """Returns the transformed routines as a list.

        :return: list of transformed routines.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Routine`]
        """
        return self.called_routine_trans.transformed

    @property
    def transformed_symbols(self):
        """Returns the routine symbols of the transformed routines as a \
        list.

        :return: list of transformed routine symbols.
        :rtype: List[:py:class:`psyclone.psyir.symbols.RoutineSymbol`]
        """
        return self.called_routine_trans.transformed_symbols

    def validate(self, call, options=None):
        """Validates the arguments of the `apply` method.

        :param assignment: node to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if call is of the wrong type.
        :raises NotImplementedError: if the parent node of Call is not a \
                                     Schedule ie. this is not called as a \
                                     subroutine.
        :raises NotImplementedError: if the call uses named arguments.
        """
        # pylint: disable=arguments-renamed

        super().validate(call, options)

        if not isinstance(call, Call):
            raise TransformationError(
                f"'call' argument should be a PSyIR 'Call' but found "
                f"'{type(call).__name__}'."
            )

        if not isinstance(call.parent, Schedule):
            raise NotImplementedError(
                "Transforming function calls is not supported yet."
            )

        for name in call.argument_names:
            if name is not None:
                raise NotImplementedError(
                    "Transforming Call with named "
                    "arguments is not implemented yet."
                )

        # Call RoutineSymbol
        call_symbol = call.routine
        already_transformed_names = [
            symbol.name
            for symbol in self.routine_trans.container_trans.routine_map
        ]

        if (
            call_symbol.name not in already_transformed_names
            and call_symbol.name
            not in [
                routine.name
                for routine 
                in self.routine_trans.container_trans.container.walk(Routine)
            ]
        ):
            raise TransformationError(
                f"Called routine named '{call_symbol.name}' "
                f"can be found neither in the routine_map "
                f"(already transformed) "
                f"nor in the names of routines in the container "
                f"(possible to transform)."
            )

    @abstractmethod
    def apply(self, call, options=None):
        """Applies the transformation, by applying automatic \
        differentiation transformations to the call arguments.

        :param assignment: node to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=arguments-renamed, unnecessary-pass
        pass

    @abstractmethod
    def transform_literal_argument(self, literal, options=None):
        """Transforms a Literal argument of the Call.

        :param literal: literal argument to transform.
        :type literal: :py:class:`psyclone.psyir.nodes.Literal`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if literal is of the wrong type.
        """
        if not isinstance(literal, Literal):
            raise TypeError(
                f"'literal' argument should be of type "
                f"'Literal' but found "
                f"'{type(literal).__name__}'."
            )

    @abstractmethod
    def transform_reference_argument(self, reference, options=None):
        """Transforms a Reference argument of the Call.

        :param reference: reference argument to transform.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if reference is of the wrong type.
        """
        if not isinstance(reference, Reference):
            raise TypeError(
                f"'reference' argument should be of type "
                f"'Reference' but found "
                f"'{type(reference).__name__}'."
            )

    @abstractmethod
    def transform_operation_argument(self, operation, options=None):
        """Transforms an Operation argument of the Call.

        :param operation: operation argument to transform.
        :type operation: :py:class:`psyclone.psyir.nodes.Operation`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if operation is of the wrong type.
        """
        if not isinstance(operation, Operation):
            raise TypeError(
                f"'operation' argument should be of type "
                f"'Operation' but found "
                f"'{type(operation).__name__}'."
            )

    @abstractmethod
    def transform_call_arguments(self, call, options=None):
        """Transforms all arguments of the Call.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if call is of the wrong type.
        """
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of type "
                f"'Call' but found "
                f"'{type(call).__name__}'."
            )

    # TODO: this should depend on activity analysis
    @abstractmethod
    def transform_called_routine(self, routine, options=None):
        """Transforms the routine found in a Call.

        :param routine: routine to be transformed.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if routine is of the wrong type.
        """
        if not isinstance(routine, Routine):
            raise TypeError(
                f"'routine' argument should be of type "
                f"'Routine' but found "
                f"'{type(routine).__name__}'."
            )
