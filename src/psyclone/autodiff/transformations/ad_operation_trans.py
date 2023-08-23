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
differentiation of PSyIR Operation nodes."""

from abc import ABCMeta, abstractmethod

from psyclone.psyir.nodes import (
    UnaryOperation,
    BinaryOperation,
    NaryOperation,
    Operation,
)
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADElementTrans


class ADOperationTrans(ADElementTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of Operation nodes.
    """

    def validate(self, operation, options=None):
        """Validates the arguments of the `apply` method.

        :param operation: operation Node to be transformed.
        :type operation: :py:class:`psyclone.psyir.nodes.Operation`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if operation is of the wrong type.
        """
        super().validate(operation, options)

        if not isinstance(operation, Operation):
            raise TransformationError(
                f"'operation' argument in ADReverseOperationTrans should be a "
                f"PSyIR 'Operation' but found '{type(operation).__name__}'."
            )

    @abstractmethod
    def apply(self, operation, options=None):
        """Applies the transformation.

        :param operation: operation Node to be transformed.
        :type operation: :py:class:`psyclone.psyir.nodes.Operation`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """
        
    @abstractmethod
    def differentiate(self, operation):
        """Compute the derivative(s) of the operation argument.

        :param operation: operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.Operation`

        :raises TypeError: if operation is of the wrong type.
        :raises NotImplementedError: if operation is an NaryOperation instance.
        """
        if not isinstance(operation, Operation):
            raise TypeError(
                f"Argument in differentiate should be a "
                f"PSyIR 'Operation' but found '{type(operation).__name__}'."
            )

        if isinstance(operation, NaryOperation):
            raise NotImplementedError(
                "Differentiating NaryOperation nodes " "isn't implement yet."
            )

    @abstractmethod
    def differentiate_unary(self, operation):
        """Compute the derivative of the operation argument.

        :param operation: unary operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.UnaryOperation`

        :raises TypeError: if operation is of the wrong type.
        """
        if not isinstance(operation, UnaryOperation):
            raise TypeError(
                f"Argument in differentiate_unary should be a "
                f"PSyIR UnaryOperation but found '{type(operation).__name__}'."
            )

    @abstractmethod
    def differentiate_binary(self, operation):
        """Compute the derivative(s) of the operation argument.

        :param operation: binary operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.BinarOperation`

        :raises TypeError: if operation is of the wrong type.
        """
        if not isinstance(operation, BinaryOperation):
            raise TypeError(
                f"Argument in differentiate_binary should be a "
                f"PSyIR BinaryOperation but found '{type(operation).__name__}'."
            )

    # TODO: implement these
    #@abstractmethod
    # def differentiate_nary_operation(self, operation):
    #    pass