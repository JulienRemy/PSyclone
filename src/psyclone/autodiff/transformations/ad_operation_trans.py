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
of PSyIR Operation and IntrinsicCall nodes.
"""

from abc import ABCMeta, abstractmethod

from psyclone.psyir.nodes import (
    UnaryOperation,
    BinaryOperation,
    Operation,
    IntrinsicCall
)
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADElementTrans


class ADOperationTrans(ADElementTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of \
    Operation and IntrinsicCall nodes.
    """

    def validate(self, operation, options=None):
        """Validates the arguments of the `apply` method.

        :param operation: operation or intrinsic Node to be transformed.
        :type operation: Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if operation is of the wrong type.
        """
        # pylint: disable=arguments-renamed

        super().validate(operation, options)

        if not isinstance(operation, (Operation, IntrinsicCall)):
            raise TransformationError(
                f"'operation' argument should be a "
                f"PSyIR 'Operation' or 'IntrinsicCall' but found "
                f"'{type(operation).__name__}'."
            )

    @abstractmethod
    def apply(self, operation, options=None):
        """Applies the transformation.

        :param operation: operation Node to be transformed.
        :type operation: Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=arguments-renamed, unnecessary-pass
        pass

    @abstractmethod
    def differentiate(self, operation, options=None):
        """Compute the derivative(s) of the 'operation' argument.

        :param operation: operation Node to be differentiated.
        :type operation: Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if operation is of the wrong type.
        """
        if not isinstance(operation, (Operation, IntrinsicCall)):
            raise TypeError(
                f"Argument in differentiate should be a "
                f"PSyIR 'Operation' or 'IntrinsicCall' but found "
                f"'{type(operation).__name__}'."
            )
        self.typecheck_options(options)

    @abstractmethod
    def differentiate_unary(self, operation, options=None):
        """Compute the derivative of the operation argument.

        :param operation: unary operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.UnaryOperation`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if operation is of the wrong type.
        """
        if not isinstance(operation, UnaryOperation):
            raise TypeError(
                f"Argument in differentiate_unary should be a "
                f"PSyIR UnaryOperation but found '{type(operation).__name__}'."
            )
        self.typecheck_options(options)

    @abstractmethod
    def differentiate_binary(self, operation, options=None):
        """Compute the derivative(s) of the operation argument.

        :param operation: binary operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.BinarOperation`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if operation is of the wrong type.
        """
        if not isinstance(operation, BinaryOperation):
            raise TypeError(
                f"Argument in differentiate_binary should be a "
                f"PSyIR BinaryOperation but found '{type(operation).__name__}'."
            )
        self.typecheck_options(options)

    @abstractmethod
    def differentiate_intrinsic(self, intrinsic_call, options=None):
        """Compute the derivative(s) of the operation argument.

        :param intrinsic_call: intrinsic call Node to be differentiated.
        :type intrinsic_call: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if intrinsic_call is of the wrong type.
        """
        if not isinstance(intrinsic_call, IntrinsicCall):
            raise TypeError(
                f"Argument in differentiate_intrinsic should be a "
                f"PSyIR IntrinsicCall but found '{type(intrinsic_call).__name__}'."
            )
        self.typecheck_options(options)
