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
differentiation of PSyIR Assignment nodes in both modes."""

from abc import ABCMeta, abstractmethod

from psyclone.psyir.nodes import Assignment, Reference
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADElementTrans


class ADAssignmentTrans(ADElementTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of Assignment \
    nodes.
    """

    def validate(self, assignment, options=None):
        """Validates the arguments of the `apply` method.

        :param assignment: node to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if assignment is of the wrong type.
        """
        super().validate(assignment, options)

        if not isinstance(assignment, Assignment):
            raise TransformationError(
                f"'assignment' argument should be a "
                f"PSyIR 'Assignment' but found '{type(assignment).__name__}'."
            )

    @abstractmethod
    def apply(self, assignment, options=None):
        """Applies the transformation, generating the recording and returning \
        motions associated to this Assignment.

        :param assignment: node to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]
        """
        pass

    # TODO: This method is not actually needed
    # it's only used as a check for now
    def is_iterative(self, assignment):
        """Checks whether the Assignment node is iterative, ie. if the LHS \
        is assigned a value that depends on itself.

        :param assignment: assignment to analyze.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`

        :raises TypeError: if assignment is of the wrong type.
        :raises NotImplementedError: if assigning to an array.

        :return: True if the assignment is iterative, False otherwise.
        :rtype: bool
        """
        # TODO: this should deal with EQUIVALENCE once supported by PSyclone
        if not isinstance(assignment, Assignment):
            raise TypeError(
                f"'assignment' argument in is_iterative should be a "
                f"PSyIR 'Assignment' but found '{type(assignment).__name__}'."
            )
        if assignment.is_array_assignment:
            raise NotImplementedError("Array assignment are not implemented yet.")

        lhs = assignment.lhs
        rhs = assignment.rhs

        refs = rhs.walk(Reference)

        for ref in refs:
            if ref == lhs:
                return True

        return False
