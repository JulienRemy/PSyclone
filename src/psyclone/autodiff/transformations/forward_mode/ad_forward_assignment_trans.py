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

"""This module provides a Transformation for forward-mode automatic 
differentiation of PSyIR Assignment nodes."""

from psyclone.psyir.nodes import Literal, Reference, Operation

from psyclone.autodiff import assign_zero, assign
from psyclone.autodiff.transformations import ADAssignmentTrans


class ADForwardAssignmentTrans(ADAssignmentTrans):
    """A class for automatic differentation transformations of Assignment nodes \
    using forward-mode.
    Requires an ADForwardRoutineTrans instance as context, where the derivative symbols \
    can be found.
    If the RHS is an Operation node, `apply` applies an `ADForwardOperationTrans` to it.
    """

    def apply(self, assignment, options=None):
        """Applies the transformation, generating the transformed \
        statement associated with this Assignment.
            
        Options:
        - bool 'verbose' : toggles preceding and inline comments around the derivative \
            of the assignment in the transformed motion.

        :param assignment: node to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :return: list containing the transformed statements.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Assignment`]
        """
        self.validate(assignment, options)

        # verbose option adds comments to the first and last returning statements
        verbose = self.unpack_option("verbose", options)

        # DataNodes on both sides
        lhs = assignment.lhs
        rhs = assignment.rhs

        # Returned list
        transformed = []

        # Derivative symbol of LHS
        lhs_d = self.routine_trans.data_symbol_derivative_map[lhs.symbol]

        if isinstance(rhs, Literal):
            # RHS is a constant:
            #   LHS derivative is set to 0
            lhs_d_zero = assign_zero(lhs_d)
            transformed.append(lhs_d_zero)

        elif isinstance(rhs, Reference):
            # RHS is a reference:
            #   LHS derivative is set to RHS derivative
            rhs_d = self.routine_trans.data_symbol_derivative_map[rhs.symbol]
            lhs_d_assignment = assign(lhs_d, rhs_d)
            transformed.append(lhs_d_assignment)

        elif isinstance(rhs, Operation):
            # RHS is an operation:
            #   LHS derivative is set to operation derivative
            result = self.routine_trans.operation_trans.apply(rhs, options)
            lhs_d_assignment = assign(lhs_d, result)
            transformed.append(lhs_d_assignment)

        # TODO: rhs is Call to function
        # elif isinstance(rhs, Call):

        else:
            raise NotImplementedError(
                f"Transforming an Assignment with rhs "
                f"of type '{type(rhs).__name__}' is not "
                f"implemented yet."
            )

        # Append the original assignement
        transformed.append(assignment.copy())

        # Verbose mode adds comments to the derivative statement
        # TODO: writer should be initialization argument of the (container?) transformation
        if verbose:
            from psyclone.psyir.backend.fortran import FortranWriter

            fwriter = FortranWriter()
            src = fwriter(assignment.copy())
            # indexing to remove the line break
            transformed[0].preceding_comment = f"Derivating {src[:-1]}"

        return transformed
