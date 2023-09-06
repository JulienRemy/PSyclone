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
differentiation of PSyIR Assignment nodes."""

from psyclone.psyir.nodes import Literal, Reference, Operation
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff import assign_zero, assign, increment
from psyclone.autodiff.transformations import ADAssignmentTrans


class ADReverseAssignmentTrans(ADAssignmentTrans):
    """A class for automatic differentation transformations of Assignment \
    nodes using reverse-mode.
    Requires an ADReverseRoutineTrans instance as context, where the adjoint \
    symbols can be found.
    Applying it returns both the recording and returning motions associated to \
    the transformed Assignment.
    """

    # TODO: iterative assignments should deal with EQUIVALENCE once implemented
    def apply(self, assignment, options=None):
        """Applies the transformation, generating the recording and returning \
        motions associated to this Assignment.
        If the RHS is an Operation node, `apply` applies an \
        `ADReverseOperationTrans` to it.

        The `recording` motion is a copy of the Assignment.

        The `returning` motion is built in the following way.
        If the RHS is a Literal node, this sets the LHS adjoint to 0.
        If the RHS is a Reference node:
        - if it is also the LHS, it does nothing.
        - otherwise it increments the RHS adjoint by the LHS one, \
            then sets the LHS adjoint to 0.
        If the RHS is an Operation node, this applies an \
            `ADReverseOperationTrans` to it, then using its results:
        - it increments the adjoints of all Reference nodes on the RHS, \
            *except for the LHS one if the assignment is iterative*,
        - it sets the LHS adjoint to 0,
        - it increments the LHS adjoint due to its RHS occurences if this was \
            an iterative assignment.
            
        | Options:
        | - bool 'verbose' : toggles preceding and inline comments around the \
                           adjoining of the assignment in the returning motion.

        :param assignment: node to be transformed.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this Assignment.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Assignment`], \
                List[:py:class:`psyclone.psyir.nodes.Assignment`]
        """
        self.validate(assignment, options)

        # verbose option adds comments to the first and last returning
        # statements
        verbose = self.unpack_option("verbose", options)
        verbose_comment = ""

        # DataNodes on both sides
        lhs = assignment.lhs
        rhs = assignment.rhs

        # Adjoint symbol of LHS
        lhs_adj = self.routine_trans.data_symbol_differential_map[lhs.symbol]

        # In any case the LHS adjoint will be set to 0
        # after it's been used in incrementing the adjoints of variables
        # on the RHS
        lhs_adj_zero = assign_zero(lhs_adj)

        # Copy the assignment for the recording routine
        recording = [assignment.copy()]

        # List of statements for the returning routine
        returning = []

        if isinstance(rhs, Literal):
            # RHS is a constant:
            #   LHS is restored from value_tape (in the ADReverseRoutineTrans)
            #   LHS adjoint is set to 0

            returning.append(lhs_adj_zero)

        elif isinstance(rhs, (Reference, Operation)):
            # RHS is a Reference/Operation:
            #   - not iterative:
            #       - LHS is restored from value_tape if needed
            #           (in the ADReverseScheduleTrans)
            #       - (all) RHS adjoint(s) are incremented, using the LHS adjoint
            #       - LHS adjoint is set to 0
            #   - iterative:
            #       - LHS is restored from value_tape if needed
            #           (in the ADReverseScheduleTrans)
            #       - (all) RHS adjoint(s) **except the LHS adjoint**
            #           are incremented, using the LHS adjoint
            #       - LHS adjoint is set to 0
            #       - LHS adjoint is incremented for all its RHS occurences

            # TODO: if adjoint aliasing due to memory locations reuse,
            # deal with them using temporary variables
            # Use ADReverseRoutineTrans.new_temp_symbol for it

            # RHS is a variable
            # If it's the LHS one, we'll simply assign zero to its adjoint after
            if isinstance(rhs, Reference):
                # This is not an iterative assignment
                if lhs != rhs:
                    # Adjoint symbol of RHS
                    rhs_adj = self.routine_trans.data_symbol_differential_map[
                        rhs.symbol
                    ]
                    # Increment it
                    rhs_adj_op = increment(rhs_adj, lhs_adj)
                    # Add the incrementation to the returning motion
                    returning.append(rhs_adj_op)

                    # Set the LHS adjoint to zero
                    returning.append(lhs_adj_zero)

                # This is var = var, adjoint is unchanged
                else:
                    if verbose:
                        verbose_comment += ", this is self-assignment"

            else:  # isinstance(rhs, Operation)
                # Apply the ADElementTransOperation to all children of the 
                # operation with parent_adj being the LHS adjoint

                (
                    op_returning,
                    lhs_adj_incrementations,
                ) = self.routine_trans.operation_trans.apply(
                    rhs, lhs_adj, options
                )

                # NOTE: List lhs_adj_incrementations is non-empty
                #   iff the assignment is iterative.
                # It contains the incrementations to the LHS adjoint
                #   for all its RHS occurences.
                # These should be done **last**,
                #   after using the LHS adjoint in incrementing other
                #       RHS adjoints
                # **and** setting the LHS adjoint to 0

                # Add the returning statements
                returning.extend(op_returning)

                # If there are not iterative incrementations to the LHS adjoint,
                # set it to 0
                if len(lhs_adj_incrementations) == 0:
                    returning.append(lhs_adj_zero)

                # Otherwise the first incrementation should be an assignment
                else:
                    # TODO: make this cleaner
                    # We get the second operand of the RHS of the incrementation
                    # and use it as RHS of an assignment
                    incrementation = lhs_adj_incrementations[0]
                    incr = incrementation.rhs.children[1]
                    lhs_adj_assign = assign(lhs_adj, incr)
                    lhs_adj_incrementations[0] = lhs_adj_assign

                    # If this was an iterative statement,
                    # verbose comment on the LHS adjoint incrementations 
                    # coming last
                    if verbose:
                        verbose_comment += ", iterative"
                        lhs_adj_incrementations[
                            0
                        ].preceding_comment = ("Iterative assignment, "
                                               "so LHS adjoint comes last. "
                                               "First assign...")

                        if len(lhs_adj_incrementations) > 1:
                            lhs_adj_incrementations[
                                1
                            ].preceding_comment = "... then increment"

                        lhs_adj_incrementations[
                            -1
                        ].inline_comment = "Finished incrementing LHS adjoint"

                # Now increment the LHS adjoint as needed for its RHS occurences
                returning.extend(lhs_adj_incrementations)

                # TODO: this should always pass, is_iterative method is 
                # not actually needed
                # TODO: drop these once it's certain
                if len(lhs_adj_incrementations) != 0 and not self.is_iterative(
                    assignment
                ):
                    raise TransformationError("Iterative but also not?")
                if len(lhs_adj_incrementations) == 0 and self.is_iterative(
                    assignment
                ):
                    raise TransformationError("Iterative but also not?")

        # TODO: rhs is Call to function
        # elif isinstance(rhs, Call):

        else:
            raise NotImplementedError(
                f"Transforming an Assignment with rhs "
                f"of type '{type(rhs).__name__}' is not "
                f"implemented yet."
            )

        # Verbose mode adds comments to the first and last returning statements
        # TODO: writer should be initialization argument of the (container?) 
        # transformation
        if verbose and len(returning) != 0:
            from psyclone.psyir.backend.fortran import FortranWriter

            fwriter = FortranWriter()
            src = fwriter(assignment.copy())
            # indexing to remove the line break
            returning[
                0
            ].preceding_comment = f"Adjoining {src[:-1]}{verbose_comment}"
            returning[-1].inline_comment = f"Finished adjoining {src}"

        return recording, returning
