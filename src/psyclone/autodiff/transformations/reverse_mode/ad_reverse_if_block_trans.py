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
differentiation of PSyIR IfBlock nodes."""

from psyclone.psyir.nodes import (
    Assignment,
    Call,
    ArrayReference,
    IfBlock,
    OMPRegionDirective,
)
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADIfBlockTrans


class ADReverseIfBlockTrans(ADIfBlockTrans):
    """A class for automatic differentation transformations of IfBlock \
    nodes using reverse-mode.
    Requires an ADReverseRoutineTrans instance as context, where the adjoint \
    symbols can be found.
    Applying it returns both the recording and returning motions associated to \
    the transformed IfBlock.
    """

    def validate(self, if_block, control_tape_ref, options=None):
        """Validates the arguments of the `apply` method.

        :param if_block: node to be transformed.
        :type if_block: :py:class:`psyclone.psyir.nodes.IfBlock`
        :param control_tape_ref: reference to the boolean value of the \
                                 condition in the control tape.
        :type control_tape_ref: :py:class:`psyclone.psyir.nodes.ArrayReference`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if if_block is of the wrong type.
        :raises TransformationError: if control_tape_ref is of the wrong type.
        """
        # pylint: disable=arguments-renamed

        super().validate(if_block, options)

        if not isinstance(control_tape_ref, ArrayReference):
            raise TransformationError(
                f"'control_tape_ref' argument should be a "
                f"PSyIR 'ArrayReference' but found "
                f"'{type(control_tape_ref).__name__}'."
            )

    def apply(self, if_block, control_tape_ref, options=None):
        """Applies the transformation, generating the recording and returning \
        motions associated to this IfBlock.
        
        ************************************
        ************************************
        ************************************
        TODO: DESCRIBE THIS
        ************************************
        ************************************
        ************************************
            
        | Options:
        | - bool 'verbose' : toggles preceding and inline comments around the \
                           adjoining of the assignment in the returning motion.

        :param if_block: node to be transformed.
        :type if_block: :py:class:`psyclone.psyir.nodes.IfBlock`
        :param control_tape_ref: reference to the boolean value of the \
                                 condition in the control tape.
        :type control_tape_ref: :py:class:`psyclone.psyir.nodes.ArrayReference`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this IfBlock.
        :rtype: :py:class:`psyclone.psyir.nodes.IfBlock`, \
                :py:class:`psyclone.psyir.nodes.IfBlock`
        """
        # pylint: disable=arguments-renamed
        self.validate(if_block, control_tape_ref, options)

        # Transform the if body to get both motions
        (recording_if_block, returning_if_block) = self.transform_body(
            if_block.if_body, options
        )
        returning_if_block.reverse()

        # If it exists, transform the else body to get both motions
        if if_block.else_body is not None:
            (recording_else_block, returning_else_block) = self.transform_body(
                if_block.else_body, options
            )
            returning_else_block.reverse()
        else:
            recording_else_block = returning_else_block = None

        recording = IfBlock.create(
            control_tape_ref.copy(), recording_if_block, recording_else_block
        )

        returning = IfBlock.create(
            control_tape_ref.copy(), returning_if_block, returning_else_block
        )

        return recording, returning

    def transform_body(self, body, options=None):
        """Transforms all statements found in an if/else body and generates \
        both the recording and returning motions.

        :param body: body to transform, as a PSyIR Schedule.
        :type body: :py:class:`psyclone.psyir.nodes.Schedule`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: couple formed of two lists of transformed nodes, the first \
                 for the recording motion and the second for the returning one.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`],
                List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        recording_body = []
        returning_body = []

        for child in body.children:
            # Transform the child, get lists of nodes for both motions
            if isinstance(child, Assignment):
                (recording, returning) = (
                    self.routine_trans.transform_assignment(child, options)
                )
            elif isinstance(child, Call):
                (recording, returning) = self.routine_trans.transform_call(
                    child, options
                )
            elif isinstance(child, OMPRegionDirective):
                (recording, returning) = (
                    self.routine_trans.omp_region_trans.apply(child, options)
                )
            elif isinstance(child, IfBlock):
                # Tape record the condition value first
                control_tape_record = self.routine_trans.control_tape.record(
                    child.condition
                )
                recording = [control_tape_record]

                # Get the ArrayReference of the control tape element
                control_tape_ref = self.routine_trans.control_tape.restore(
                    child.condition
                )
                (rec, returning) = self.routine_trans.transform_if_block(
                    child, control_tape_ref, options
                )
                recording.extend(rec)
            else:
                raise NotImplementedError(
                    f"Transformations for "
                    f"'{type(child).__name__}' found in "
                    f"IfBlock body were not implemented "
                    f"yet."
                )

            # Add to recording motion in same order
            recording_body.extend(recording)

            # Add to returning motion in reversed order and before the existing
            # statements
            returning.reverse()
            returning_body.extend(returning)

        return recording_body, returning_body
