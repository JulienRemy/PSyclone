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
differentiation of PSyIR IfBlock nodes."""

from psyclone.psyir.nodes import IfBlock, Assignment, Call, Loop

from psyclone.autodiff.transformations import ADIfBlockTrans


class ADForwardIfBlockTrans(ADIfBlockTrans):
    """A class for automatic differentation transformations of IfBlock \
    nodes using forward-mode.
    Requires an ADForwardRoutineTrans instance as context, where the \
    derivative symbols can be found.
    """

    def apply(self, if_block, options=None):
        """Applies the transformation, generating the transformed \
        statement associated with this IfBlock.
        Copies the if condition as is and recursively transforms statements \
        found in the if and optional else bodies.
            
        | Options:
        | - bool 'verbose' : toggles preceding and inline comments around the \
            derivatives of assignment statements in the transformed motion.

        :param if_block: if block to be transformed.
        :type if_block: :py:class:`psyclone.psyir.nodes.IfBlock`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: transformed if block.
        :rtype: :py:class:`psyclone.psyir.nodes.IfBlock`
        """
        self.validate(if_block, options)

        transformed_if_body = self.transform_body(if_block.if_body, options)
        if if_block.else_body is not None:
            transformed_else_body = self.transform_body(if_block.else_body,
                                                        options)
        else:
            transformed_else_body = None

        return IfBlock.create(if_block.condition.copy(),
                              transformed_if_body,
                              transformed_else_body)

    def transform_body(self, body, options=None):
        """Transforms all statements found in an if/else body.

        :param body: body to transform, as a PSyIR Schedule.
        :type body: :py:class:`psyclone.psyir.nodes.Schedule`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: list of transformed nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        transformed_body = []

        for node in body.children:
            if isinstance(node, Assignment):
                transformed_body.extend(
                    self.routine_trans.assignment_trans.apply(node, options)
                    )
            elif isinstance(node, Call):
                transformed_body.extend(
                    self.routine_trans.call_trans.apply(node, options)
                    )
            elif isinstance(node, IfBlock):
                transformed_body.append(self.apply(node, options))
            elif isinstance(node, Loop):
                transformed_body.append(
                    self.routine_trans.loop_trans.apply(node, options)
                    )
            else:
                raise NotImplementedError(f"Transformations for "
                                          f"'{type(node).__name__}' found in "
                                          f"IfBlock body were not implemented "
                                          f"yet.")

        return transformed_body
