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

from psyclone.psyir.nodes import (IfBlock, Assignment, Call, Loop, Literal,
                                  Reference, Operation, IntrinsicCall)

from psyclone.autodiff.transformations import ADLoopTrans
from psyclone.autodiff import assign, increment


class ADForwardLoopTrans(ADLoopTrans):
    """A class for automatic differentation transformations of Loop \
    nodes using forward-mode.
    Requires an ADForwardRoutineTrans instance as context, where the \
    derivative symbols can be found.
    """

    def apply(self, loop, options=None):
        """Applies the transformation, generating the transformed \
        statements associated with this Loop.
        Copies the loop header as is and recursively transforms statements \
        found in the loop body.
        If the start [and step] variables are References or Operations, uses \
        the correct derivatives for the loop variables.
            
        | Options:
        | - bool 'verbose' : toggles preceding and inline comments around the \
            derivatives of assignment statements in the transformed motion.

        :param loop: loop to be transformed.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: transformed loop.
        :rtype: :py:class:`psyclone.psyir.nodes.Loop`
        """
        self.validate(loop, options)

        #result = []

        ## First consider the start of the loop
        ## If it's a literal, the loop variable has null derivative, proceed
        #if isinstance(loop.start_expr, Literal):
        #    pass
        ## If it's a reference, operation or intrinsic call, the loop variable
        ## does have a derivative.
        ## Fake a "loop_var = start" assignement, transform it and put the
        ## transformed statements before the transformed loop body
        #elif isinstance(loop.start_expr, (Reference, Operation, IntrinsicCall)):
        #    fake_assignement = assign(loop.variable,
        #                              loop.start_expr.copy())
        #    trans = self.routine_trans.assignment_trans.apply(fake_assignement)
        #
        #    verbose = self.unpack_option("verbose", options)
        #    if verbose:
        #        trans[0].preceding_comment = "Using derivative of start var"
        #
        #    # Exclude the fake assignment itself from what's added
        #    result.extend(trans[:-1])
        #else:
        #    raise NotImplementedError(f"Found a "
        #                              f"'{type(loop.start_expr).__name__}' "
        #                              f"node as start variable of a Loop node, "
        #                              f"this is not implemented yet.")

        # Now transform the body of the loop
        transformed_body = []

        ## First consider the step of the loop
        ## If it's a literal, the loop variable has null derivative, proceed
        #if isinstance(loop.step_expr, Literal):
        #    pass
        ## If it's a reference, operation or intrinsic call, the loop variable
        ## does have a derivative.
        ## Fake a "loop_var = loop_var + step" assignement, transform it and
        ## put the transformed statements at the start of the transformed loop
        ## body
        #elif isinstance(loop.step_expr, (Reference, Operation, IntrinsicCall)):
        #    fake_increment = increment(loop.variable, loop.step_expr.copy())
        #    trans = self.routine_trans.assignment_trans.apply(fake_increment)
        #
        #    verbose = self.unpack_option("verbose", options)
        #    if verbose:
        #        trans[0].preceding_comment = "Using derivative of step var"
        #
        #    # Exclude the fake increment itself from what's added
        #    transformed_body.extend(trans[:-1])
        #else:
        #    raise NotImplementedError(f"Found a "
        #                              f"'{type(loop.start_expr).__name__}' "
        #                              f"node as step variable of a Loop node, "
        #                              f"this is not implemented yet.")

        for node in loop.loop_body.children:
            if isinstance(node, Assignment):
                transformed_body.extend(
                    self.routine_trans.assignment_trans.apply(node, options)
                    )
            elif isinstance(node, Call):
                transformed_body.extend(
                    self.routine_trans.call_trans.apply(node, options)
                    )
            elif isinstance(node, IfBlock):
                transformed_body.append(
                    self.routine_trans.if_block_trans.apply(node, options)
                    )
            elif isinstance(node, Loop):
                transformed_body.append(self.apply(node, options))
            else:
                raise NotImplementedError(f"Transformations for "
                                          f"'{type(node).__name__}' found in "
                                          f"Loop body were not implemented "
                                          f"yet.")

        ## Now extend the list of nodes with the transformed loop and return
        #result.append(Loop.create(loop.variable,
        #                          loop.start_expr.copy(),
        #                          loop.stop_expr.copy(),
        #                          loop.step_expr.copy(),
        #                          transformed_body))

        return Loop.create(loop.variable,
                                  loop.start_expr.copy(),
                                  loop.stop_expr.copy(),
                                  loop.step_expr.copy(),
                                  transformed_body)
