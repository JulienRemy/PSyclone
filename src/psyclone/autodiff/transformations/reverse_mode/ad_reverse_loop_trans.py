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

from psyclone.psyir.nodes import Assignment, Call, Reference, IfBlock, Loop
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADLoopTrans


class ADReverseLoopTrans(ADLoopTrans):
    """A class for automatic differentation transformations of Loop \
    nodes using reverse-mode.
    Requires an ADReverseRoutineTrans instance as context, where the adjoint \
    symbols can be found.
    Applying it returns both the recording and returning motions associated to \
    the transformed Loop.
    """

    def validate(self, loop, options=None):
        """Validates the arguments of the `apply` method.

        :param loop: node to be transformed.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if loop is of the wrong type.
        """
        # pylint: disable=arguments-renamed

        super().validate(loop, options)

    def apply(self, loop, options=None):
        """Applies the transformation, generating the recording and returning \
        motions associated to this Loop.
        
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

        :param loop: node to be transformed.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this Loop.
        :rtype: :py:class:`psyclone.psyir.nodes.Loop`, \
                :py:class:`psyclone.psyir.nodes.Loop`
        """
        # pylint: disable=arguments-renamed
        self.validate(loop, options)

        # Transform the statements found in the for loop body (Schedule)
        recording_body, returning_body = self.transform_children(loop, options)

        # Get the loop start, stop and step reversed
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        # TODO: this is wrong, simply to check basic transfo features
        #       reverse the bounds
        # rev_start, rev_stop, rev_step = self.reverse_bounds(loop, options)
        rev_start, rev_stop, rev_step = (loop.start_expr.copy(),
                                         loop.stop_expr.copy(),
                                         loop.step_expr.copy())
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        


        # Postprocess (simplify, substitute operation adjoints) the returning
        # routine
        #self.postprocess(self.returning, options)

        recording_loop = Loop.create(loop.variable,
                                     loop.start_expr.copy(),
                                     loop.stop_expr.copy(),
                                     loop.step_expr.copy(),
                                     recording_body)

        returning_loop = Loop.create(loop.variable, ########## use a different one !
                                     rev_start,
                                     rev_stop,
                                     rev_step,
                                     returning_body)
        
        ##################
        # TODO
        # Need to count n = the records on each tape for one iteration
        # Need to assign the right value to the tape do_offset using the loop
        # var and n
        # Should not tape undef values at first iteration
        
        # If the do_offset symbol of a tape was used in the bodies,
        # it needs to be defined at the beginning of each iteration (in both)
        # for ref in recording_loop.loop_body.walk(Reference):
        #     if ref.symbol == self.routine_trans.value_tape:
        #         # do_offset = n * (i - start)/step
        #         var_minus_start = sub(loop.variable, loop.start_expr)
        #         value_tape_def = assign()
        #         recording_loop.loop_body.addchild(value_tape_def, 0)
        #         ##################################
        #         # returning_loop 
        #         ##################################

        return recording_loop, returning_loop

    ############################################################################
    ############################################################################
    # TODO: all transform_... methods below are almost the same as the ones
    #       in ADReverseRoutineTrans, except for the do_loop argument to the
    #       tape. They should be combined.
    ############################################################################
    ############################################################################

    def transform_children(self, loop, options=None):
        """Transforms all the children of the loop body being transformed \
        and returns lists of Nodes for the recording and returning loops.

        :param loop: the loop to be transformed.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises NotImplementedError: if assignment is done to the loop variable.
        :raises NotImplementedError: if the child transformation is not \
                                     implemented yet. For now only those for \
                                     Assignment, IfBlock, Loop and Call \
                                     instances are.
        """
        recording_body = []
        returning_body = []

        for child in loop.loop_body.children:
            # Transform the child, get lists of nodes for both motions
            if isinstance(child, Assignment):
                if child.lhs.symbol == loop.variable:
                    raise NotImplementedError("Can't yet transform a Loop with "
                                              "assignments to its loop "
                                              "variable.")
                (recording,
                 returning) = self.transform_assignment(child,
                                                        options)
            elif isinstance(child, Call):
                # TODO: this should check if the called subroutine might modify
                #       the loop variable, but that requires checking intents...
                (recording,
                 returning) = self.transform_call(child,
                                                  options)
            elif isinstance(child, IfBlock):
                (recording,
                 returning) = self.transform_if_block(child,
                                                      options)
            elif isinstance(child, Loop):
                (recording,
                 returning) = self.apply(child, options)
                recording = [recording]
                returning = [returning]
            else:
                raise NotImplementedError(f"Transformations for "
                                          f"'{type(child).__name__}' found in "
                                          f"Loop body were not implemented "
                                          f"yet.")

            # Add to recording motion in same order
            recording_body.extend(recording)

            # Add to returning motion in reversed order and before the existing
            # statements
            returning.reverse()
            returning_body = returning + returning_body

        return recording_body, returning_body

    def transform_assignment(self, assignment, options=None):
        """Transforms an Assignment child of the loop and returns the \
        statements to add to the recording and returning loop bodies.

        :param assignment: assignment to transform.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if assignment is of the wrong type.

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this Assignment.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Assignment`], \
                List[:py:class:`psyclone.psyir.nodes.Assignment`]
        """
        if not isinstance(assignment, Assignment):
            raise TypeError(
                f"'assignment' argument should be of "
                f"type 'Assignment' but found"
                f"'{type(assignment).__name__}'."
            )

        overwriting = self.routine_trans.is_overwrite(assignment.lhs)

        recording = []
        returning = []

        # TODO: if NOT overwriting this should NOT tape at first iteration???

        # Tape record and restore first
        if overwriting:
            value_tape_record \
                = self.routine_trans.value_tape.record(assignment.lhs,
                                                       do_loop = True)
            recording.append(value_tape_record)

            value_tape_restore \
                = self.routine_trans.value_tape.restore(assignment.lhs,
                                                        do_loop = True)
            returning.append(value_tape_restore)

        # Apply the transformation
        rec, ret = self.routine_trans.assignment_trans.apply(assignment,
                                                             options)
        recording.extend(rec)
        returning.extend(ret)

        return recording, returning

    def transform_call(self, call, options=None):
        """Transforms a Call child of the routine and adds the \
        statements to the recording and returning routines.

        :param call: call to transform.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if call is of the wrong type.

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this Call.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`], \
                List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of "
                f"type 'Call' but found"
                f"'{type(call).__name__}'."
            )

        recording = []
        returning = []

        # Tape record/restore the Reference arguments of the Call
        # Symbols already value_taped due to this call
        # to avoid taping multiple times if it appears as multiple
        # arguments of the call
        value_taped_symbols = []
        # accumulate the restores for now
        value_tape_restores = []
        for arg in call.children:
            if isinstance(arg, Reference):
                # Check whether the lhs variable was written before
                # or if it is an argument of a call before
                overwriting = self.routine_trans.is_overwrite(arg)
                # TODO: this doesn't deal with the intents
                # of the routine being called for now
                # ie. intent(in) doesn't need to be value_taped?
                # see self.is_call_argument_before

                # TODO: if NOT overwriting this should NOT tape at first 
                #       iteration???

                if overwriting:
                    # Symbol wasn't value_taped yet
                    if arg.symbol not in value_taped_symbols:
                        # Tape record in the recording routine
                        value_tape_record = self.routine_trans \
                                                .value_tape \
                                                .record(arg,
                                                        do_loop = True)
                        recording.append(value_tape_record)

                        # Associated value_tape restore in the returning routine
                        value_tape_restore = self.routine_trans \
                                                 .value_tape \
                                                 .restore(arg,
                                                          do_loop = True)
                        value_tape_restores.append(value_tape_restore)

                        # Don't value_tape the same symbol again in this call
                        value_taped_symbols.append(arg.symbol)

        # Apply an ADReverseCallTrans
        rec, returning = self.routine_trans.call_trans.apply(call,
                                                             options)

        # Add transformed call to recording statements
        recording.extend(rec)

        # Add value tapes restores before returning statements
        returning = value_tape_restores + returning

        return recording, returning

    def transform_if_block(self, if_block, options=None):
        """Transforms an IfBlock child of the routine and adds the \
        statements to the recording and returning routines.

        :param if_block: if block to transform.
        :type if_block: :py:class:`psyclone.psyir.nodes.IfBlock`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if if_block is of the wrong type.

        :return: couple composed of the recording and returning motions \
                 that correspond to the transformation of this IfBlock.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`], \
                List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        if not isinstance(if_block, IfBlock):
            raise TypeError(
                f"'if_block' argument should be of "
                f"type 'IfBlock' but found"
                f"'{type(if_block).__name__}'."
            )

        recording = []
        returning = []

        # Tape record the condition value first
        control_tape_record \
            = self.routine_trans.control_tape.record(if_block.condition,
                                                     do_loop = True)
        recording.append(control_tape_record)

        # Get the ArrayReference of the control tape element
        control_tape_ref \
            = self.routine_trans.control_tape.restore(if_block.condition,
                                                      do_loop = True)

        # Apply the transformation
        rec, ret = self.routine_trans.if_block_trans.apply(if_block,
                                                           control_tape_ref,
                                                           options)

        recording.append(rec)
        returning.append(ret)

        return recording, returning
