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

from psyclone.psyir.nodes import (
    Assignment,
    Call,
    Reference,
    IfBlock,
    Loop,
    Operation,
    ArrayReference,
    Range,
    BinaryOperation,
    Routine,
    IntrinsicCall,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADLoopTrans

from psyclone.autodiff import (
    sub,
    mul,
    assign,
    div,
    add,
    minus,
    one,
    zero,
    add_datanodes,
    substract_datanodes,
    multiply_datanodes,
)


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
        Checks that the loop variable, stop or step bounds are not modified
        (by being LHS of assignments) in the loop body.

        :param loop: node to be transformed.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if loop is of the wrong type.
        :raises NotImplementedError: if the loop contains assignments to the
                                     loop variable, stop or step bound.
        """
        # pylint: disable=arguments-renamed

        super().validate(loop, options)

        # TODO: conditional branches in loops are not implemented yet.
        if loop.loop_body.walk(IfBlock) != []:
            raise NotImplementedError(
                "Loops containing conditional branches are not implemented yet."
            )

        if not self._is_only_nested_loops(loop):
            raise NotImplementedError(
                "For now, only stencil-like nested loops, where only the "
                "innermost one has children other than a single loop, are "
                "supported."
            )

        # Go through the nested loops, get all references in the stop and step
        # bounds expression to check that they are not modified in the loop body
        for nested_loop in loop.walk(Loop):
            if isinstance(nested_loop.stop_expr, (Operation, Call)):
                stop_refs = nested_loop.stop_expr.walk(Reference)
            elif isinstance(nested_loop.stop_expr, Reference):
                stop_refs = [nested_loop.stop_expr]
            else:
                stop_refs = []

            if isinstance(nested_loop.step_expr, (Operation, Call)):
                step_refs = nested_loop.step_expr.walk(Reference)
            elif isinstance(nested_loop.step_expr, Reference):
                step_refs = [nested_loop.step_expr]
            else:
                step_refs = []

            for assignment in nested_loop.loop_body.walk(Assignment):
                if assignment.lhs.symbol == nested_loop.variable:
                    raise NotImplementedError(
                        "Loops that modify their iteration "
                        "variable are not implemented yet."
                    )
                for ref in stop_refs:
                    if assignment.lhs.symbol == ref.symbol:
                        raise NotImplementedError(
                            "Loops that modify their stop "
                            "bound are not implemented "
                            "yet."
                        )
                for ref in step_refs:
                    if assignment.lhs.symbol == ref.symbol:
                        raise NotImplementedError(
                            "Loops that modify their step "
                            "bound are not implemented yet."
                        )

    def apply(self, loop, options=None):
        """Applies the transformation, generating the recording and returning \
        motions associated to this Loop.
        The returning loop is reversed.
        The loops cannot modify their loop variables, stop or step expression \
        inside their bodies.
        Where possible (quite restrictive conditions for now), the arrays are \
        taped and restored before the loop.
            
        | Options:
        | - bool 'verbose' : toggles preceding and inline comments around the \
                           adjoining of the assignment in the returning motion.

        :param loop: node to be transformed.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: couple composed of the recording and returning motions \
                 as lists of nodes (tape records/restores and transformed loop).
        :rtype: List[:py:class:`psyclone.psyir.nodes.Loop`,
                     :py:class:`psyclone.psyir.nodes.Assignment`], \
                List[:py:class:`psyclone.psyir.nodes.Loop`,
                     :py:class:`psyclone.psyir.nodes.Assignment`]
        """
        # pylint: disable=arguments-renamed
        self.validate(loop, options)

        recording = []
        returning = []

        # First make sure the offset is assigned
        value_tape_offset_assignment_before = (
            self.routine_trans.value_tape.offset_assignment
        )

        #################
        # Taping some arrays outside of the nested loop (under some conditions)
        #################
        # Some arrays can be taped outside under some
        arrays_to_tape_outside = self._list_arrays_to_tape_outside(loop)

        # Get all nested loops variables and bounds
        loops_vars_and_bounds = self.nested_loops_variables_and_bounds(loop)
        loops_vars = self.nested_loops_variables(loop)
        nodes_taped_outside = []
        # For all LHS arrays in assignments
        for written_array_ref in arrays_to_tape_outside:
            # TODO: TBR analysis

            # NOTE: tbr stands for to be recorded
            is_tbr = True

            if is_tbr:
                tbr_slice = []
                # Go through the indices they are being written at
                for index in written_array_ref.children:
                    index_sym = index.symbol
                    # For indices that are loop variables
                    if index_sym in loops_vars:
                        # Get the bounds of the loop with that same variable
                        # and make a range from them
                        for var, start, stop, step in loops_vars_and_bounds:
                            if var == index_sym:
                                slice_range = Range.create(
                                    start.copy(), stop.copy(), step.copy()
                                )
                                tbr_slice.append(slice_range)
                                break
                    # For indices that are not loop variable, use the ref
                    else:
                        tbr_slice.append(index.copy())

                # Create an ArrayReference to that range/slice for the array
                # to be recorded
                tbr_array_ref = ArrayReference.create(
                    written_array_ref.symbol, tbr_slice
                )

                # Add the tape records and restores operations to the correct
                # motions
                recording.append(
                    self.routine_trans.value_tape.record(tbr_array_ref)
                )
                returning.insert(
                    0, self.routine_trans.value_tape.restore(tbr_array_ref)
                )

                # Keep track of the nodes that were taped outside
                nodes_taped_outside.append(tbr_array_ref)

        # If some nodes are taped outside, make sure the offset from *before*
        # they are taped is inserted *before* the tape records in the returning
        # motion
        if len(nodes_taped_outside) != 0:
            returning.insert(0, value_tape_offset_assignment_before)

        ##########################################
        # Now deal with the loop body itself (and its internal taping)
        recording_offsets_and_loop = []
        returning_offsets_and_loop = []

        # If some nodes were taped outside, the offset changed and should be
        # defined *before* the loop body in *both* motions
        if len(nodes_taped_outside) != 0:
            # Update the tape offset and tape mask
            value_tape_offset_assignment = (
                self.routine_trans.value_tape.update_offset_and_mask(
                    one(),
                    nodes_taped_outside,
                    [one()] * len(nodes_taped_outside),
                )
            )
            recording_offsets_and_loop.append(value_tape_offset_assignment)
            returning_offsets_and_loop.append(
                value_tape_offset_assignment.copy()
            )
        else:
            returning_offsets_and_loop.append(
                value_tape_offset_assignment_before
            )
        #################
        # Transform the loop body and create loops for both motions
        #################
        # Get the nodes that were recorded to the tapes before this loop
        # or outside of it
        number_of_previously_taped_nodes = len(
            self.routine_trans.value_tape.recorded_nodes
        )

        # Transform the statements found in the for loop body (Schedule)
        # This creates tape record/restore statements and extends the tapes
        recording_body, returning_body = self.transform_children(
            loop, arrays_to_tape_outside, options
        )
        returning_body.reverse()

        # Get the loop start, stop and step reversed
        rev_start, rev_stop, rev_step = self.reverse_bounds(loop, options)

        # Create the recording loop (same var, same bounds)
        recording_loop = Loop.create(
            loop.variable,
            loop.start_expr.copy(),
            loop.stop_expr.copy(),
            loop.step_expr.copy(),
            recording_body,
        )

        # Create the returning loop (same var, reversed bounds)
        returning_loop = Loop.create(
            loop.variable,  # Use same var or different one ?
            rev_start,
            rev_stop,
            rev_step,
            returning_body,
        )

        # verbose option adds comments to the returning do loop
        # specifying the original loop bounds
        verbose = self.unpack_option("verbose", options)

        if verbose:
            self._verbose_comment_loop(returning_loop, loop)

        #################
        # Insert the transformed loops in the motions
        #################
        recording_offsets_and_loop.append(recording_loop)
        returning_offsets_and_loop.append(returning_loop)  # + returning

        ##################
        # TODO
        # Should not tape undef values at first iteration,
        # rather than a conditional branch it should probably extract the
        # first iteration

        #################
        # Deal with the nodes which were taped inside the loop body, as they
        # need to be taken into account for offsetting the indices
        #################
        # Get the newly recorded nodes
        nodes_taped_inside = self.routine_trans.value_tape.recorded_nodes[
            number_of_previously_taped_nodes:
        ]
        new_multiplicities = self.routine_trans.value_tape.multiplicities[
            number_of_previously_taped_nodes:
        ]

        # If some taping went on inside the loop body, then the offset needs
        # to be updated
        if len(nodes_taped_inside) != 0:
            # Update the tape offset and tape mask
            value_tape_offset_assignment = (
                self.routine_trans.value_tape.update_offset_and_mask(
                    self.number_of_iterations(loop),
                    nodes_taped_inside,
                    new_multiplicities,
                )
            )
            recording_offsets_and_loop.append(value_tape_offset_assignment)

            ######
            # The do offset (used within the loop, dependent on the loops vars)
            # needs to be computed and defined at the start of each iteration

            # Multiply the iteration counter by the length of all
            # recorded nodes
            product = mul(
                self.iteration_counter_within_innermost_nested_loop(loop),
                self.routine_trans.value_tape.length_of_nodes_with_multiplicities(
                    nodes_taped_inside, new_multiplicities
                ),
            )
            symbol = self.routine_trans.value_tape.do_offset_symbol
            value_do_offset_assignment = assign(symbol, product)

            # For nested loop, the assignment goes in the innermost loop
            # body so walk and get last
            innermost_recording_loop = recording_loop.walk(Loop)[-1]
            innermost_recording_loop.loop_body.addchild(
                value_do_offset_assignment, 0
            )
            innermost_returning_loop = returning_loop.walk(Loop)[-1]
            innermost_returning_loop.loop_body.addchild(
                value_do_offset_assignment.copy(), 0
            )

        recording.extend(recording_offsets_and_loop)
        returning = returning_offsets_and_loop + returning

        return recording, returning

    def _verbose_comment_loop(self, returning_loop, loop):
        """Add a verbose comment to returning_loop, using information from the \
        original loop, specifying the original bounds.

        :param returning_loop: transformed loop to be commented.
        :type returning_loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param loop: original (primal) loop.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        """
        #pylint: disable=import-outside-toplevel
        from psyclone.psyir.backend.fortran import FortranWriter

        fwriter = FortranWriter()
        start = fwriter(loop.start_expr)
        stop = fwriter(loop.stop_expr)
        step = fwriter(loop.step_expr)
        verbose_comment = (
            f"Reversing do loop 'do {loop.variable.name} "
            + f"= {start}, {stop}, {step}'"
        )
        returning_loop.preceding_comment = verbose_comment

    def reverse_bounds(self, loop, options=None):
        """Reverses the start, stop and step values for the returning loop.

        :param loop: node to be transformed.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: start, stop and step PSyIR node expressions, reversed.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`, \
                :py:class:`psyclone.psyir.nodes.DataNode`, \
                :py:class:`psyclone.psyir.nodes.DataNode`
        """
        # If step is '1' then go from stop to start by -1 steps
        if loop.step_expr == one():
            return loop.stop_expr.copy(), loop.start_expr.copy(), minus(one())

        # Last loop iteration is step*((stop - start)/step) + start
        # (using integer division, could use MOD otherwise)
        substraction = sub(loop.stop_expr, loop.start_expr)
        division = div(substraction, loop.step_expr)
        product = mul(loop.step_expr, division)
        addition = add(product, loop.start_expr)
        rev_start = addition
        rev_stop = loop.start_expr.copy()
        rev_step = minus(loop.step_expr.copy())
        return rev_start, rev_stop, rev_step

    def nested_loops_variables_and_bounds(self, loop):
        """Get a list of [variable, start, stop, step] elements for all nested \
        loops within loop, including it. 

        :param loop: loop to walk.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`

        :return: list of lists containing the variable symbol, start, stop and \
                 step expressions of each nested loop.
        :rtype: List[List[:py:class:`psyclone.psyir.symbols.DataSymbol`,
                          :py:class:`psyclone.psyir.nodes.DataNode`,
                          :py:class:`psyclone.psyir.nodes.DataNode`,
                          :py:class:`psyclone.psyir.nodes.DataNode`]]
        """
        # Includes the current loop
        nested_loops = loop.walk(Loop)

        variables_and_bounds = []
        for nested_loop in nested_loops:
            variables_and_bounds.append(
                [
                    nested_loop.variable,
                    nested_loop.start_expr,
                    nested_loop.stop_expr,
                    nested_loop.step_expr,
                ]
            )

        return variables_and_bounds

    def nested_loops_variables(self, loop):
        """Get a list of loop variable symbols for all nested loops within \
        loop, including it. 

        :param loop: loop to walk.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`

        :return: list of lists containing the variable symbol, start, stop and \
                 step expressions of each nested loop.
        :rtype: List[:py:class:`psyclone.psyir.symbols.DataSymbol`]
        """
        return [
            var_bounds[0]
            for var_bounds in self.nested_loops_variables_and_bounds(loop)
        ]

    def _single_loop_number_of_iterations(self, start, stop, step):
        """Computes the total number of iterations of a single loop based on \
        its bounds.

        :param start: start expression of the loop.
        :type start: :py:class:`psyclone.psyir.nodes.DataNode`
        :param stop: stop expression of the loop.
        :type stop: :py:class:`psyclone.psyir.nodes.DataNode`
        :param step: step expression of the loop.
        :type step: :py:class:`psyclone.psyir.nodes.DataNode`

        :return: total number of iterations performed by the loop.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        if step == one():
            if start == one():
                return stop.copy()

            return substract_datanodes([stop, one()], [start])

        # (step * ((stop - start + 1)/step) + start)/step
        # TODO: some literal simplifications
        return div(
            add_datanodes(
                [
                    mul(
                        step,
                        div(substract_datanodes([stop, one()], [start]), step),
                    ),
                    start,
                ]
            ),
            step,
        )

    def number_of_iterations(self, loop):
        """Computes the total number of iterations of a nested loop.

        :param loop: nested loop.
        :type start: :py:class:`psyclone.psyir.nodes.Loop`

        :return: total number of iterations performed by the nested loops.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        variables_and_bounds = self.nested_loops_variables_and_bounds(loop)

        _, start, stop, step = variables_and_bounds[0]
        number_of_iterations = self._single_loop_number_of_iterations(
            start, stop, step
        )

        if len(variables_and_bounds) > 1:
            for _, start, stop, step in variables_and_bounds[1:]:
                number_of_iterations = multiply_datanodes(
                    [
                        number_of_iterations,
                        self._single_loop_number_of_iterations(
                            start, stop, step
                        ),
                    ],
                )

        return number_of_iterations

    def _single_loop_current_iteration(self, var, start, _, step):
        """Computes the iteration counter on a single loop based on its \
        loop variable and bounds.
        This starts at 0.

        :param var: loop variable as a symbol.
        :type var: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param start: start expression of the loop.
        :type start: :py:class:`psyclone.psyir.nodes.DataNode`
        :param stop: stop expression of the loop.
        :type stop: :py:class:`psyclone.psyir.nodes.DataNode`
        :param step: step expression of the loop.
        :type step: :py:class:`psyclone.psyir.nodes.DataNode`

        :return: iteration counter for the loop.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        # (i - start)/step
        if step == one():
            if start == zero():
                return Reference(var)
            return sub(var, start)

        # TODO: some literal simplifications
        return div(sub(var, start), step)

    def _iteration_counter_from_variables_and_bounds(
        self, variables_and_bounds
    ):
        """Takes a list of loop variables and bounds and returns an iteration \
        counter. The list should be order from the outermost to the innermost \
        loop.
        eg. for a loop with variable i containing a loop with variable j,
        given [[var_i, start_i, stop_i, step_i], 
               [var_j, start_j, stop_j, step_j]]
        this method returns the counter to be used inside the j loop.

        :param variables_and_bounds: list of lists containing the loop \
                                     variable as a symbol and the loop bounds \
                                    as datanodes.
        :type variables_and_bounds: \
                    List[List[:py:class:`psyclone.psyir.symbols.DataSymbol`,
                              :py:class:`psyclone.psyir.nodes.DataNode`,
                              :py:class:`psyclone.psyir.nodes.DataNode`,
                              :py:class:`psyclone.psyir.nodes.DataNode`]

        :return: iteration counter for the loops.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        # if not isinstance(variables_and_bounds, list):
        #     raise TypeError("")
        # for var_bounds in variables_and_bounds:
        #     if not isinstance(var_bounds, list):
        #         raise TypeError("")
        #     if len(var_bounds) != 4:
        #         raise TypeError("")
        #     var = var_bounds[0]
        #     bounds = var_bounds[1:]
        #     if not isinstance(var, DataSymbol):
        #         raise TypeError("")
        #     for bound in bounds:
        #         if not isinstance(bound, DataNode):
        #             raise TypeError("")

        if variables_and_bounds == []:
            return zero()

        variables_and_bounds.reverse()

        inner_var, start, stop, step = variables_and_bounds[0]
        current_iteration = self._single_loop_current_iteration(
            inner_var, start, stop, step
        )
        inner_max = self._single_loop_number_of_iterations(start, stop, step)

        if len(variables_and_bounds) > 1:
            for var, start, stop, step in variables_and_bounds[1:]:
                loop_iteration = mul(
                    self._single_loop_current_iteration(var, start, stop, step),
                    inner_max,
                )
                loop_max = self._single_loop_number_of_iterations(
                    start, stop, step
                )
                current_iteration = add(loop_iteration, current_iteration)
                inner_max = mul(loop_max, inner_max)

        return current_iteration

    # def iteration_counter_within_this_loop(self, loop):
    #     """Computes the iteration counter to be used in this loop.
    #     The 'loop' argument is the current one. Takes into account the \
    #     surrounding loops.

    #     :param loop: current loop.
    #     :type loop: :py:class:`psyclone.psyir.nodes.Loop`

    #     :return: iteration counter for this loop.
    #     :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
    #     """
    #     variables_and_bounds = self.surrounding_loops_variables_and_bounds(loop)
    #     return self._iteration_counter_from_variables_and_bounds(
    #         variables_and_bounds
    #     )

    def iteration_counter_within_innermost_nested_loop(self, loop):
        """Computes the iteration counter to be used within the innermost \
        nested loop within 'loop'.
        The 'loop' argument is the outermost (purely) nested one.

        :param loop: outermost nested loop.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`

        :return: iteration counter for the nested loops.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        # nested i, j, k => iter_i * (max_j * max_k) + iter_j * max_k + iter_k
        # variables_and_bounds = self.surrounding_loops_variables_and_bounds(loop)[:-1]
        variables_and_bounds = self.nested_loops_variables_and_bounds(loop)
        return self._iteration_counter_from_variables_and_bounds(
            variables_and_bounds
        )

    def _is_only_nested_loops(self, loop):
        """Checks if only the innermost nested loop contains nodes different \
        from a loop.

        :param loop: loop to walk.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`

        :return: True if only nested loops, False otherwise.
        :rtype: bool
        """
        # This contains loop itself
        nested_loops = loop.walk(Loop)
        # If only loop, then True
        if len(nested_loops) == 1:
            return True
        # Otherwise, check all the nested loops, except for the innermost one,
        # contain only one child (a loop)
        else:
            for nested_loop in nested_loops[:-1]:
                if len(nested_loop.loop_body.children) != 1:
                    return False
        return True

    # def _all_lhs_of_assignments_are_array_references(self, loop):
    #     """Check if the LHS of assignment nodes are all ArrayReferences.

    #     :param loop: loop to be checked.
    #     :type loop: :py:class:`psyclone.psyir.nodes.Loop`
    #     :return: whether all LHS of assignment nodes verify this property.
    #     :rtype: bool
    #     """
    #     # Get all the LHS of assignements and check them
    #     for assignment in loop.walk(Assignment):
    #         if not isinstance(assignment.lhs, ArrayReference):
    #             return False
    #     return True

    # def _all_arrays_are_indexed_on_lhs_without_offsets(self, loop):
    #     """Check if the arrays are indexed using the exact loop(s) variable(s) \
    #     on the LHS of assignements ie. without offsets, eg. `array(i,j) = ...` \
    #     is fine but `array(i+1, j) = ...` is not).

    #     :param loop: loop to be checked.
    #     :type loop: :py:class:`psyclone.psyir.nodes.Loop`
    #     :return: whether all arrays verify this property.
    #     :rtype: bool
    #     """
    #     # Get the symbols of all nested loops
    #     nested_loops_vars = self.nested_loops_variables(loop)
    #     # Get all the LHS of assignements
    #     for assignment in loop.walk(Assignment):
    #         # If an ArrayReference, make sure that indexing uses exactly the
    #         # same set of symbols as the nested loops
    #         if isinstance(assignment.lhs, ArrayReference):
    #             indices_symbols = [
    #                 index.symbol for index in assignment.lhs.children
    #             ]
    #             if set(nested_loops_vars) != set(indices_symbols):
    #                 return False
    #                 # raise NotImplementedError(
    #                 #     "For now, only nested loops "
    #                 #     "which write to an array at "
    #                 #     "indices which are exactly "
    #                 #     "the loop variables are "
    #                 #     "supported."
    #                 # )
    #     return True

    # def _all_arrays_can_be_taped_outside_the_loops(self, loop):
    #     """Check if the arrays are all written to or incremented *once* and \
    #     *before* being read from. If so, only their post-values are used in \
    #     adjoint computations, which means that their pre-values can be \
    #     restored from the tape after exiting the loop(s) body.

    #     :param loop: loop to be checked.
    #     :type loop: :py:class:`psyclone.psyir.nodes.Loop`
    #     :return: whether all arrays verify this property.
    #     :rtype: bool
    #     """
    #     all_assignments = loop.walk(Assignment)
    #     # Get all the LHS of assignments
    #     lhs_of_assignments = []
    #     for assignment in all_assignments:
    #         lhs_of_assignments.append(assignment.lhs)

    #     # Check for all LHS, for all assignments up to this one
    #     for i, lhs in enumerate(lhs_of_assignments):
    #         # only for LHS which are array elements
    #         if isinstance(lhs_of_assignments, ArrayReference):
    #             assignments_up_to_this_one = all_assignments[: i + 1]
    #             rhs_of_assignments = [
    #                 assignment.rhs for assignment in assignments_up_to_this_one
    #             ]
    #             # For these RHS, only increments are allowed. Check that the path
    #             # from the lhs to the rhs is only made of BinaryOperation nodes
    #             # with '+' operator.
    #             # NOTE: b(i) = b(i) + a(i)**2 + w
    #             # parses as b(i) = (b(i) + a(i)**2) + w, hence the path
    #             for rhs in rhs_of_assignments:
    #                 rhs_array_refs = rhs.walk(ArrayReference)
    #                 if lhs in rhs_array_refs:
    #                     for ref in rhs_array_refs:
    #                         if ref == lhs:
    #                             indices = ref.path_from(lhs.parent)
    #                             cursor = lhs.parent
    #                             for index in indices[:-1]:
    #                                 cursor = cursor.children[index]
    #                                 # print(cursor.view())
    #                                 if not isinstance(cursor, BinaryOperation):
    #                                     return False
    #                                 if (
    #                                     cursor.operator
    #                                     is not BinaryOperation.Operator.ADD
    #                                 ):
    #                                     return False
    #                     # return False
    #                     # raise NotImplementedError("For now only loops were "
    #                     #                         "references "
    #                     #                         "that are written to are written "
    #                     #                         "BEFORE they are read are "
    #                     #                         "supported "
    #                     #                         "due to tape restores being made "
    #                     #                         "after the loop.")

    #             # Also ensure that this is on LHS only once
    #             if lhs_of_assignments.count(lhs) != 1:
    #                 return False
    #                 # raise NotImplementedError("For now only one write to every "
    #                 #                           "reference per loop body is "
    #                 #                           "supported.")
    #     return True

    def _list_arrays_to_tape_outside(self, loop):
        """List all arrays that use the nested loop variables as \
        indices (without increment or decrement), are written only once and \
        are either written to or incremented before being read from.

        :param loop: nested loop to look outside.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`

        :return: list of array references
        :rtype: List[:py:class:`psyclone.psyir.nodes.ArrayReference`]
        """
        # Get the symbols of all nested loops
        # nested_loops_vars = self.nested_loops_variables(loop)

        all_calls = loop.walk(Call)
        all_calls = [
            call for call in all_calls if type(call) is not IntrinsicCall
        ]
        if len(all_calls) != 0:
            return []

        all_assignments = loop.walk(Assignment)
        # Get all the LHS of assignments to ArrayReferences
        lhs_of_assignments = []
        for assignment in all_assignments:
            lhs_of_assignments.append(assignment.lhs)

        arrays_which_can_be_taped_outside = []

        # Check for all LHS
        for i, lhs in enumerate(lhs_of_assignments):
            # Ensure that this is on LHS only once
            if lhs_of_assignments.count(lhs) != 1:
                continue

            # Only consider assignments to array elements
            if not isinstance(lhs, ArrayReference):
                continue

            # Only allow Reference indices (ie. not operations, literals, etc)
            indices_are_refs = [
                isinstance(index, Reference) for index in lhs.children
            ]
            if not all(indices_are_refs):
                continue

            # # Only consider those that are indexed using exactly (some of)
            # # the nested loops variables
            # indices_symbols = [index.symbol for index in lhs.children]
            # if (
            #     len(set(indices_symbols).difference(set(nested_loops_vars)))
            #     != 0
            # ):
            #     continue

            can_be_taped_outside = True

            # Check all assignments up to there
            assignments_up_to_this_one = all_assignments[: i + 1]
            rhs_of_assignments = [
                assignment.rhs for assignment in assignments_up_to_this_one
            ]
            # For these RHS, only increments are allowed. Check that the path
            # from the lhs to the rhs is only made of BinaryOperation nodes
            # with '+' operator.
            # NOTE: b(i) = b(i) + a(i)**2 + w
            # parses as b(i) = (b(i) + a(i)**2) + w, hence the path
            for rhs in rhs_of_assignments:
                rhs_array_refs = rhs.walk(ArrayReference)
                if lhs in rhs_array_refs:
                    for ref in rhs_array_refs:
                        if ref == lhs:
                            if ref.ancestor(Assignment) != lhs.parent:
                                can_be_taped_outside = False
                            else:
                                indices = ref.path_from(lhs.parent)
                                cursor = lhs.parent
                                for index in indices[:-1]:
                                    cursor = cursor.children[index]
                                    # print(cursor.view())
                                    if not isinstance(cursor, BinaryOperation):
                                        can_be_taped_outside = False
                                    if (
                                        cursor.operator
                                        is not BinaryOperation.Operator.ADD
                                    ):
                                        can_be_taped_outside = False

            if can_be_taped_outside:
                # print(lhs.symbol, "can be taped outside")
                arrays_which_can_be_taped_outside.append(lhs)

        return arrays_which_can_be_taped_outside

    ############################################################################
    ############################################################################
    # TODO: all transform_... methods below are almost the same as the ones
    #       in ADReverseRoutineTrans, except for the do_loop argument to the
    #       tape. They should be combined.
    ############################################################################
    ############################################################################

    def transform_children(self, loop, arrays_to_tape_outside, options=None):
        """Transforms all the children of the loop body being transformed \
        and returns lists of Nodes for the recording and returning loops.

        :param loop: the loop to be transformed.
        :type loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param arrays_to_tape_outside: list of array references that should \
                                       not be taped inside the loop when \
                                       assigned to.
        :type arrays_to_tape_outside: 
                           List[:py:class:`psyclone.psyir.nodes.ArrayReference`]
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
                    raise NotImplementedError(
                        "Can't yet transform a Loop with "
                        "assignments to its loop "
                        "variable."
                    )
                (recording, returning) = self.transform_assignment(
                    child, arrays_to_tape_outside, options
                )
            elif isinstance(child, Call):
                # raise NotImplementedError(
                #     "Calls in loop " "bodies are not implemented yet."
                # )
                # TODO: this should check if the called subroutine might modify
                #       the loop variable, but that requires checking intents...
                (recording, returning) = self.transform_call(child, options)
            elif isinstance(child, IfBlock):
                raise NotImplementedError(
                    "Conditional branches in loop "
                    "bodies are not implemented yet."
                )
                (recording, returning) = self.transform_if_block(child, options)
            elif isinstance(child, Loop):
                #################
                # TODO: might benefit from a LoopTrans split
                #################
                if self._is_only_nested_loops(loop):
                    (recording, returning) = self.transform_inner_loop(
                        child, arrays_to_tape_outside, options
                    )
                else:
                    (recording, returning) = self.apply(child, options)
            else:
                raise NotImplementedError(
                    f"Transformations for "
                    f"'{type(child).__name__}' found in "
                    f"Loop body were not implemented "
                    f"yet."
                )

            # Add to recording motion in same order
            recording_body.extend(recording)

            # Add to returning motion in reversed order and after the existing
            # statements
            returning.reverse()
            # returning_body = returning + returning_body

            returning_body.extend(returning)

        return recording_body, returning_body

    def transform_inner_loop(
        self, inner_loop, arrays_to_tape_outside, options=None
    ):
        """Transform an inner loop in case of nested loops.

        :param inner_loop: inner loop to transform.
        :type inner_loop: :py:class:`psyclone.psyir.nodes.Loop`
        :param arrays_to_tape_outside: list of array references that should \
                                       not be taped inside the loop when \
                                       assigned to.
        :type arrays_to_tape_outside: 
                           List[:py:class:`psyclone.psyir.nodes.ArrayReference`]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: recording inner loop, returning inner loop.
        :rtype: :py:class:`psyclone.psyir.nodes.Loop`,
                :py:class:`psyclone.psyir.nodes.Loop`
        """

        # Get the loop start, stop and step reversed
        rev_start, rev_stop, rev_step = self.reverse_bounds(inner_loop, options)

        recording_body, returning_body = self.transform_children(
            inner_loop, arrays_to_tape_outside, options
        )

        returning_body.reverse()

        recording_loop = Loop.create(
            inner_loop.variable,
            inner_loop.start_expr.copy(),
            inner_loop.stop_expr.copy(),
            inner_loop.step_expr.copy(),
            recording_body,
        )

        returning_loop = Loop.create(
            inner_loop.variable,  # Use same var or different one ?
            rev_start,
            rev_stop,
            rev_step,
            returning_body,
        )

        # verbose option adds comments to the returning do loop
        # specifying the original loop bounds
        verbose = self.unpack_option("verbose", options)

        if verbose:
            self._verbose_comment_loop(returning_loop, inner_loop)

        return [recording_loop], [returning_loop]

    def transform_assignment(
        self, assignment, arrays_to_tape_outside, options=None
    ):
        """Transforms an Assignment child of the loop and returns the \
        statements to add to the recording and returning loop bodies.

        :param assignment: assignment to transform.
        :type assignment: :py:class:`psyclone.psyir.nodes.Assignement`
        :param arrays_to_tape_outside: list of array references that should \
                                       not be taped inside the loop when \
                                       assigned to.
        :type arrays_to_tape_outside: 
                           List[:py:class:`psyclone.psyir.nodes.ArrayReference`]
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

        recording = []
        returning = []

        # print("Assignment with lhs", assignment.lhs.name)

        # If taping is done inside
        if assignment.lhs.symbol not in [
            array_ref.symbol for array_ref in arrays_to_tape_outside
        ]:

            # TODO: if NOT overwriting this should NOT tape at first iteration
            # Should extract the first iteration rather than using a conditional
            # branch?

            # Tape record and restore first
            value_tape_record = self.routine_trans.value_tape.record(
                assignment.lhs, do_loop=True
            )
            recording.append(value_tape_record)

            value_tape_restore = self.routine_trans.value_tape.restore(
                assignment.lhs, do_loop=True
            )
            returning.append(value_tape_restore)

        # Apply the transformation
        rec, ret = self.routine_trans.assignment_trans.apply(
            assignment, options
        )
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

        # Call RoutineSymbol
        call_symbol = call.routine
        # Routine
        routine = self.routine_trans.container_trans.routine_from_symbol(
            call_symbol
        )
        routine_arguments = routine.symbol_table.argument_list

        # Tape record/restore the Reference arguments of the Call
        # Symbols already value_taped due to this call
        # to avoid taping multiple times if it appears as multiple
        # arguments of the call
        value_taped_symbols = []
        # accumulate the restores for now
        value_tape_restores = []
        for call_arg, routine_arg in zip(call.children, routine_arguments):
            if isinstance(call_arg, Reference):
                # TODO: if NOT overwriting this should NOT tape at first
                #       iteration???

                # intent(in) arguments of the routine cannot be modified,
                # so they do not need to be taped.
                # All other intents can be modifed and need to be taped.
                if (
                    routine_arg.interface.access
                    is not ArgumentInterface.Access.READ
                ):
                    # Symbol wasn't value_taped yet
                    if call_arg.symbol not in value_taped_symbols:
                        # Tape record in the recording routine
                        value_tape_record = (
                            self.routine_trans.value_tape.record(
                                call_arg, do_loop=True
                            )
                        )
                        recording.append(value_tape_record)

                        # Associated value_tape restore in the returning routine
                        value_tape_restore = (
                            self.routine_trans.value_tape.restore(
                                call_arg, do_loop=True
                            )
                        )
                        value_tape_restores.append(value_tape_restore)

                        # Don't value_tape the same symbol again in this call
                        value_taped_symbols.append(call_arg.symbol)

        # Apply an ADReverseCallTrans
        rec, returning = self.routine_trans.call_trans.apply(call, options)

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
        control_tape_record = self.routine_trans.control_tape.record(
            if_block.condition, do_loop=True
        )
        recording.append(control_tape_record)

        # Get the ArrayReference of the control tape element
        control_tape_ref = self.routine_trans.control_tape.restore(
            if_block.condition, do_loop=True
        )

        # Apply the transformation
        rec, ret = self.routine_trans.if_block_trans.apply(
            if_block, control_tape_ref, options
        )

        recording.append(rec)
        returning.append(ret)

        return recording, returning
