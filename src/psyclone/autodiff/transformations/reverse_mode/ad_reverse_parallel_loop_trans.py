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
differentiation of PSyIR Loop nodes found within an OpenMP parallel region."""

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
    OMPRegionDirective,
    OMPAtomicDirective,
    OMPParallelDirective,
    OMPReductionClause,
)
from psyclone.psyir.symbols import ArgumentInterface, ArrayType
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADReverseLoopTrans

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


class ADReverseParallelLoopTrans(ADReverseLoopTrans):
    """A class for automatic differentation transformations of Loop \
    nodes found within a OpenMP parallel region using reverse-mode.
    Requires an ADReverseRoutineTrans instance as context, where the adjoint \
    symbols can be found.
    Applying it returns both the recording and returning motions associated to \
    the transformed Loop.
    """

    def validate(self, loop, options=None):
        super().validate(loop, options)

        if loop.ancestor(OMPParallelDirective) is None:
            raise TransformationError(
                "'loop' argument should be within an "
                "OpenMP parallel region but is not."
            )

    def list_scalar_adjoint_symbols_to_increment_atomically(
        self, loop, options=None
    ):
        scalar_adjoint_symbols_to_increment_atomically = []
        for assignment in loop.walk(Assignment):
            for ref in assignment.rhs.walk(Reference):
                if not isinstance(ref, ArrayReference):
                    primal_symbol = ref.symbol
                    adjoint_symbol = (
                        self.routine_trans.data_symbol_differential_map[
                            primal_symbol
                        ]
                    )
                    if (
                        adjoint_symbol
                        not in scalar_adjoint_symbols_to_increment_atomically
                    ):
                        scalar_adjoint_symbols_to_increment_atomically.append(
                            adjoint_symbol
                        )

        return scalar_adjoint_symbols_to_increment_atomically

    # TODO: array symbol to array ref map instead
    def build_primal_array_symbol_to_read_indices_map(self, loop, options=None):
        primal_array_symbol_to_read_indices_map = dict()

        for assignment in loop.walk(Assignment):
            for ref in assignment.rhs.walk(Reference):
                if isinstance(ref, ArrayReference):
                    if ref.symbol in primal_array_symbol_to_read_indices_map:
                        primal_array_symbol_to_read_indices_map[
                            ref.symbol
                        ].append(ref.indices)
                    else:
                        primal_array_symbol_to_read_indices_map[ref.symbol] = [
                            ref.indices
                        ]

        return primal_array_symbol_to_read_indices_map

    # TODO: array symbol to array ref map instead
    def build_adjoint_array_symbol_to_increment_indices_map(
        self, loop, options=None
    ):
        primal_array_symbol_to_read_indices_map = (
            self.build_primal_array_symbol_to_read_indices_map(loop, options)
        )

        return {
            self.routine_trans.data_symbol_differential_map[
                primal_symbol
            ]: indices
            for primal_symbol, indices in primal_array_symbol_to_read_indices_map.items()
        }

    def list_array_adjoint_symbols_to_increment_atomically(
        self, loop, options=None
    ):
        adjoint_array_symbols_multi_incremented = []

        adjoint_array_symbol_to_indices_map = (
            self.build_adjoint_array_symbol_to_increment_indices_map(
                loop, options
            )
        )

        for (
            adjoint_symbol,
            indices_list,
        ) in adjoint_array_symbol_to_indices_map.items():
            if len(indices_list) > 1:
                found_different_indices = False
                first_indices = indices_list[0]
                for other_indices in indices_list[1:]:
                    if found_different_indices:
                        break

                    for first_index, other_index in zip(
                        first_indices, other_indices
                    ):
                        if first_index != other_index:
                            adjoint_array_symbols_multi_incremented.append(
                                adjoint_symbol
                            )
                            found_different_indices = True
                            break

        return adjoint_array_symbols_multi_incremented

    def make_adjoint_increments_atomic(
        self, adjoint_symbols, returning_body, options=None
    ):
        for i, statement in enumerate(returning_body):
            for assignment in statement.walk(Assignment):
                if assignment.lhs.symbol in adjoint_symbols:
                    if assignment.lhs in assignment.rhs.children:
                        atomic_assignment = OMPAtomicDirective(
                            children=[assignment.copy()]
                        )
                        if statement is assignment:
                            returning_body[i] = atomic_assignment
                        else:
                            assignment.replace_with(atomic_assignment)
        return returning_body

    # TODO: reductions for scalars (and arrays)
    # def make_scalar_adjoints_reductions(self, adjoint_symbols, omp_region_directive, options=None):
    #     for adjoint_symbol in adjoint_symbols:
    #         reduction_clause = OMPReductionClause(children=[Reference(adjoint_symbol)])

    def make_scalar_adjoint_increments_atomic(
        self, loop, returning_body, options=None
    ):
        scalar_adjoint_symbols_to_increment_atomically = (
            self.list_scalar_adjoint_symbols_to_increment_atomically(
                loop, options
            )
        )
        self.make_adjoint_increments_atomic(
            scalar_adjoint_symbols_to_increment_atomically,
            returning_body,
            options,
        )
        return returning_body

    def make_array_adjoint_increments_atomic(
        self, loop, returning_body, options=None
    ):
        array_adjoint_symbols_to_increment_atomically = (
            self.list_array_adjoint_symbols_to_increment_atomically(
                loop, options
            )
        )
        self.make_adjoint_increments_atomic(
            array_adjoint_symbols_to_increment_atomically,
            returning_body,
            options,
        )
        return returning_body

    def make_all_adjoint_increments_atomic(
        self, loop, returning_body, options=None
    ):
        returning_body = self.make_scalar_adjoint_increments_atomic(
            loop, returning_body, options
        )
        return self.make_array_adjoint_increments_atomic(
            loop, returning_body, options
        )
    
    def compute_gather_like_loop_bounds_from_scatter_like_indices(self, outer_loop, adjoint_array_symbol_to_increments_indices_map,
                                                                  options=None):
        nested_loop_variables_and_bounds = self.nested_loops_variables_and_bounds(outer_loop)
        nested_loop_variables = self.nested_loops_variables(outer_loop)
        nested_loop_variable_names = [var.name for var in nested_loop_variables]

        # TODO, would be better to associate {array symbol: [original array references]} and then proceed from there?
        # TODO, keep track of new indices to new bounds link
        # put (the array symbol and original indices) OR (the original array reference) in new_vars_and_bounds ?
        adjoint_array_symbol_to_gather_like_indices_map = dict()
        new_vars_and_bounds = []

        for adjoint_symbol, indices_list in adjoint_array_symbol_to_increments_indices_map.values():
            new_indices_list = []
            for indices in indices_list:
                gather_like_indices = []

                for index in indices:
                    refs_in_index = index.walk(Reference)
                    if len(refs_in_index) == 0:
                        gather_like_indices.append(index.copy())
                    else:
                        for ref in refs_in_index:
                            if ref.symbol.name in nested_loop_variable_names:
                                gather_like_indices.append(ref.copy())
                                var_i = nested_loop_variables.index(ref.symbol)
                                var_bounds = nested_loop_variables_and_bounds[var_i][1:-1] #exclude step
                                if index == ref:
                                    new_vars_and_bounds.append([ref.symbol, *var_bounds])
                                else:
                                    if not (isinstance(index, BinaryOperation) and index.operator in (BinaryOperation.Operator.ADD, BinaryOperation.Operator.SUB) and ref in index.children):
                                        raise NotImplementedError("Only indices which are +- binary operations on loops variables can be transformed for now.")
                                    children_copy = index.children.copy()
                                    children_copy.remove(ref)
                                    offset = children_copy[0]
                                    for ref in offset.walk(Reference):
                                        if ref.symbol.name in nested_loop_variable_names:
                                            raise NotImplementedError("Offsets containing loop variables are not implemented.")
                                    if index.operator is BinaryOperation.Operator.ADD:
                                        new_vars_and_bounds.append([ref.copy(), *[sub(bound, offset) for bound in var_bounds]])#, var_bounds[2]]),
                                    elif index.operator is BinaryOperation.Operator.SUB:
                                        new_vars_and_bounds.append([ref.copy(), *[add(bound, offset) for bound in var_bounds]])#, var_bounds[2]])
                                break

                new_indices_list.append(gather_like_indices)

            adjoint_array_symbol_to_gather_like_indices_map[adjoint_symbol] = new_indices_list

        var_to_new_bounds_map = {var : [] for var in nested_loop_variables}
        for var, new_start, new_stop in new_vars_and_bounds:
            var_to_new_bounds_map[var].append([new_start, new_stop])
        for var, new_bounds in var_to_new_bounds_map.items():
            if len(new_bounds) == 0:
                var_i = nested_loop_variables.index(var)
                old_bounds = nested_loop_variables_and_bounds[var_i][1:-1] #exclude step
                new_bounds.append(old_bounds)

        # Build intervals
        from sympy.sets.sets import Interval, Union
        from psyclone.psyir.backend.sympy_writer import SymPyWriter, SymPyReader
        sympy_writer = SymPyWriter()
        sympy_reader = SymPyReader(sympy_writer)
        var_to_sympy_intervals_map = dict()
        for var, new_bounds in var_to_new_bounds_map.items():
            new_intervals = []
            for start, stop, step in new_bounds:
                new_intervals.append(Interval(*sympy_writer([start, stop])))
            var_to_sympy_intervals_map[var] = new_intervals

        # TODO: this and union should log which indices are being excluded for which ARRAY
        # - in the intersect case so they can be computed outside
        # - in the union case so they can be made into branches inside
        # Intersect
        var_to_sympy_interval_intersection = dict()
        for var, new_intervals in var_to_sympy_intervals_map.items():
            intersection = new_intervals[0]
            if len(new_intervals) > 1:
                for interval in new_intervals[1:]:
                    intersection = intersection.intersect(interval)
            var_to_sympy_interval_intersection[var] = intersection
        var_to_psyir_intersection_bounds = dict()
        for var, interval_intersection in var_to_sympy_interval_intersection.items():
            sympy_start = interval_intersection.start
            sympy_end = interval_intersection.end
            symbol_table = outer_loop.ancestor(Routine).symbol_table
            start = sympy_reader.psyir_from_expression(sympy_start, symbol_table)
            stop = sympy_reader.psyir_from_expression(sympy_end, symbol_table)
            step = nested_loop_variables_and_bounds[var][2]
            var_to_psyir_intersection_bounds[var] = [start, stop, step]

        # Union
        var_to_sympy_interval_union = dict()
        for var, new_intervals in var_to_sympy_intervals_map.items():
            union = new_intervals[0]
            if len(new_intervals) > 1:
                for interval in new_intervals[1:]:
                    union = union.union(interval)
                if not isinstance(union, Union):
                    raise TransformationError("Interval union should be an Interval.")
            var_to_sympy_interval_union[var] = union

        var_to_psyir_union_bounds = dict()
        for var, interval_union in var_to_sympy_interval_union.items():
            sympy_start = interval_union.start
            sympy_end = interval_union.end
            symbol_table = outer_loop.ancestor(Routine).symbol_table
            start = sympy_reader.psyir_from_expression(sympy_start, symbol_table)
            stop = sympy_reader.psyir_from_expression(sympy_end, symbol_table)
            step = nested_loop_variables_and_bounds[var][2]
            var_to_psyir_union_bounds[var] = [start, stop, step]

        

