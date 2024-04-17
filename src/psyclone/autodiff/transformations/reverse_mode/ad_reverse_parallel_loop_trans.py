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
    Literal,
)
from psyclone.psyir.symbols import ArgumentInterface, ArrayType, ScalarType
from psyclone.psyir.symbols import ArgumentInterface, ArrayType, ScalarType
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

    """
    1/ adjoint array symbol => [scatter-like adjoint increment ASSIGNMENTS] 
        using the returning loop scatter-like, no-atomics instead?
            gets the original reverse bounds too, sympyfy them as a product of intervals, keep the vars !!!
    2/ adjoint array symbol => [scatter-like adjoint increment ASSIGNMENTS, gather-like adjoint increment ASSIGNMENTS, new loop bounds]
    3/ sympy on the loop bounds -> sympy product of intervals
    4a/ intersect, union and list increments that are not in the intersection, so they can either be moved outside or put in branches
    5/ replace the scatter-like increments assignments, dealing with """

    def build_adjoint_array_symbol_to_assignments_map(
        self, returning_outer_loop, options=None
    ):
        adjoint_array_symbol_to_assignments_map = dict()

        for assignment in returning_outer_loop.walk(Assignment):
            if (
                isinstance(assignment.lhs, ArrayReference)
                and assignment.lhs.symbol
                in self.routine_trans.data_symbol_differential_map.values()
            ):
                if (
                    assignment.lhs.symbol
                    in adjoint_array_symbol_to_assignments_map
                ):
                    adjoint_array_symbol_to_assignments_map[
                        assignment.lhs.symbol
                    ].append(assignment)
                else:
                    adjoint_array_symbol_to_assignments_map[
                        assignment.lhs.symbol
                    ] = [assignment]

        return adjoint_array_symbol_to_assignments_map

    def transform_scatter_adjoints_to_gather_adjoints(
        self, advancing_outer_loop, returning_outer_loop, options=None
    ):
        from sympy.sets.sets import Interval, ProductSet, Union, EmptySet
        from psyclone.psyir.backend.sympy_writer import SymPyWriter, SymPyReader

        sympy_writer = SymPyWriter()
        sympy_reader = SymPyReader(sympy_writer)
        symbol_table = advancing_outer_loop.ancestor(Routine).symbol_table

        # For debug
        original_loop = returning_outer_loop.copy()

        returning_loops_vars_and_bounds = (
            self.nested_loops_variables_and_bounds(returning_outer_loop)
        )

        loop_variables = []
        psyir_original_bounds = []
        original_intervals = []
        for var, start, stop, step in returning_loops_vars_and_bounds:
            psyir_original_bounds.append([stop, start])
            original_intervals.append(
                Interval(*sympy_writer([stop, start]))
            )
            loop_variables.append(var)

        original_product_interval = ProductSet(*original_intervals)

        adjoint_array_symbol_to_assignments_map = (
            self.build_adjoint_array_symbol_to_assignments_map(
                returning_outer_loop, options
            )
        )
        adjoint_array_symbol_to_write_indices_list_map = {
            adjoint_array_symbol: [
                assignment.lhs.indices for assignment in assignments
            ]
            for adjoint_array_symbol, assignments in adjoint_array_symbol_to_assignments_map.items()
        }
        # #is_scatter_like = {adjoint_array_symbol : False for adjoint_array_symbol in adjoint_array_symbol_to_assignments_map.keys()}
        scatter_like_symbols_list = []
        for (
            adjoint_array_symbol,
            write_indices_list,
        ) in adjoint_array_symbol_to_write_indices_list_map.items():
            # scatter = False
            first_write_indices = write_indices_list[0]
            if len(write_indices_list) > 1:
                for write_indices in write_indices_list[1:]:
                    if any(
                        [
                            write_index != first_write_index
                            for write_index, first_write_index in zip(
                                write_indices, first_write_indices
                            )
                        ]
                    ):
                        # scatter = True
                        scatter_like_symbols_list.append(adjoint_array_symbol)
                        break
            # is_scatter_like[adjoint_array_symbol] = scatter

        scatters_gathers_and_product_intervals_list = []
        for (
            adjoint_array_symbol,
            original_assignments,
        ) in adjoint_array_symbol_to_assignments_map.items():
            if adjoint_array_symbol in scatter_like_symbols_list:
                transformed_assignments = []
                transformed_product_intervals = []
                original_write_indices_list = (
                    adjoint_array_symbol_to_write_indices_list_map[
                        adjoint_array_symbol
                    ]
                )
                for original_write_indices, original_assignment in zip(
                    original_write_indices_list, original_assignments
                ):
                    transformed_write_indices = []
                    substitution_indices_map = dict()
                    transformed_bounds_map = dict()
                    for original_write_index in original_write_indices:
                        if isinstance(original_write_index, Literal):
                            transformed_write_indices.append(
                                original_write_index.copy()
                            )
                        elif isinstance(original_write_index, Reference):
                            transformed_write_indices.append(
                                original_write_index.copy()
                            )
                            # no need to transform the bound even if a ref to a loop var
                        elif isinstance(original_write_index, BinaryOperation):
                            refs_symbols_in_op = [
                                ref.symbol
                                for ref in original_write_index.walk(Reference)
                            ]
                            loops_vars_in_op = set(loop_variables).intersection(
                                refs_symbols_in_op
                            )
                            if len(loops_vars_in_op) == 0:
                                transformed_write_indices.append(
                                    original_write_index.copy()
                                )
                            elif len(loops_vars_in_op) > 1:
                                raise NotImplementedError(
                                    "Only binary operations on a single loop variable are implemented."
                                )
                            else:
                                loop_var = list(loops_vars_in_op)[0]
                                loop_var_index = loop_variables.index(loop_var)
                                original_stop, original_start = (
                                    psyir_original_bounds[loop_var_index]
                                )
                                if refs_symbols_in_op.count(loop_var) > 1:
                                    raise NotImplementedError(
                                        "Only binary operations countaining a single reference to a loop variable are implemented."
                                    )
                                else:
                                    loop_var_index_in_op = None
                                    for i, child in enumerate(
                                        original_write_index.children
                                    ):
                                        if (
                                            isinstance(child, Reference)
                                            and child.symbol == loop_var
                                        ):
                                            loop_var_index_in_op = i
                                            break
                                    if loop_var_index_in_op is None:
                                        raise NotImplementedError(
                                            "Only binary operation with loop variables as an operand are implemented."
                                        )
                                    other_index_in_op = (
                                        loop_var_index_in_op + 1
                                    ) % 2
                                    other_operand = (
                                        original_write_index.children[
                                            other_index_in_op
                                        ]
                                    )
                                if (
                                    original_write_index.operator
                                    is BinaryOperation.Operator.ADD
                                ):
                                    transformed_write_indices.append(
                                        Reference(loop_var)
                                    )
                                    substitution_indices_map[loop_var] = sub(
                                        Reference(loop_var), other_operand.copy()
                                    )
                                    transformed_bounds_map[loop_var] = [
                                        add(original_stop.copy(), other_operand.copy()),
                                        add(original_start.copy(), other_operand.copy()),
                                    ]
                                elif (
                                    original_write_index.operator
                                    is BinaryOperation.Operator.SUB
                                ):
                                    transformed_write_indices.append(
                                        Reference(loop_var)
                                    )
                                    substitution_indices_map[loop_var] = add(
                                        Reference(loop_var), other_operand.copy()
                                    )
                                    transformed_bounds_map[loop_var] = [
                                        sub(original_stop.copy(), other_operand.copy()),
                                        sub(original_start.copy(), other_operand.copy()),
                                    ]
                                else:
                                    raise NotImplementedError(
                                        "Only + and - operations as indices are implemented."
                                    )

                        else:
                            raise NotImplementedError(
                                "Only literals, references and binary operations as indices are implemented."
                            )

                    transformed_bounds_list = []
                    for loop_var, original_bounds in zip(
                        loop_variables, psyir_original_bounds
                    ):
                        if loop_var in transformed_bounds_map:
                            transformed_bounds_list.append(
                                transformed_bounds_map[loop_var]
                            )
                        else:
                            transformed_bounds_list.append(original_bounds)

                    print(adjoint_array_symbol)
                    print("Original assignment: ", original_assignment.debug_string())
                    print("Original loop bounds: ", [bound.debug_string() for bound in original_bounds])

                    transformed_intervals = [
                        Interval(*sympy_writer([stop, start]))
                        for stop, start in transformed_bounds_list
                    ]

                    sympy_transformed_product_interval = ProductSet(
                        *transformed_intervals
                    )
                    transformed_product_intervals.append(
                        sympy_transformed_product_interval
                    )

                    transformed_lhs = ArrayReference.create(
                        original_assignment.lhs.symbol,
                        transformed_write_indices,
                    )

                    # Still to transform the rhs
                    transformed_assignment = Assignment.create(
                        transformed_lhs, original_assignment.rhs.copy()
                    )

                    for ref in transformed_assignment.rhs.walk(Reference):
                        if ref.symbol in substitution_indices_map:
                            ref.replace_with(
                                substitution_indices_map[ref.symbol].copy()
                            )

                    transformed_assignments.append(transformed_assignment)

                    print("Transformed assignment: ", transformed_assignment.debug_string())
                    print("Transformed bounds: ", [[bound.debug_string() for bound in bounds] for bounds in transformed_bounds_list])
                    print("Transformed interval: ", sympy_transformed_product_interval)

                for (
                        original_assignment,
                        transformed_assignment,
                        transformed_product_interval,
                    ) in zip(
                        original_assignments,
                        transformed_assignments,
                        transformed_product_intervals,
                    ):
                    scatters_gathers_and_product_intervals_list.append([original_assignment,
                        transformed_assignment,
                        transformed_product_interval])

            else:
                for original_assignment in original_assignments:
                    scatters_gathers_and_product_intervals_list.append([
                        original_assignment,
                        original_assignment.copy(),
                        original_product_interval,
                    ])

        product_intervals = []
        for (
            _,
            _,
            product_interval,
        ) in scatters_gathers_and_product_intervals_list:
            product_intervals.append(product_interval)
        for (
            _,
            _,
            product_interval,
        ) in scatters_gathers_and_product_intervals_list:
            product_intervals.append(product_interval)

        product_intervals_intersection = product_intervals[0]
        product_intervals_union = product_intervals[0]
        if len(product_intervals) > 1:
            for product_interval in product_intervals[1:]:
                product_intervals_intersection = (
                    product_intervals_intersection.intersect(product_interval)
                )
                product_intervals_union = product_intervals_union.union(
                    product_interval
                )

        # for (
        #     symbol,
        #     scatters_gathers_and_product_intervals,
        # ) in symbol_to_scatters_gathers_and_product_intervals_map.items():
        #     print(symbol)
        #     for (
        #         scatter,
        #         gather,
        #         product_interval,
        #     ) in scatters_gathers_and_product_intervals:
        #         print(scatter.debug_string())
        #         print(gather.debug_string())
        #         print(product_interval)

        print("Intersection:", product_intervals_intersection)
        print("Union:", product_intervals_union)
        print("Intersection to bounds:", self.sympy_product_set_to_nested_loops_bounds_list(product_intervals_intersection))
        print("Union to bounds: ", self.sympy_product_set_to_nested_loops_bounds_list(product_intervals_union))



        main_gather_loop = returning_outer_loop
        inner_gather_loop = main_gather_loop
        while isinstance(inner_gather_loop.loop_body.children[0], Loop):
            inner_gather_loop = inner_gather_loop.loop_body.children[0]

        main_gather_loop_vars_and_bounds = self.nested_loops_variables_and_bounds(main_gather_loop)

        product_interval_to_scatters_and_gathers_map = dict()
        for scatter, gather, product_interval in scatters_gathers_and_product_intervals_list:
            if product_interval in product_interval_to_scatters_and_gathers_map:
                product_interval_to_scatters_and_gathers_map[product_interval].append([scatter, gather])
            else:
                product_interval_to_scatters_and_gathers_map[product_interval] = [[scatter, gather]]
        for scatter_and_gathers in product_interval_to_scatters_and_gathers_map.values():
            scatter_and_gathers.sort(key = (lambda scatter_and_gather: scatter_and_gather[0].abs_position))

       
        mode = "no_branches"

        # Union and branches mode
        if mode == "branches":
            new_main_gather_loop_bounds = self.sympy_product_set_to_nested_loops_bounds_list(product_intervals_union)
            assert len(new_main_gather_loop_bounds) == 1
            assert len(new_main_gather_loop_bounds[0]) == len(main_gather_loop_vars_and_bounds)
            for [_, old_start, old_stop, _], [new_start, new_stop] in zip(main_gather_loop_vars_and_bounds, new_main_gather_loop_bounds[0]):
                old_start.replace_with(sympy_reader.psyir_from_expression(new_start, symbol_table))
                old_stop.replace_with(sympy_reader.psyir_from_expression(new_stop, symbol_table))

            print("loop was ", original_loop.debug_string())
            print("now is ", main_gather_loop.debug_string())

            # scatters_and_gathers_in_intersection = []
            # gathers_in_intersection = []
            # for scatters_and_gathers in product_interval_to_scatters_and_gathers_map.values():
            #     for scatter, gather in scatters_and_gathers:
            #         if gather not in gathers_in_intersection:
            #             scatters_and_gathers_in_intersection.append([scatter, gather])
            #             gathers_in_intersection.append(gather)

            # product_interval_to_scatters_and_gathers_map[product_intervals_intersection] = scatters_and_gathers_in_intersection

            scatters_to_put_in_branches = [elem[0] for elem in scatters_gathers_and_product_intervals_list]

            # debug
            union_check = EmptySet()

            for assignment in main_gather_loop.walk(Assignment):
                if assignment in scatters_to_put_in_branches:
                    index = scatters_to_put_in_branches.index(assignment)
                    scatter, gather, product_interval = scatters_gathers_and_product_intervals_list[index]

                    print("Product interval:", product_interval)
                    print("\t scatter => gather :", scatter.debug_string() + " => " + gather.debug_string())
                    # debug
                    union_check = EmptySet()

                    if product_intervals_union == product_interval:
                        union_check = Union(union_check, product_interval)
                    else:
                        # if product_interval == product_intervals_intersection:
                        #     gather_indices = product_intervals_intersection
                        # else:
                            #gather_indices = product_intervals_intersection.complement(product_interval)
                        gather_indices = product_interval
                        union_check = Union(union_check, gather_indices)

                        if isinstance(gather_indices, ProductSet):
                            nested_loops_bounds_list = self.sympy_product_set_to_nested_loops_bounds_list(gather_indices)
                        else:
                            raise NotImplementedError("")
                        
                        assert len(nested_loops_bounds_list) == 1
                        nested_loops_bounds = nested_loops_bounds_list[0]
                        
                        if_condition = None
                        for i, (single_loop_bounds, old_loop_var_and_bounds) in enumerate(zip(nested_loops_bounds, returning_loops_vars_and_bounds)):
                            loop_var, _, _, step = old_loop_var_and_bounds

                            new_start, new_stop = single_loop_bounds
                            if new_start == new_stop:
                                loop_value = sympy_reader.psyir_from_expression(new_start, symbol_table)
                                loop_var_condition = BinaryOperation.create(BinaryOperation.Operator.EQ, Reference(loop_var), loop_value)
                            else:
                                new_start = sympy_reader.psyir_from_expression(new_start, symbol_table)
                                new_stop = sympy_reader.psyir_from_expression(new_stop, symbol_table)
                                le = BinaryOperation.create(BinaryOperation.Operator.LE, Reference(loop_var), new_start)
                                ge = BinaryOperation.create(BinaryOperation.Operator.GE, Reference(loop_var), new_stop)
                                loop_var_condition = BinaryOperation.create(BinaryOperation.Operator.AND, le, ge)

                            if i == 0:
                                if_condition = loop_var_condition
                            else:
                                if_condition = BinaryOperation.create(BinaryOperation.Operator.AND, if_condition, loop_var_condition)

                        if_block = IfBlock.create(if_condition, [gather.copy()])
                        scatter.replace_with(if_block)

            print("Union:", product_intervals_union)
            print("Union check:", union_check)

            print(self.make_scalar_adjoint_increments_atomic(advancing_outer_loop, inner_gather_loop.loop_body.children, options))

            print("Loop with branches and scalar atomics is: ", main_gather_loop.debug_string())

            return main_gather_loop


        # Intersection mode
        #mode = "no_branches"
        #mode = "no_branches"
        if mode == "no_branches":

            # # Only keep the scalar adjoints increments in the main gather loop
            # scalar_adjoints_atomic_loop = returning_outer_loop.copy()
            # inner_loop = scalar_adjoints_atomic_loop
            # while isinstance(inner_loop.loop_body.children[0], Loop):
            #     inner_loop = inner_loop.loop_body.children[0]
            
            # scalar_adjoints_symbols_list = []
            # for assignment in original_loop.walk(Assignment):
            #     ref = assignment.lhs
            #     if not isinstance(ref, ArrayReference) and isinstance(ref.datatype, ScalarType) and ref.symbol in self.routine_trans.data_symbol_differential_map.values():
            #         scalar_adjoints_symbols_list.append(ref.symbol)
            # scalar_adjoints_symbols_names_list = [symbol.name for symbol in scalar_adjoints_symbols_list]
            # print("scalar_adjoints_symbols_names_list:", scalar_adjoints_symbols_names_list)
            # nodes_to_detach = []
            # for node in inner_loop.loop_body.children:
            #     print("found node ", node.debug_string())
            #     if not isinstance(node, Assignment):
            #         raise NotImplementedError("")
            #     if node.lhs.symbol.name not in scalar_adjoints_symbols_names_list:
            #         nodes_to_detach.append(node)
            # for node in nodes_to_detach:
            #     node.detach()
            


            # # Detach the scalar adjoints increments from the main gather loop
            # inner_loop = main_gather_loop
            # while isinstance(inner_loop.loop_body.children[0], Loop):
            #     inner_loop = inner_loop.loop_body.children[0]
            # nodes_to_detach = []
            # for node in inner_loop.loop_body.children:
            #     if not isinstance(node, Assignment):
            #         raise NotImplementedError("")
            #     if node.lhs.symbol.name in scalar_adjoints_symbols_names_list:
            #         nodes_to_detach.append(node)
            # for node in nodes_to_detach:
            #     node.detach()


            # returning_outer_loop.detach()

            all_scatters = [elem[0] for elem in scatters_gathers_and_product_intervals_list]
            scalar_adjoint_symbols_to_increment_atomically = self.list_scalar_adjoint_symbols_to_increment_atomically(advancing_outer_loop, options)

            gather_other_statements = []

            new_main_gather_loop_bounds = self.sympy_product_set_to_nested_loops_bounds_list(product_intervals_intersection)
            assert len(new_main_gather_loop_bounds) == 1
            assert len(new_main_gather_loop_bounds[0]) == len(main_gather_loop_vars_and_bounds)
            for [_, old_start, old_stop, _], [new_start, new_stop] in zip(main_gather_loop_vars_and_bounds, new_main_gather_loop_bounds[0]):
                old_start.replace_with(sympy_reader.psyir_from_expression(new_start, symbol_table))
                old_stop.replace_with(sympy_reader.psyir_from_expression(new_stop, symbol_table))

            print("loop was ", original_loop.debug_string())
            print("now is ", main_gather_loop.debug_string())

            new_loops = []
            for product_interval, scatters_and_gathers in product_interval_to_scatters_and_gathers_map.items():
                print("Product interval:", product_interval)
                print("\t scatters => gathers :", [scatter.debug_string() + " => " + gather.debug_string() for scatter, gather in scatters_and_gathers])
                # debug
                union_check = EmptySet()

                if product_intervals_intersection == product_interval:
                    union_check = Union(union_check, product_interval)
                else:
                    gather_indices = product_intervals_intersection.complement(product_interval)

                    if isinstance(gather_indices, Union):
                        nested_loops_bounds_list = self.sympy_union_to_disjunction(gather_indices)
                    elif isinstance(gather_indices, ProductSet):
                        nested_loops_bounds_list = self.sympy_product_set_to_nested_loops_bounds_list(gather_indices)
                    else:
                        raise NotImplementedError("")
                    
                    inner_loops = []
                    for nested_loops_bounds in nested_loops_bounds_list:
                        nested_loop = None
                        inner_loop = None
                        for i, (single_loop_bounds, old_loop_var_and_bounds) in enumerate(zip(nested_loops_bounds, returning_loops_vars_and_bounds)):
                            loop_var, _, _, step = old_loop_var_and_bounds
                            # TODO: no loop for single value
                            new_start, new_stop = single_loop_bounds
                            new_start = sympy_reader.psyir_from_expression(new_start, symbol_table)
                            new_stop = sympy_reader.psyir_from_expression(new_stop, symbol_table)
                            new_loop = Loop.create(loop_var, new_start, new_stop, step.copy(), [])
                            if i == 0:
                                nested_loop = new_loop
                                inner_loop = new_loop
                            else:
                                inner_loop.loop_body.addchild(new_loop)
                                inner_loop = new_loop
                        inner_loops.append(inner_loop)
                        new_loops.append(nested_loop)

                    # debug
                    union_check = Union(union_check, gather_indices)

                    all_scatters_to_substitute = [elem[0] for elem in scatters_and_gathers]

                    original_inner_loop = main_gather_loop
                    while isinstance(original_inner_loop.loop_body.children[0], Loop):
                        original_inner_loop = original_inner_loop.loop_body.children[0]
                    
                    for inner_loop in inner_loops:
                        for statement in original_inner_loop.loop_body.children:
                            if not isinstance(statement, Assignment):
                                raise NotImplementedError("")
                            # Use gathers where they should be in the loop
                            if statement in all_scatters_to_substitute:
                                index = all_scatters_to_substitute.index(statement)
                                scatter, gather = scatters_and_gathers[index]
                                assert scatter == statement
                                inner_loop.loop_body.addchild(gather.copy())
                            # If no gather to use
                            else:
                                # Don't copy other scatters
                                if statement in all_scatters:
                                    pass
                                # If not a scatter
                                else:
                                    # If a scalar adjoint increment, don't copy
                                    if statement.lhs.symbol in scalar_adjoint_symbols_to_increment_atomically:
                                        pass
                                    # Otherwise copy
                                    else:
                                        inner_loop.loop_body.addchild(statement.copy())

                    # for scatter, gather in scatters_and_gathers:
                    #     scatter.detach()
                        
                    #     for inner_loop in inner_loops:
                    #         inner_loop.loop_body.addchild(gather.copy())

            gather_other_statements.extend(new_loops)

            print("Union:", product_intervals_union)
            print("Union check:", union_check)
            print("Union:", product_intervals_union)
            print("Union check:", union_check)

            print("Main gather loop:", main_gather_loop.debug_string())
            print("Other loops and statements:")
            statements_strings = [statement.debug_string() for statement in gather_other_statements]
            for string in statements_strings:
                print(string)

            return main_gather_loop, *gather_other_statements

            # print("Remaining scalar atomic loop:", scalar_adjoints_atomic_loop.debug_string())


    def sympy_finite_set_to_value(self, sympy_finite_set):
        from sympy.sets.sets import Interval, ProductSet, Union, Set, FiniteSet
        if not isinstance(sympy_finite_set, FiniteSet):
            raise TypeError("")
        
        if len(sympy_finite_set) != 1:
            raise NotImplementedError("")
        return sympy_finite_set.args[0]

    def sympy_union_to_disjunction(self, sympy_union):
        from sympy.sets.sets import Interval, ProductSet, Union, Set, FiniteSet
        if not isinstance(sympy_union, Union):
            raise TypeError("")
        disjunction = []
        for arg in sympy_union.args:
            if isinstance(arg, Interval):
                disjunction.append(self.sympy_interval_to_single_loop_bounds(arg))
            elif isinstance(arg, FiniteSet):
                disjunction.append(self.sympy_finite_set_to_value(arg))
            elif isinstance(arg, ProductSet):
                disjunction.extend(self.sympy_product_set_to_nested_loops_bounds_list(arg))
        return disjunction

    def sympy_product_set_to_nested_loops_bounds_list(self, sympy_product_set):
        from sympy.sets.sets import Interval, ProductSet, Union, Set, FiniteSet
        if not isinstance(sympy_product_set, ProductSet):
            raise TypeError("")
        
        #print("Product set", sympy_product_set)
        
        nested_loops_bounds_list = [[]]
        for sympy_set in sympy_product_set.sets:
            if isinstance(sympy_set, Union):
                #print("Current list", nested_loops_bounds_list)
                #print("Union", sympy_set)
                disjunction = self.sympy_union_to_disjunction(sympy_set)
                #print("Disjunction", disjunction)
                new_nested_loops_bounds_list = []
                for case in disjunction:
                    for nested_loops_bounds in nested_loops_bounds_list:
                        new_nested_loops_bounds = nested_loops_bounds.copy()
                        new_nested_loops_bounds.append(case)
                        new_nested_loops_bounds_list.append(new_nested_loops_bounds)
                nested_loops_bounds_list = new_nested_loops_bounds_list
            elif isinstance(sympy_set, Interval):
                single_loop_bounds = self.sympy_interval_to_single_loop_bounds(sympy_set)
                for nested_loops_bounds in nested_loops_bounds_list:
                    nested_loops_bounds.append(single_loop_bounds)
            elif isinstance(sympy_set, FiniteSet):
                single_value = self.sympy_finite_set_to_value(sympy_set)
                for nested_loops_bounds in nested_loops_bounds_list:
                    nested_loops_bounds.append(single_value)
            else:
                raise NotImplementedError(type(sympy_set).__name__)
        
        return nested_loops_bounds_list
            
    def sympy_interval_to_single_loop_bounds(self, sympy_interval):
        from sympy.sets.sets import Interval, ProductSet, Union, Set, FiniteSet
        if not isinstance(sympy_interval, Interval):
            raise TypeError("")

        start = sympy_interval.start
        end = sympy_interval.end
        if sympy_interval.left_open:
            start += 1
        if sympy_interval.right_open:
            end -= 1

        # TODO: no loop for single value
        # if start == end:
        #     return start

        single_loop_bounds = [end, start]

        return single_loop_bounds


    # def compute_gather_like_loop_bounds_from_scatter_like_indices(
    #     self,
    #     outer_loop,
    #     adjoint_array_symbol_to_increments_indices_map,
    #     options=None,
    # ):
    #     nested_loop_variables_and_bounds = (
    #         self.nested_loops_variables_and_bounds(outer_loop)
    #     )
    #     nested_loop_variables = self.nested_loops_variables(outer_loop)
    #     nested_loop_variable_names = [var.name for var in nested_loop_variables]

    #     # TODO, would be better to associate {array symbol: [original array references]} and then proceed from there?
    #     # TODO, keep track of new indices to new bounds link
    #     # put (the array symbol and original indices) OR (the original array reference) in new_vars_and_bounds ?
    #     adjoint_array_symbol_to_gather_like_indices_map = dict()
    #     new_vars_and_bounds = []

    #     for (
    #         adjoint_symbol,
    #         indices_list,
    #     ) in adjoint_array_symbol_to_increments_indices_map.values():
    #         new_indices_list = []
    #         for indices in indices_list:
    #             gather_like_indices = []

    #             for index in indices:
    #                 refs_in_index = index.walk(Reference)
    #                 if len(refs_in_index) == 0:
    #                     gather_like_indices.append(index.copy())
    #                 else:
    #                     for ref in refs_in_index:
    #                         if ref.symbol.name in nested_loop_variable_names:
    #                             gather_like_indices.append(ref.copy())
    #                             var_i = nested_loop_variables.index(ref.symbol)
    #                             var_bounds = nested_loop_variables_and_bounds[
    #                                 var_i
    #                             ][
    #                                 1:-1
    #                             ]  # exclude step
    #                             if index == ref:
    #                                 new_vars_and_bounds.append(
    #                                     [ref.symbol, *var_bounds]
    #                                 )
    #                             else:
    #                                 if not (
    #                                     isinstance(index, BinaryOperation)
    #                                     and index.operator
    #                                     in (
    #                                         BinaryOperation.Operator.ADD,
    #                                         BinaryOperation.Operator.SUB,
    #                                     )
    #                                     and ref in index.children
    #                                 ):
    #                                     raise NotImplementedError(
    #                                         "Only indices which are +- binary operations on loops variables can be transformed for now."
    #                                     )
    #                                 children_copy = index.children.copy()
    #                                 children_copy.remove(ref)
    #                                 offset = children_copy[0]
    #                                 for ref in offset.walk(Reference):
    #                                     if (
    #                                         ref.symbol.name
    #                                         in nested_loop_variable_names
    #                                     ):
    #                                         raise NotImplementedError(
    #                                             "Offsets containing loop variables are not implemented."
    #                                         )
    #                                 if (
    #                                     index.operator
    #                                     is BinaryOperation.Operator.ADD
    #                                 ):
    #                                     new_vars_and_bounds.append(
    #                                         [
    #                                             ref.copy(),
    #                                             *[
    #                                                 sub(bound, offset)
    #                                                 for bound in var_bounds
    #                                             ],
    #                                         ]
    #                                     )  # , var_bounds[2]]),
    #                                 elif (
    #                                     index.operator
    #                                     is BinaryOperation.Operator.SUB
    #                                 ):
    #                                     new_vars_and_bounds.append(
    #                                         [
    #                                             ref.copy(),
    #                                             *[
    #                                                 add(bound, offset)
    #                                                 for bound in var_bounds
    #                                             ],
    #                                         ]
    #                                     )  # , var_bounds[2]])
    #                             break

    #             new_indices_list.append(gather_like_indices)

    #         adjoint_array_symbol_to_gather_like_indices_map[adjoint_symbol] = (
    #             new_indices_list
    #         )

    #     var_to_new_bounds_map = {var: [] for var in nested_loop_variables}
    #     for var, new_start, new_stop in new_vars_and_bounds:
    #         var_to_new_bounds_map[var].append([new_start, new_stop])
    #     for var, new_bounds in var_to_new_bounds_map.items():
    #         if len(new_bounds) == 0:
    #             var_i = nested_loop_variables.index(var)
    #             old_bounds = nested_loop_variables_and_bounds[var_i][
    #                 1:-1
    #             ]  # exclude step
    #             new_bounds.append(old_bounds)

    #     # Build intervals
    #     from sympy.sets.sets import Interval, Union
    #     from psyclone.psyir.backend.sympy_writer import SymPyWriter, SymPyReader

    #     sympy_writer = SymPyWriter()
    #     sympy_reader = SymPyReader(sympy_writer)
    #     var_to_sympy_intervals_map = dict()
    #     for var, new_bounds in var_to_new_bounds_map.items():
    #         new_intervals = []
    #         for start, stop, step in new_bounds:
    #             new_intervals.append(Interval(*sympy_writer([start, stop])))
    #         var_to_sympy_intervals_map[var] = new_intervals

    #     # TODO: this and union should log which indices are being excluded for which ARRAY
    #     # - in the intersect case so they can be computed outside
    #     # - in the union case so they can be made into branches inside
    #     # Intersect
    #     var_to_sympy_interval_intersection = dict()
    #     for var, new_intervals in var_to_sympy_intervals_map.items():
    #         intersection = new_intervals[0]
    #         if len(new_intervals) > 1:
    #             for interval in new_intervals[1:]:
    #                 intersection = intersection.intersect(interval)
    #         var_to_sympy_interval_intersection[var] = intersection
    #     var_to_psyir_intersection_bounds = dict()
    #     for (
    #         var,
    #         interval_intersection,
    #     ) in var_to_sympy_interval_intersection.items():
    #         sympy_start = interval_intersection.start
    #         sympy_end = interval_intersection.end
    #         symbol_table = outer_loop.ancestor(Routine).symbol_table
    #         start = sympy_reader.psyir_from_expression(
    #             sympy_start, symbol_table
    #         )
    #         stop = sympy_reader.psyir_from_expression(sympy_end, symbol_table)
    #         step = nested_loop_variables_and_bounds[var][2]
    #         var_to_psyir_intersection_bounds[var] = [start, stop, step]

    #     # Union
    #     var_to_sympy_interval_union = dict()
    #     for var, new_intervals in var_to_sympy_intervals_map.items():
    #         union = new_intervals[0]
    #         if len(new_intervals) > 1:
    #             for interval in new_intervals[1:]:
    #                 union = union.union(interval)
    #             if not isinstance(union, Union):
    #                 raise TransformationError(
    #                     "Interval union should be an Interval."
    #                 )
    #         var_to_sympy_interval_union[var] = union

    #     var_to_psyir_union_bounds = dict()
    #     for var, interval_union in var_to_sympy_interval_union.items():
    #         sympy_start = interval_union.start
    #         sympy_end = interval_union.end
    #         symbol_table = outer_loop.ancestor(Routine).symbol_table
    #         start = sympy_reader.psyir_from_expression(
    #             sympy_start, symbol_table
    #         )
    #         stop = sympy_reader.psyir_from_expression(sympy_end, symbol_table)
    #         step = nested_loop_variables_and_bounds[var][2]
    #         var_to_psyir_union_bounds[var] = [start, stop, step]
