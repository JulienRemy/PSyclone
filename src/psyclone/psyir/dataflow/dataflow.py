# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2021-2024, Science and Technology Facilities Council.
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
# Author  J. Remy, Université Grenoble Alpes & Inria Grenoble
# -----------------------------------------------------------------------------

from types import NoneType
from psyclone.core import AccessType
from psyclone.psyir.nodes import (
    Operation,
    Directive,
    CodeBlock,
    WhileLoop,
    Return,
    PSyDataNode,
    IntrinsicCall,
    Literal,
    DataNode,
    Schedule,
    Routine,
    Statement,
    Assignment,
    Reference,
    Call,
    Loop,
    IfBlock,
    ArrayReference,
    OMPPrivateClause,
    ACCCopyOutClause,
    OMPFirstprivateClause,
    OMPSharedClause,
    ACCCopyClause,
    ACCCopyInClause,
    OMPDependClause,
    RegionDirective,
)
from psyclone.psyir.symbols import (
    Symbol,
    DataSymbol,
    RoutineSymbol,
    ArgumentInterface,
    ScalarType,
    ArrayType,
)

# TODO: cleanup
# TODO: doc
# TODO: tests


class DataFlowNode:
    def __init__(
        self, dag, psyir, access_type=AccessType.UNKNOWN
    ):  # , forward_dependence):
        if not isinstance(dag, DataFlowDAG):
            raise TypeError("")
        if not isinstance(psyir, (DataNode, DataSymbol)):
            raise TypeError("")
        if isinstance(psyir, DataSymbol) and not isinstance(
            psyir.interface, ArgumentInterface
        ):
            raise TypeError("")
        if not isinstance(access_type, AccessType):
            raise TypeError("")
        if (
            not isinstance(psyir, (Reference, DataSymbol))
            and access_type is not AccessType.UNKNOWN
        ):
            raise ValueError("")
        if isinstance(psyir, (Reference, DataSymbol)) and access_type not in (
            AccessType.READ,
            AccessType.WRITE,
        ):
            raise ValueError("")

        self._psyir = psyir
        self._dag = dag
        self._forward_dependences = []
        self._backward_dependences = []
        self._access_type = access_type

        self.dag.dag_nodes.append(self)

        # if forward_dependence is not None:
        #     self.add_forward_dependence(forward_dependence)

    def add_backward_dependence_to_last_write(self):
        if isinstance(self.psyir, (Reference, DataSymbol)):
            symbol = self.psyir if isinstance(self.psyir, DataSymbol) else self.psyir.symbol
            # Reference to a whole array, should have backward dependences to
            # all last writes to its elements
            if self.psyir.is_array and not isinstance(self.psyir, ArrayReference):
                last_writes_to_ref = self.dag.get_all_last_writes_to_array_symbol(symbol)
                for write in last_writes_to_ref:
                    if write is not self:
                        self.add_backward_dependence(write)
            else:
                last_write_to_ref = self.dag.last_write_to(self.psyir)

                # if self.psyir.name == "c" and isinstance(self.psyir.parent, Call):
                #     print("adding last write to c, got ", last_write_to_ref.psyir)

                if last_write_to_ref is not None and last_write_to_ref is not self:
                    self.add_backward_dependence(last_write_to_ref)

    def recurse_on_children(self):
        psyir = self.psyir
        if isinstance(psyir, DataSymbol):
            pass
        elif isinstance(psyir, Reference):
            pass
        elif isinstance(psyir, ArrayReference):
            # TODO: dependence on indices or not?
            pass
        elif isinstance(psyir, Literal):
            pass
        elif isinstance(psyir, IntrinsicCall):
            # TODO: there are unpure function IntrinsicCall
            if not psyir.is_pure:
                raise NotImplementedError("")
            # child_nodes = []

            # print("intrinsic call:", psyir.debug_string())
            for arg in psyir.children:
                # print("arg:", arg.debug_string())
                arg_node = DataFlowNode.create_or_get(
                    self.dag,
                    arg,
                    (
                        AccessType.READ
                        if isinstance(arg, Reference)
                        else AccessType.UNKNOWN
                    ),
                )
                arg_node.add_forward_dependence(self)
                # child_nodes.append(child_node)

        elif isinstance(psyir, Call):
            called_routine = self.get_called_routine_from_name(
                psyir.routine.name
            )

            # We know the intents of arguments in the called routine
            if called_routine is not None:

                args_symbols = called_routine.symbol_table.argument_list
                args_intents = [
                    symbol.interface.access for symbol in args_symbols
                ]

                # draw_called_routines_dag = True
                # if draw_called_routines_dag:
                #     called_routine_dag = DataFlowDAG.create_from_schedule(called_routine)
                #     # for node in called_routine_dag.dag_nodes:
                #     #     node.copy_single_node_to(self.dag)
                #     called_args_nodes = [called_routine_dag.get_dag_node_for(arg_sym) for arg_sym in args_symbols]

                in_arg_nodes = []
                out_arg_nodes = []
                for i, (arg, intent) in enumerate(
                    zip(psyir.children, args_intents)
                ):
                    if intent is not ArgumentInterface.Access.WRITE:
                        in_arg_node = DataFlowNode.create_or_get(
                            self.dag,
                            arg,
                            (
                                AccessType.READ
                                if isinstance(arg, Reference)
                                else AccessType.UNKNOWN
                            ),
                        )
                        in_arg_node.add_forward_dependence(self)

                        # if draw_called_routines_dag:
                        #     in_arg_node.add_forward_dependence(called_args_nodes[i])

                    if intent is not ArgumentInterface.Access.READ:
                        # NOTE: create to allow for duplicate DAG nodes
                        # with same PSyIR
                        out_arg_node = DataFlowNode.create(
                            self.dag,
                            arg,
                            (
                                AccessType.WRITE
                                if isinstance(arg, Reference)
                                else AccessType.UNKNOWN
                            ),
                        )
                        out_arg_node.add_backward_dependence(self)

                        # if draw_called_routines_dag:
                        #     out_arg_node.add_backward_dependence(called_args_nodes[i])

                        # self.dag._update_last_write(out_arg_node)

            # We don't know the intents of the arguments
            # so treat everything as inout
            else:

                in_arg_nodes = [
                    DataFlowNode.create_or_get(
                        self.dag,
                        arg,
                        (
                            AccessType.READ
                            if isinstance(arg, Reference)
                            else AccessType.UNKNOWN
                        ),
                    )
                    for arg in psyir.children
                ]
                # NOTE: create to allow for duplicate DAG nodes
                # with same PSyIR
                out_arg_nodes = [
                    DataFlowNode.create(
                        self.dag,
                        arg,
                        (
                            AccessType.WRITE
                            if isinstance(arg, Reference)
                            else AccessType.UNKNOWN
                        ),
                    )
                    for arg in psyir.children
                ]
                for in_arg_node in in_arg_nodes:
                    in_arg_node.add_forward_dependence(self)
                for out_arg_node in out_arg_nodes:
                    out_arg_node.add_backward_dependence(self)
                    # self.dag._update_last_write(out_arg_node)

        elif isinstance(psyir, Operation):
            for operand in psyir.children:
                operand_node = DataFlowNode.create_or_get(
                    self.dag,
                    operand,
                    (
                        AccessType.READ
                        if isinstance(operand, Reference)
                        else AccessType.UNKNOWN
                    ),
                )
                self.add_backward_dependence(operand_node)

        elif isinstance(psyir, CodeBlock):
            # TODO?
            raise NotImplementedError("")

        else:
            raise NotImplementedError(type(psyir).__name__)

    def get_called_routine_from_name(self, name):
        # TODO: this looks from the root for a routine, not through imports, enclosing containers, etc
        all_routines = self.psyir.root.walk(Routine)
        called_routine = None
        for routine in all_routines:
            if routine.name == name:
                called_routine = routine
                break

        return called_routine

    def get_intent_from_called_routine(self, called_routine):
        if not isinstance(self.psyir.parent, Call):
            raise TypeError("")
        if not isinstance(called_routine, Routine):
            raise TypeError("")
        call = self.psyir.parent
        if call.routine.name != called_routine.name:
            raise ValueError("")

        arg_index = self.psyir.parent.children.index(self.psyir)
        routine_arg = called_routine.symbol_table.argument_list[arg_index]

        return routine_arg.interface.access

    def get_call_argument_intent(self):
        if not isinstance(self.psyir.parent, Call):
            raise TypeError("")
        call = self.psyir.parent
        called_routine_name = call.routine.name
        called_routine = self.get_called_routine_from_name(called_routine_name)

        # Not found
        if called_routine is None:
            return ArgumentInterface.Access.UNKNOWN

        # Found
        return self.get_intent_from_called_routine(called_routine)

    @classmethod
    def create(cls, dag, psyir, access_type=AccessType.UNKNOWN):
        # Arguments are typechecked by the constructor

        dag_node = cls(dag, psyir, access_type)
        dag_node.add_backward_dependence_to_last_write()
        dag_node.recurse_on_children()

        return dag_node

    @classmethod
    def create_or_get(cls, dag, psyir, access_type=AccessType.UNKNOWN):
        if not isinstance(dag, DataFlowDAG):
            raise TypeError("")
        if not isinstance(psyir, (DataNode, DataSymbol)):
            raise TypeError("")
        if not isinstance(access_type, AccessType):
            raise TypeError("")
        if (
            not isinstance(psyir, (Reference, DataSymbol))
            and access_type is not AccessType.UNKNOWN
        ):
            raise ValueError("")
        if isinstance(psyir, (Reference, DataSymbol)) and access_type not in (
            AccessType.READ,
            AccessType.WRITE,
        ):
            raise ValueError("")

        existing_dag_node = dag.get_dag_node_for(psyir, access_type)
        if existing_dag_node is not None:
            return existing_dag_node
        else:
            return cls.create(dag, psyir, access_type)

    def copy_single_node_to(self, new_dag):
        if not isinstance(new_dag, DataFlowDAG):
            raise TypeError("")

        return DataFlowNode(new_dag, self.psyir, self.access_type)

    def copy_or_get_single_node_to(self, new_dag):
        if not isinstance(new_dag, DataFlowDAG):
            raise TypeError("")

        existing_dag_node = new_dag.get_dag_node_for(
            self.psyir, self.access_type
        )
        if existing_dag_node is not None:
            return existing_dag_node
        else:
            return self.copy_single_node_to(new_dag)

    def add_forward_dependence(self, forward_dependence):
        if not isinstance(forward_dependence, DataFlowNode):
            raise TypeError(type(forward_dependence).__name__)
        if self is forward_dependence:
            raise ValueError("")

        if forward_dependence not in self.forward_dependences:
            self.forward_dependences.append(forward_dependence)
        if self not in forward_dependence.backward_dependences:
            forward_dependence.add_backward_dependence(self)

        # print(self.psyir, "======>", forward_dependence.psyir)
        # print("")

    def add_backward_dependence(self, backward_dependence):
        if not isinstance(backward_dependence, DataFlowNode):
            raise TypeError("")
        if self is backward_dependence:
            raise ValueError("")

        if backward_dependence not in self.backward_dependences:
            self.backward_dependences.append(backward_dependence)
        if self not in backward_dependence.forward_dependences:
            backward_dependence.add_forward_dependence(self)

        # print(self.psyir, "<======", backward_dependence.psyir)
        # print("")

    @property
    def dag(self):
        return self._dag

    @property
    def psyir(self):
        return self._psyir

    @property
    def forward_dependences(self):
        return self._forward_dependences

    @property
    def backward_dependences(self):
        return self._backward_dependences

    @property
    def access_type(self):
        return self._access_type

    @property
    def is_call_argument_reference(self):
        return (
            isinstance(self.psyir, Reference)  # symbols have no parent
            and isinstance(self.psyir.parent, Call)  # should be an arg
        )

    def copy_forward(self, dag_copy=None, originals=[], copies=[]):
        if not isinstance(dag_copy, (DataFlowDAG, NoneType)):
            raise TypeError("")
        if not isinstance(originals, list):
            raise TypeError("")
        if not isinstance(copies, list):
            raise TypeError("")
        if len(originals) != len(copies):
            raise ValueError("")
        for original in originals:
            if not isinstance(original, DataFlowNode):
                raise TypeError("")
        for copy in copies:
            if not isinstance(copy, DataFlowNode):
                raise TypeError("")

        if dag_copy is None:
            dag_copy = DataFlowDAG()

        # Condition due to inout arguments of calls being duplicated in DAG
        if self.is_call_argument_reference:
            copy = self.copy_single_node_to(dag_copy)
        else:
            copy = self.copy_or_get_single_node_to(dag_copy)
        originals.append(self)
        copies.append(copy)
        for fwd in self.forward_dependences:
            if fwd in originals:
                index = originals.index(fwd)
                fwd_copy = copies[index]
            else:
                fwd_copy = fwd.copy_forward(dag_copy, originals, copies)
            copy.add_forward_dependence(fwd_copy)
        return copy

    def copy_backward(self, dag_copy=None, originals=[], copies=[]):
        if not isinstance(dag_copy, (DataFlowDAG, NoneType)):
            raise TypeError("")
        if not isinstance(originals, list):
            raise TypeError("")
        if not isinstance(copies, list):
            raise TypeError("")
        if len(originals) != len(copies):
            raise ValueError("")
        for original in originals:
            if not isinstance(original, DataFlowNode):
                raise TypeError("")
        for copy in copies:
            if not isinstance(copy, DataFlowNode):
                raise TypeError("")

        if dag_copy is None:
            dag_copy = DataFlowDAG()

        if self.is_call_argument_reference:
            copy = self.copy_single_node_to(dag_copy)
        else:
            copy = self.copy_or_get_single_node_to(dag_copy)
        originals.append(self)
        copies.append(copy)
        for bwd in self.backward_dependences:
            if bwd in originals:
                index = originals.index(bwd)
                bwd_copy = copies[index]
            else:
                bwd_copy = bwd.copy_backward(dag_copy, originals, copies)
            copy.add_backward_dependence(bwd_copy)
        return copy

    def to_psyir_list_forward(self):
        psyir_list = [self.psyir]
        for dep in self.forward_dependences:
            dep_psyir_list = dep.to_psyir_list_forward()
            for psyir in dep_psyir_list:
                if psyir not in psyir_list:
                    psyir_list.append(psyir)
            # psyir_list.extend(dep.to_psyir_list_forward())

        return psyir_list

    def to_psyir_list_backward(self):
        psyir_list = [self.psyir]
        for dep in self.backward_dependences:
            dep_psyir_list = dep.to_psyir_list_backward()
            for psyir in dep_psyir_list:
                if psyir not in psyir_list:
                    psyir_list.append(psyir)

        return psyir_list

    def __str__(self):
        string = "DataFlowNode<"
        if isinstance(self.psyir, DataSymbol):
            string += str(self.psyir)
        else:
            string += (
                f"{type(self.psyir).__name__}: '{self.psyir.debug_string()}'"
            )
        string += f", access_type: {self.access_type.name}"
        string += f", {len(self.backward_dependences)} backward dependences"
        string += f", {len(self.forward_dependences)} forward dependences>"
        return string

    def __repr__(self):
        return str(self)


class DataFlowDAG:  # (dict)
    def __init__(self):
        self._schedule = None
        self._dag_nodes = []

    # Schedule or None
    @property
    def schedule(self):
        return self._schedule

    @property
    def dag_nodes(self):
        return self._dag_nodes

    @property
    def in_arguments_nodes(self):
        in_arguments_nodes = []
        for dag_node in self.dag_nodes:
            if (
                isinstance(dag_node.psyir, DataSymbol)
                and dag_node.access_type is AccessType.WRITE
            ):
                in_arguments_nodes.append(dag_node)
        return in_arguments_nodes

    @property
    def out_arguments_nodes(self):
        out_arguments_nodes = []
        for dag_node in self.dag_nodes:
            if (
                isinstance(dag_node.psyir, DataSymbol)
                and dag_node.access_type is AccessType.READ
            ):
                out_arguments_nodes.append(dag_node)
        return out_arguments_nodes

    @property
    def forward_leaves(self):
        forward_leaves = []
        for dag_node in self.dag_nodes:
            if (
                len(dag_node.forward_dependences) == 0
                and dag_node not in forward_leaves
                # and dag_node not in self.read_arguments_nodes
            ):
                forward_leaves.append(dag_node)
        return forward_leaves

    @property
    def backward_leaves(self):
        backward_leaves = []
        for dag_node in self.dag_nodes:
            if (
                len(dag_node.backward_dependences) == 0
                and dag_node not in backward_leaves
                # and dag_node not in self.out_arguments_nodes
            ):
                backward_leaves.append(dag_node)
        return backward_leaves

    @classmethod
    def create_from_schedule(cls, schedule):
        if not isinstance(schedule, Schedule):
            raise TypeError("")

        dag = cls()
        dag._schedule = schedule

        if isinstance(schedule, Routine):
            for argument in schedule.symbol_table.argument_list:
                # intent(in) => write-like node at routine start
                if argument.interface.access is ArgumentInterface.Access.READ:
                    DataFlowNode.create(dag, argument, AccessType.WRITE)
                # intent(out) => read-like node at routine end
                elif (
                    argument.interface.access is ArgumentInterface.Access.WRITE
                ):
                    DataFlowNode.create(dag, argument, AccessType.READ)
                # intent(inout) or unknown => write-like node at routine start
                #                             and read-like node at routine end
                else:
                    DataFlowNode.create(dag, argument, AccessType.WRITE)
                    DataFlowNode.create(dag, argument, AccessType.READ)

        # Transform all statements found in the schedule, recursing if need be
        dag._statement_list_to_dag_nodes(schedule.children)

        # Link the (in)out argument output node to the previous writes
        for out_arg_node in dag.out_arguments_nodes:
            out_arg_node.add_backward_dependence_to_last_write()

        return dag

    def get_all_last_writes_to_array_symbol(self, array_symbol):
        if not isinstance(array_symbol, DataSymbol):
            raise TypeError("")
        if not isinstance(array_symbol.datatype, ArrayType):
            raise TypeError("")

        all_writes_to_symbol = self.all_writes_to(array_symbol)
        all_last_writes_to_symbol = all_writes_to_symbol.copy()
        # For every write, check if *it's* previous write is in the list
        # and remove it if so
        for write in all_writes_to_symbol:
            previous_write = self.last_write_to(write.psyir)
            # Do it conservatively with respect to indices, ie. only
            # remove if the indices are the same
            if (
                isinstance(write, ArrayReference)
                and isinstance(previous_write, ArrayReference)
                and write.indices == previous_write.indices
            ):
                if previous_write in all_last_writes_to_symbol:
                    all_last_writes_to_symbol.remove(previous_write)

        return all_last_writes_to_symbol

    def get_dag_node_for(self, psyir, access_type):
        if not isinstance(psyir, (DataNode, DataSymbol)):
            raise TypeError("")
        if not isinstance(access_type, AccessType):
            raise TypeError("")

        for dag_node in self.dag_nodes:
            if dag_node.psyir is psyir and dag_node.access_type is access_type:
                return dag_node

        return None

    def to_psyir_list(self):
        psyir_list = []
        for leaf in self.backward_leaves:
            leaf_psyir_list = leaf.to_psyir_list_forward()
            for psyir in leaf_psyir_list:
                if psyir not in psyir_list:
                    psyir_list.append(psyir)
        return psyir_list

    def dataflow_tree_from(self, psyir):
        from_node = None
        nodes = self.backward_leaves
        new_nodes = []
        while from_node is None:
            for node in nodes:
                if node.psyir == psyir:
                    from_node = node
                else:
                    new_nodes.extend(node.forward_dependences)

            if from_node is None and len(new_nodes) == 0:
                raise ValueError("Not found")

            nodes = new_nodes

        copy_from_node = from_node.copy_forward()
        tree_from_node = copy_from_node.dag

        tree_backward_leaves = tree_from_node.backward_leaves
        if (
            len(tree_backward_leaves) != 1
            or tree_backward_leaves[0] is not copy_from_node
        ):
            raise ValueError("")

        return tree_from_node

    def dataflow_tree_to(self, psyir):
        to_node = None
        nodes = self.forward_leaves
        new_nodes = []
        while to_node is None:
            for node in nodes:
                if node.psyir == psyir:
                    to_node = node
                else:
                    new_nodes.extend(node.backward_dependences)

            if to_node is None and len(new_nodes) == 0:
                raise ValueError("Not found")

            nodes = new_nodes

        copy_to_node = to_node.copy_backward()
        tree_to_node = copy_to_node.dag

        tree_forward_leaves = tree_to_node.forward_leaves
        if (
            len(tree_forward_leaves) != 1
            or tree_forward_leaves[0] is not copy_to_node
        ):
            raise ValueError("")

        return tree_to_node

    @property
    def all_writes(self):
        return [
            dag_node
            for dag_node in self.dag_nodes
            if dag_node.access_type is AccessType.WRITE
        ]

    def all_writes_to(self, psyir):  # , with_same_indices = None):
        if not isinstance(psyir, (Reference, DataSymbol)):
            raise TypeError()

        all_writes_to = []
        if isinstance(psyir, DataSymbol):
            for write in self.all_writes:
                if isinstance(write.psyir, DataSymbol):
                    if write.psyir == psyir:
                        all_writes_to.append(write)
                elif isinstance(write.psyir, Reference):
                    if write.psyir.symbol == psyir:
                        all_writes_to.append(write)
        else:
            for write in self.all_writes:
                if isinstance(write.psyir, DataSymbol):
                    if write.psyir == psyir.symbol:
                        all_writes_to.append(write)
                elif isinstance(write.psyir, Reference):
                    if write.psyir.symbol == psyir.symbol:
                        all_writes_to.append(write)

        return all_writes_to

    @staticmethod
    def node_position(node):
        if not isinstance(node, DataFlowNode):
            raise TypeError("")

        if isinstance(node.psyir, DataSymbol):
            if node.access_type is AccessType.WRITE:
                return -1
            elif node.access_type is AccessType.READ:
                return 1000000
            else:
                raise ValueError("")
        else:
            return node.psyir.abs_position

    def last_write_to(self, psyir):  # , with_same_indices = None):
        # Arguments are typechecked by all_writes_to

        last_write = None
        all_writes_to = self.all_writes_to(psyir)  # , with_same_indices)
        all_writes_to.sort(key=self.node_position, reverse=True)

        # TODO: extensive testing...
        # Consider all writes in parallel directives with variable scoping
        # - private or firstprivate for OpenMP
        # - copyin for ACC
        # ie. which are not written back outside of scope
        all_writes_in_private_directives = []
        associated_directives = []
        for write in all_writes_to:
            if isinstance(write.psyir, Reference):
                directive_ancestors = []
                directive_ancestor = write.psyir.ancestor(Directive)
                while directive_ancestor is not None:
                    directive_ancestors.append(directive_ancestor)
                    directive_ancestor = directive_ancestor.ancestor(Directive)
                if len(directive_ancestors) != 0:
                    for directive in directive_ancestors:
                        for clause in directive.clauses:
                            if isinstance(
                                clause,
                                (
                                    OMPPrivateClause,
                                    OMPFirstprivateClause,
                                    ACCCopyInClause,
                                ),
                            ):
                                for ref in clause.children:
                                    if ref.symbol == (
                                        psyir
                                        if isinstance(psyir, DataSymbol)
                                        else psyir.symbol
                                    ):
                                        all_writes_in_private_directives.append(
                                            write
                                        )
                                        associated_directives.append(directive)

        # If looking for the last write to a Symbol, this is an out argument,
        # all these private writes can be removed
        if isinstance(psyir, DataSymbol):
            for write in all_writes_in_private_directives:
                all_writes_to.remove(write)
        # Otherwise, remove all the writes for which there is no path from the
        # directive to the psyir, that is for which the psyir and the write
        # are not in the same directive
        else:
            for write, directive in zip(
                all_writes_in_private_directives, associated_directives
            ):
                try:
                    psyir.path_from(directive)
                except ValueError:
                    all_writes_to.remove(write)

        if len(all_writes_to) != 0:
            # For array references, we need to consider indices in some cases
            # - if not, we only look at the symbol
            if isinstance(psyir, ArrayReference):
                all_writes_to_possibly_same_index = []
                for write in all_writes_to:
                    # - in arguments write to all indices, so yes
                    if isinstance(write.psyir, DataSymbol):
                        all_writes_to_possibly_same_index.append(write)
                    elif isinstance(write.psyir, ArrayReference):
                        psyir_loop_ancestor = psyir.ancestor(Loop)
                        write_loop_ancestor = write.psyir.ancestor(Loop)
                        # - if within same loop or not in loops,
                        # we require indices equality
                        if psyir_loop_ancestor is write_loop_ancestor:
                            if psyir.indices == write.psyir.indices:
                                all_writes_to_possibly_same_index.append(write)
                        # - otherwise it might be the same indices
                        else:
                            all_writes_to_possibly_same_index.append(write)
                    # - if a reference to the whole array then yes
                    else:
                        all_writes_to_possibly_same_index.append(write)

                if len(all_writes_to_possibly_same_index) != 0:
                    last_write = all_writes_to_possibly_same_index[0]

            # For scalar references or references to the whole array
            else:
                last_write = all_writes_to[0]

        return last_write

    def _statement_list_to_dag_nodes(self, statement_list):
        if not isinstance(statement_list, list):
            raise TypeError("")
        for statement in statement_list:
            if not isinstance(statement, Statement):
                raise TypeError("")
            
        for statement in statement_list:
            if isinstance(statement, Assignment):
                # print("dealing with assignment ", statement.debug_string())
                # print("last writes are ", [w.psyir.name for w in self._last_writes])
                lhs, rhs = statement.children
                # NOTE: rhs then lhs is important for last writes ordering
                rhs_node = DataFlowNode.create_or_get(
                    self,
                    rhs,
                    (
                        AccessType.READ
                        if isinstance(rhs, Reference)
                        else AccessType.UNKNOWN
                    ),
                )
                lhs_node = DataFlowNode.create_or_get(self, lhs, AccessType.WRITE)
                rhs_node.add_forward_dependence(lhs_node)
            elif isinstance(statement, Call):
                if isinstance(statement, IntrinsicCall):
                    # TODO: which are these? ALLOCATE and so on?
                    raise NotImplementedError("")
                DataFlowNode.create_or_get(self, statement)
            elif isinstance(statement, Loop):
                # TODO
                # loop_var_node =
                for expr in (
                    statement.start_expr,
                    statement.stop_expr,
                    statement.step_expr,
                ):
                    # No point in creating nodes for literals
                    if len(expr.walk(Reference)) != 0:
                        DataFlowNode.create_or_get(
                            self,
                            expr,
                            (
                                AccessType.READ
                                if isinstance(expr, Reference)
                                else AccessType.UNKNOWN
                            ),
                        )

                self._statement_list_to_dag_nodes(statement.loop_body.children)

            elif isinstance(statement, WhileLoop):
                # No point in creating nodes for literals
                if len(statement.condition.walk(Reference)) != 0:
                    DataFlowNode.create_or_get(
                        self,
                        statement.condition,
                        (
                            AccessType.READ
                            if isinstance(statement.condition, Reference)
                            else AccessType.UNKNOWN
                        ),
                    )

                self._statement_list_to_dag_nodes(statement.loop_body.children)

            elif isinstance(statement, IfBlock):
                # No point in creating nodes for literals
                if len(statement.condition.walk(Reference)) != 0:
                    DataFlowNode.create_or_get(
                        self,
                        statement.condition,
                        (
                            AccessType.READ
                            if isinstance(statement.condition, Reference)
                            else AccessType.UNKNOWN
                        ),
                    )

                self._statement_list_to_dag_nodes(statement.if_body.children)
                if statement.else_body is not None:
                    self._statement_list_to_dag_nodes(statement.else_body.children)

            elif isinstance(statement, Directive):
                for clause in statement.clauses:
                    # TODO: OMPReductionClause once implemented
                    if isinstance(clause, (OMPPrivateClause, ACCCopyOutClause)):
                        for ref in clause.children:
                            DataFlowNode.create(self, ref, AccessType.WRITE)
                    elif isinstance(
                        clause,
                        (
                            OMPFirstprivateClause,
                            OMPSharedClause,
                            ACCCopyClause,
                            ACCCopyInClause,
                            OMPDependClause,
                        ),
                    ):
                        for ref in clause.children:
                            DataFlowNode.create(self, ref, AccessType.READ)
                if isinstance(statement, RegionDirective):
                    self._statement_list_to_dag_nodes(statement.dir_body.children)

            elif isinstance(statement, CodeBlock):
                # TODO?
                raise NotImplementedError("")
            elif isinstance(statement, PSyDataNode):
                # Treating these as external to the actual program
                pass
            elif isinstance(statement, Return):
                break
            # TODO
            else:
                raise NotImplementedError("")

    def to_dot_format(self):
        if isinstance(self.schedule, Routine):
            digraph_name = self.schedule.name
        else:
            digraph_name = "G"
        lines = [f"digraph {digraph_name}", "{"]

        # TODO: subgraphs?

        id_counter = 0
        id_to_dag_node = dict()
        for dag_node in self.dag_nodes:
            id = f"node_{id_counter}"
            id_to_dag_node[id] = dag_node
            id_counter += 1

            if dag_node.access_type is AccessType.READ:
                color = "blue"
            elif dag_node.access_type is AccessType.WRITE:
                color = "red"
            else:
                color = "black"

            if isinstance(dag_node.psyir, DataSymbol):
                label = f"{dag_node.psyir.name} ({dag_node.psyir.interface.access.name})"
                lines.append(
                    f'{id} [label="{label}", shape="invtriangle", color="{color}"]'
                )
            else:
                label = dag_node.psyir.debug_string()
                if isinstance(dag_node.psyir, Call):
                    lines.append(
                        f'{id} [label="{label}", shape="box", color="{color}"]'
                    )
                elif isinstance(dag_node.psyir, Operation):
                    lines.append(
                        f'{id} [label="{label}", shape="oval", color="{color}"]'
                    )
                elif isinstance(dag_node.psyir, (Reference, Literal)):
                    lines.append(f'{id} [label="{label}", color="{color}"]')
                else:
                    raise ValueError(type(dag_node.psyir).__name__)

        for this_node in self.dag_nodes:
            for id, node in id_to_dag_node.items():
                if this_node is node:
                    this_id = id
                    break

            for fwd_node in this_node.forward_dependences:
                for id, node in id_to_dag_node.items():
                    if fwd_node is node:
                        fwd_id = id
                        break

                lines.append(f"{this_id} -> {fwd_id}")

        lines.append("}")

        return "\n".join(lines)

    def render_graph(self, filename="graph"):
        import pydot

        dot_graph = self.to_dot_format()
        (graph,) = pydot.graph_from_dot_data(dot_graph)
        graph.write_png(f"{filename}.png")

    def __str__(self):
        sorted_dag_nodes = self.dag_nodes
        sorted_dag_nodes.sort(key=self.node_position)
        strings = [str(dag_node) for dag_node in sorted_dag_nodes]
        return "\n".join(strings)

    def __repr__(self):
        return str(self)


from psyclone.psyir.frontend.fortran import FortranReader

reader = FortranReader()

psyir = reader.psyir_from_source(
    """
subroutine foo(a, b)
    real, intent(inout) :: a
    real, intent(out), dimension(10) :: b
    real :: c, d, e, f
    integer :: i, j

    b = 3.0
    d = 4.0

    do i = 1, 9
        b(i) = a**i
        c = b(i)
        b(i + 1) = d
    end do

    b(3) = 3.0
end subroutine foo


subroutine bar(x, y)
    real, intent(inout) :: x
    real, intent(inout) :: y

    x = x + 1.0
    y = exp(x**2)
end subroutine bar
"""
)

routine = psyir.children[0]
dag = DataFlowDAG.create_from_schedule(routine)

dag_fwd_leaves = dag.forward_leaves
dag_bwd_leaves = dag.backward_leaves

# print("DAG nodes:")
# for node in dag.dag_nodes:
#     print(node.psyir)

# print("==========\nForward leaves:")
# for leaf in dag_fwd_leaves:
#     print(leaf.psyir)

# print("==========\nBackward leaves:")
# for leaf in dag_bwd_leaves:
#     print(leaf.psyir)

a, b = routine.symbol_table.argument_list
flow_from_a = dag.dataflow_tree_from(a)
flow_to_b = dag.dataflow_tree_to(b)
list_from_a = flow_from_a.to_psyir_list()
list_to_b = flow_to_b.to_psyir_list()

bwd_leaves_from_a = flow_from_a.backward_leaves
print("==========\nBackward leaves:")
for leaf in bwd_leaves_from_a:
    print(leaf.psyir)

fwd_leaves_to_b = flow_to_b.forward_leaves
print("==========\nForward leaves:")
for leaf in fwd_leaves_to_b:
    print(leaf.psyir)

# print("===========\nFwd and bwd deps for b_arg")
# b_out_arg_node = DataFlowNode.create_or_get(dag, b)
# print(b_out_arg_node.forward_dependences, b_out_arg_node.backward_dependences)


# print("=====\n from a:")
# for psyir in list_from_a:
#     print(psyir)
#     print("------------")
# print("=====\n to b:")
# for psyir in list_to_b:
#     print(psyir)
#     print("------------")
# print(flow_to_b.to_psyir_list())

dag.render_graph("dag")
flow_from_a.render_graph("from_a")
flow_to_b.render_graph("to_b")

print(dag)
