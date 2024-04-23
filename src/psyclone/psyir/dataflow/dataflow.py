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
    def __init__(self, dag, psyir):  # , forward_dependence):
        if not isinstance(dag, DataFlowDAG):
            raise TypeError("")
        if not isinstance(psyir, (DataNode, DataSymbol, RoutineSymbol)):
            raise TypeError("")
        if isinstance(psyir, DataSymbol) and not isinstance(
            psyir.interface, ArgumentInterface
        ):
            raise TypeError("")

        self._psyir = psyir
        self._dag = dag
        self._forward_dependences = []
        self._backward_dependences = []

        self.dag.dag_nodes.append(self)

        # if forward_dependence is not None:
        #     self.add_forward_dependence(forward_dependence)

    def add_backward_dependence_to_last_write(self):
        if isinstance(self.psyir, Reference):
            last_write_to_ref = self.dag.last_write_to(self.psyir)

            # if self.psyir.name == "c" and isinstance(self.psyir.parent, Call):
            #     print("adding last write to c, got ", last_write_to_ref.psyir)

            if last_write_to_ref is not None and last_write_to_ref is not self:
                self.add_backward_dependence(last_write_to_ref)

    def recurse_on_children(self):
        psyir = self.psyir
        if isinstance(psyir, (DataSymbol, RoutineSymbol)):
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
                arg_node = DataFlowNode.create_or_get(self.dag, arg)
                arg_node.add_forward_dependence(self)
                # child_nodes.append(child_node)

        elif isinstance(psyir, Call):
            # TODO: deduplicate using edited DAG._call_to_dag_nodes

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
                        in_arg_node = DataFlowNode.create_or_get(self.dag, arg)
                        in_arg_node.add_forward_dependence(self)

                        # if draw_called_routines_dag:
                        #     in_arg_node.add_forward_dependence(called_args_nodes[i])

                    if intent is not ArgumentInterface.Access.READ:
                        # NOTE: create to allow for duplicate DAG nodes
                        # with same PSyIR
                        out_arg_node = DataFlowNode.create(
                            self.dag, arg, is_out_node_from_call=True
                        )
                        out_arg_node.add_backward_dependence(self)

                        # if draw_called_routines_dag:
                        #     out_arg_node.add_backward_dependence(called_args_nodes[i])

                        # self.dag._update_last_write(out_arg_node)

            # We don't know the intents of the arguments
            # so treat everything as inout
            else:
                in_arg_nodes = [
                    DataFlowNode.create_or_get(self.dag, arg)
                    for arg in psyir.children
                ]
                # NOTE: create to allow for duplicate DAG nodes
                # with same PSyIR
                out_arg_nodes = [
                    DataFlowNode.create(
                        self.dag, arg, is_out_node_from_call=True
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
                operand_node = DataFlowNode.create_or_get(self.dag, operand)
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
    def create(cls, dag, psyir, is_out_node_from_call=False):
        # Other arguments are typechecked by the constructor
        if not isinstance(is_out_node_from_call, bool):
            raise TypeError("")

        dag_node = cls(dag, psyir)
        if isinstance(psyir, Reference):
            # TODO: what about intrinsics that modify their arguments?
            if (
                isinstance(psyir.parent, Assignment)
                and (psyir is psyir.parent.lhs)
            ) or is_out_node_from_call:
                dag._update_last_write(dag_node)

        dag_node.add_backward_dependence_to_last_write()
        dag_node.recurse_on_children()
        # dag.dag_nodes.append(dag_node)
        if isinstance(psyir, DataSymbol) and isinstance(
            psyir.interface, ArgumentInterface
        ):
            if psyir.interface.access is not ArgumentInterface.Access.WRITE:
                dag.read_arguments_nodes.append(dag_node)
            if psyir.interface.access is not ArgumentInterface.Access.READ:
                dag.written_arguments_nodes.append(dag_node)
        return dag_node

    @classmethod
    def create_or_get(cls, dag, psyir):
        if not isinstance(dag, DataFlowDAG):
            raise TypeError("")
        if not isinstance(psyir, (DataNode, DataSymbol, RoutineSymbol)):
            raise TypeError("")

        existing_dag_node = dag.get_dag_node_for(psyir)
        if existing_dag_node is not None:
            return existing_dag_node
        else:
            return cls.create(dag, psyir)

    def copy_single_node_to(self, new_dag):
        if not isinstance(new_dag, DataFlowDAG):
            raise TypeError("")

        dag_node = DataFlowNode(new_dag, self.psyir)
        if isinstance(self.psyir, DataSymbol) and isinstance(
            self.psyir.interface, ArgumentInterface
        ):
            if (
                self.psyir.interface.access
                is not ArgumentInterface.Access.WRITE
            ):
                new_dag.read_arguments_nodes.append(dag_node)
            if self.psyir.interface.access is not ArgumentInterface.Access.READ:
                new_dag.written_arguments_nodes.append(dag_node)
        return dag_node

    def copy_or_get_single_node_to(self, new_dag):
        if not isinstance(new_dag, DataFlowDAG):
            raise TypeError("")

        # FIXME: this won't work for duplicate call args, etc
        existing_dag_node = new_dag.get_dag_node_for(self.psyir)
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

    # # TODO: test this fully...
    # @property
    # def is_out_reference_from_call(self):
    #     return (
    #         isinstance(self.psyir, DataNode)  # symbols have no parent
    #         and isinstance(self.psyir.parent, Call)  # should be an arg
    #         and len(self.backward_dependences) == 1  # only the call
    #         and self.backward_dependences[0].psyir is self.psyir.parent
    #     )

    # TODO: test this fully...
    @property
    def is_call_argument_reference(self):
        return (
            isinstance(self.psyir, Reference)  # symbols have no parent
            and isinstance(self.psyir.parent, Call)  # should be an arg
            # and len(self.backward_dependences) == 1  # only the call
            # and self.backward_dependences[0].psyir is self.psyir.parent
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

    # def get_backward_leaves(self):
    #     if len(self.backward_dependences) == 0:
    #         return {self}

    #     leaves = set()
    #     for bwd in self.backward_dependences:
    #         leaves.update(bwd.get_backward_leaves())

    #     return leaves

    # def to_psyir_list(self, psyir_type = None):
    #     if not isinstance(psyir_type, (Node, Symbol, NoneType)):
    #         raise TypeError("")
    #     psyir_list = [self.psyir]
    #     for dep in self.backward_dependences + self.forward_dependences:
    #         dep_list = dep._to_psyir_list_coming_from(self)
    #         for psyir in dep_list:
    #             if psyir not in psyir_list:
    #                 psyir_list.append(psyir)
    #     #     psyir_list.update(bwd._to_psyir_set_coming_from(self))
    #     # for fwd in self.forward_dependences:
    #     #     psyir_list.update(fwd._to_psyir_set_coming_from(self))

    #     if psyir_type is None:
    #         return psyir_list
    #     else:
    #         return [psyir for psyir in psyir_list if isinstance(psyir, psyir_type)]

    # def _to_psyir_list_coming_from(self, coming_from):
    #     if not isinstance(coming_from, DataFlowNode):
    #         raise TypeError("")

    #     psyir_list = [self.psyir]
    #     for dep in self.backward_dependences + self.forward_dependences:
    #         print(coming_from.psyir, dep.psyir)
    #         if dep is not coming_from:
    #             dep_list = dep._to_psyir_list_coming_from(self)
    #             for psyir in dep_list:
    #                 if psyir not in psyir_list:
    #                     psyir_list.append(psyir)

    #     return psyir_list


class DataFlowDAG:  # (dict)
    def __init__(self):
        self._schedule = None
        self._dag_nodes = []
        self._last_writes = []  # Internal
        self._read_arguments_nodes = []
        self._written_arguments_nodes = []
        # self._forward_leaves = []
        # self._backward_leaves = []

    # Schedule or None
    @property
    def schedule(self):
        return self._schedule

    @property
    def dag_nodes(self):
        return self._dag_nodes

    @property
    def read_arguments_nodes(self):
        return self._read_arguments_nodes

    @property
    def written_arguments_nodes(self):
        return self._written_arguments_nodes

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
        # return self._forward_leaves

    @property
    def backward_leaves(self):
        backward_leaves = []
        for dag_node in self.dag_nodes:
            if (
                len(dag_node.backward_dependences) == 0
                and dag_node not in backward_leaves
                and dag_node not in self.written_arguments_nodes
            ):
                backward_leaves.append(dag_node)
        return backward_leaves
        # return self._backward_leaves

    @classmethod
    def create_from_schedule(cls, schedule):
        if not isinstance(schedule, Schedule):
            raise TypeError("")
        
        dag = cls()

        dag._schedule = schedule
        
        if isinstance(schedule, Routine):
            for argument in schedule.symbol_table.argument_list:
                argument_node = DataFlowNode.create_or_get(dag, argument)

        for statement in schedule.children:
            dag._statement_to_dag_nodes(statement)

        for i, last_write in enumerate(dag._last_writes):
            for out_arg_node in dag._written_arguments_nodes:
                if last_write.psyir.symbol == out_arg_node.psyir:
                    out_arg_node.add_backward_dependence(last_write)
                    dag._last_writes[i] = out_arg_node

        return dag

    def get_dag_node_for(self, psyir):
        for dag_node in self.dag_nodes:
            if dag_node.psyir is psyir:
                return dag_node

        return None

    def contains_node(self, node):
        if self.get_dag_node_for(node.psyir) is not None:
            return True

        return False

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
        if len(tree_backward_leaves) != 1 or tree_backward_leaves[0] is not copy_from_node:
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
        if len(tree_forward_leaves) != 1 or tree_forward_leaves[0] is not copy_to_node:
            raise ValueError("")

        return tree_to_node

    def last_write_to(self, reference):
        if not isinstance(reference, Reference):
            raise TypeError("")

        # Scalar, no question about indices
        if isinstance(reference.datatype, ScalarType):
            for last_write in self._last_writes:
                if last_write.psyir == reference:
                    return last_write

        # Array, consider indices iff in the same loop
        elif isinstance(reference.datatype, ArrayType):
            # First consider last writes
            for last_write in self._last_writes:
                # This checks if both are within the same loop or not in loops
                if last_write.ancestor(Loop) is reference.ancestor(Loop):
                    if last_write.psyir == reference:
                        return last_write
                # If in different loops (or one in, the other not),
                # return the last time any array element was written
                else:
                    if last_write.psyir.symbol == reference.symbol:
                        return last_write
        else:
            raise NotImplementedError("")

        # If not written to yet, it may still be an in argument of a routine
        for in_arg_node in self._read_arguments_nodes:
            if in_arg_node.psyir == reference.symbol:
                return in_arg_node

        return None

    def _update_last_write(self, new_node):
        if not isinstance(new_node, (DataFlowNode)):
            raise TypeError("")
        if not isinstance(new_node.psyir, Reference):
            raise TypeError("")

        # print("updating last write to ", new_node.psyir.name)

        last_write = self.last_write_to(new_node.psyir)

        # or part for read args
        if last_write is None or last_write not in self._last_writes:
            self._last_writes.append(new_node)
            # print("appended")
        else:
            index = self._last_writes.index(last_write)
            self._last_writes[index] = new_node
            # print("updated")

    def _statement_to_dag_nodes(self, statement):
        if not isinstance(statement, Statement):
            raise TypeError("")
        if isinstance(statement, Assignment):
            # print("dealing with assignment ", statement.debug_string())
            # print("last writes are ", [w.psyir.name for w in self._last_writes])
            lhs, rhs = statement.children
            # NOTE: rhs then lhs is important for last writes ordering
            rhs_node = DataFlowNode.create_or_get(self, rhs)
            lhs_node = DataFlowNode.create_or_get(self, lhs)
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
                DataFlowNode.create_or_get(self, expr)
            for stmt in statement.loop_body.children:
                self._statement_to_dag_nodes(stmt)
        elif isinstance(statement, WhileLoop):
            DataFlowNode.create_or_get(self, statement.condition)
            for stmt in statement.loop_body.children:
                self._statement_to_dag_nodes(stmt)
        elif isinstance(statement, IfBlock):
            DataFlowNode.create_or_get(self, statement.condition)
            for stmt in statement.if_body.children:
                self._statement_to_dag_nodes(stmt)
            if statement.else_body is not None:
                for stmt in statement.else_body.children:
                    self._statement_to_dag_nodes(stmt)
        elif isinstance(statement, Directive):
            # TODO: deal with clauses...
            raise NotImplementedError("")
        elif isinstance(statement, CodeBlock):
            # TODO?
            raise NotImplementedError("")
        elif isinstance(statement, PSyDataNode):
            # Treating these as external to the actual program
            pass
        elif isinstance(statement, Return):
            # FIXME, is this right?
            pass
        # TODO
        else:
            raise NotImplementedError("")

    def to_dot_format(self):
        if isinstance(self.schedule, Routine):
            digraph_name = self.schedule.name
        else:
            digraph_name = "G"
        lines = [f"digraph {digraph_name}", "{"]

        id_counter = 0
        id_to_dag_node = dict()
        for dag_node in self.dag_nodes:
            id = f"node_{id_counter}"
            id_to_dag_node[id] = dag_node
            id_counter += 1

            if isinstance(dag_node.psyir, DataSymbol):
                label = f"{dag_node.psyir.name} ({dag_node.psyir.interface.access.name})"
                lines.append(f'{id} [label="{label}", shape="invtriangle"]')
            else:
                label = dag_node.psyir.debug_string()
                if isinstance(dag_node.psyir, Call):
                    lines.append(f'{id} [label="{label}", shape="box"]')
                elif isinstance(dag_node.psyir, Operation):
                    lines.append(f'{id} [label="{label}", shape="oval"]')
                elif isinstance(dag_node.psyir, (Reference, Literal)):
                    lines.append(f'{id} [label="{label}"]')
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


from psyclone.psyir.frontend.fortran import FortranReader

reader = FortranReader()

psyir = reader.psyir_from_source(
    """
subroutine foo(a, b)
    real, intent(in) :: a
    real, intent(out) :: b
    real :: c, d, e, f
    integer :: i, j

    c = a + 1.0
    e = a**2
    f = cos(e)
    d = c + 2.0
    c = d * a
    b = c + d

    ! j = 3
    ! do i = 1, j, 1
    !     b = b + 1.0
    ! end do

    if (b .ge. 4.0) then
        b = b - 1.0
    else
        b = b + 1.0
    end if

    call bar(c, b)

    b = b + c
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
