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
    OMPParallelDirective
)
from psyclone.psyir.symbols import (
    DataSymbol,
    ArgumentInterface,
    ArrayType,
)

# TODO: a proper dataflow would take into account if/else branches.
# TODO: we could 'inline' the called (and found) routines in the DAG.

class DataFlowNode:
    """
    Represents a node in the data flow graph.

    :param dag: The data flow graph that the node belongs to.
    :type dag: :py:class:`psyclone.psyir.dataflow.DataFlowDAG`
    :param psyir: The PSyIR node associated with the data flow node.
    :type psyir: :py:class:`psyclone.psyir.nodes.DataNode` or \
                 :py:class:`psyclone.psyir.symbols.DataSymbol`
    :param access_type: The access type of the data flow node. \
                        Defaults to `AccessType.UNKNOWN`.
    :type access_type: :py:class:`psyclone.psyir.dataflow.AccessType`

    :raises TypeError: If `dag` is not an instance of \
                       :py:class:`psyclone.psyir.dataflow.DataFlowDAG`, \
                       or if `psyir` is not an instance of \
                       :py:class:`psyclone.psyir.nodes.DataNode` or \
                       :py:class:`psyclone.psyir.symbols.DataSymbol`, \
                       or if `psyir` is a \
                       :py:class:`psyclone.psyir.symbols.DataSymbol` \
                       and its interface is not an instance of \
                       :py:class:`psyclone.psyir.symbols.ArgumentInterface`, \
                       or if `access_type` is not an instance of \
                       :py:class:`psyclone.psyir.dataflow.AccessType`, \
                       or if `psyir` is not an instance of \
                       :py:class:`psyclone.psyir.nodes.Reference` or \
                       :py:class:`psyclone.psyir.symbols.DataSymbol` and \
                       `access_type` is not `AccessType.UNKNOWN`, \
                       or if `psyir` is an instance of \
                       :py:class:`psyclone.psyir.nodes.Reference` or \
                       :py:class:`psyclone.psyir.symbols.DataSymbol` and \
                       `access_type` is not `AccessType.READ` or \
                       `AccessType.WRITE`.
    """

    def __init__(self, dag, psyir, access_type=AccessType.UNKNOWN):
        if not isinstance(dag, DataFlowDAG):
            raise TypeError(
                f"'dag' argument must be of type 'DataFlowDAG' "
                f"but found '{type(dag).__name__}'."
            )
        if not isinstance(psyir, (DataNode, DataSymbol)):
            raise TypeError(
                f"'psyir' argument must be of type 'DataNode' or "
                f"'DataSymbol' but found '{type(psyir).__name__}'."
            )
        if isinstance(psyir, DataSymbol) and not isinstance(
            psyir.interface, ArgumentInterface
        ):
            raise TypeError(
                f"'psyir' argument must have 'ArgumentInterface' "
                f"interface if it is a 'DataSymbol' but found "
                f"'{type(psyir.interface).__name__}'."
            )
        if not isinstance(access_type, AccessType):
            raise TypeError(
                f"'access_type' argument must be of type "
                f"'AccessType' but found "
                f"'{type(access_type).__name__}'."
            )
        if (
            not isinstance(psyir, (Reference, DataSymbol))
            and access_type is not AccessType.UNKNOWN
        ):
            raise ValueError(
                "'access_type' argument must be "
                "'AccessType.UNKNOWN' if 'psyir' argument is not "
                "of type 'Reference' or 'DataSymbol'."
            )
        if isinstance(psyir, (Reference, DataSymbol)) and access_type not in (
            AccessType.READ,
            AccessType.WRITE,
        ):
            raise ValueError(
                "'access_type' argument must be 'AccessType.READ' "
                "or 'AccessType.WRITE' if 'psyir' argument is of "
                "type 'Reference' or 'DataSymbol'."
            )

        self._psyir = psyir
        self._dag = dag
        self._forward_dependences = []
        self._backward_dependences = []
        self._access_type = access_type

        # Add the new node to the DAG
        self.dag.dag_nodes.append(self)

    def add_backward_dependence_to_last_write(self):
        """
        Adds backward dependences to the last write of the symbol referenced \
        by `self.psyir`. If `self.psyir` is a reference to a whole array, \
        backward dependences are added to all last writes of its elements.
        If `self.psyir` is a reference to a scalar or a single element of \
        an array, a backward dependence is added to the last write of the \
        referenced symbol.
        """
        if (
            isinstance(self.psyir, (Reference, DataSymbol))
            and self.access_type is AccessType.READ
        ):
            symbol = (
                self.psyir
                if isinstance(self.psyir, DataSymbol)
                else self.psyir.symbol
            )
            # Reference to a whole array, should have backward dependences to
            # all last writes to its elements
            if self.psyir.is_array and not isinstance(
                self.psyir, ArrayReference
            ):
                last_writes_to_ref = (
                    self.dag.get_all_last_writes_to_array_symbol(symbol)
                )
                for write in last_writes_to_ref:
                    if write is not self:
                        self.add_backward_dependence(write)
            else:
                last_write_to_ref = self.dag.last_write_before(self)

                if (
                    last_write_to_ref is not None
                    and last_write_to_ref is not self
                ):
                    self.add_backward_dependence(last_write_to_ref)

    def recurse_on_children(self):
        """
        Recursively processes the children of the current PSyIR node and \
        creates or gets data flow nodes for each child.

        :raises NotImplementedError: If the type of the child node is not \
                                     supported.

        """
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
        """
        Retrieves the called routine with the given name from the root of \
        the PSyIR tree.

        :param name: the name of the routine to retrieve.
        :type name: str

        :return: the called routine with the given name, \
                 or None if not found.
        :rtype: Union[:py:class:`Routine`, NoneType]
        """
        if not isinstance(name, str):
            raise TypeError(
                f"'name' argument must be of type 'str' "
                f"but found '{type(name).__name__}'."
            )

        # TODO: this looks from the root for a routine, not through imports,
        # enclosing containers, etc
        all_routines = self.psyir.root.walk(Routine)
        called_routine = None
        for routine in all_routines:
            if routine.name == name:
                called_routine = routine
                break

        return called_routine

    def get_intent_from_called_routine(self, called_routine):
        """
        Get the intent of a variable in the called routine based on its \
        corresponding argument in the calling routine.

        :param called_routine: the called routine.
        :type called_routine: :py:class:`Routine`

        :return: the intent of the variable in the called routine.
        :rtype: :py:class:`ArgumentInterface.Access`
        """
        if not isinstance(self.psyir.parent, Call):
            raise TypeError(
                "'self.psyir.parent' must be of type 'Call' "
                f"but found '{type(self.psyir.parent).__name__}'."
            )
        if not isinstance(called_routine, Routine):
            raise TypeError(
                f"'called_routine' argument must be of type 'Routine' but "
                f"found '{type(called_routine).__name__}'."
            )
        call = self.psyir.parent
        if call.routine.name != called_routine.name:
            raise ValueError(
                f"The name of the called routine '{call.routine.name}' does "
                f"not match the name of the given routine '{called_routine.name}'."
            )

        arg_index = self.psyir.parent.children.index(self.psyir)
        routine_arg = called_routine.symbol_table.argument_list[arg_index]

        return routine_arg.interface.access

    def get_call_argument_intent(self):
        """
        Returns the intent of the arguments passed to the parent Call node.

        :return: The intent of the arguments passed to the parent Call node.
        :rtype: ArgumentInterface.Access
        :raises TypeError: If the parent of the current node is not a Call node.
        """

        if not isinstance(self.psyir.parent, Call):
            raise TypeError(
                f"'self.psyir.parent' must be of type 'Call' but found "
                f"'{type(self.psyir.parent).__name__}'."
            )
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
        """
        Create a DataFlowNode object.

        :param dag: The DataFlowDAG in which to create this node.
        :type dag: :py:class:`DataFlowDAG`
        :param psyir: The PSyIR node associated with this DataFlowNode.
        :type psyir: :py:class:`psyclone.psyir.nodes.Node`
        :param access_type: The access type of this DataFlowNode.
        :type access_type: :py:class:`psyclone.core.access_type.AccessType`

        :returns: The created DataFlowNode object.
        :rtype: :py:class:`DataFlowNode`
        """

        # Arguments are typechecked by the constructor

        dag_node = cls(dag, psyir, access_type)
        dag_node.add_backward_dependence_to_last_write()
        dag_node.recurse_on_children()

        return dag_node

    @classmethod
    def create_or_get(cls, dag, psyir, access_type=AccessType.UNKNOWN):
        """Create a DataFlowNode object or get it if it already exists in the
        DAG.

        :param dag: The DataFlowDAG in which to create this node.
        :type dag: :py:class:`DataFlowDAG`
        :param psyir: The PSyIR node or symbol associated with this 
                      DataFlowNode.
        :type psyir: :py:class:`DataNode` or :py:class:`DataSymbol`
        :param access_type: The access type of the DataFlowNode. 
                            Defaults to AccessType.UNKNOWN.
        :type access_type: :py:class:`AccessType`

        :returns: The created or existing DataFlowNode object.
        :rtype: :py:class:`DataFlowNode`

        :raises TypeError: If dag is not an instance of DataFlowDAG,  \
                           or psyir is not an instance of DataNode or \
                           DataSymbol, \
                           or access_type is not an instance of AccessType.
        :raises ValueError: If psyir is an instance of Reference or DataSymbol \
                            and access_type is not AccessType.READ \
                            or AccessType.WRITE.
        """
        if not isinstance(dag, DataFlowDAG):
            raise TypeError(
                f"'dag' argument must be of type 'DataFlowDAG' but found "
                f"'{type(dag).__name__}'."
            )
        if not isinstance(psyir, (DataNode, DataSymbol)):
            raise TypeError(
                f"'psyir' argument must be of type 'DataNode' or 'DataSymbol' "
                f"but found '{type(psyir).__name__}'."
            )
        if not isinstance(access_type, AccessType):
            raise TypeError(
                f"'access_type' argument must be of type 'AccessType' but "
                f"found '{type(access_type).__name__}'."
            )
        if (
            not isinstance(psyir, (Reference, DataSymbol))
            and access_type is not AccessType.UNKNOWN
        ):
            raise ValueError(
                f"'access_type' argument must be 'AccessType.UNKNOWN' if "
                f"'psyir' argument is not of type 'Reference' or 'DataSymbol' "
                f"but found '{access_type.name}'."
            )
        if isinstance(psyir, (Reference, DataSymbol)) and access_type not in (
            AccessType.READ,
            AccessType.WRITE,
        ):
            raise ValueError(
                f"'access_type' argument must be 'AccessType.READ' or "
                f"'AccessType.WRITE' if 'psyir' argument is of type "
                f"'Reference' or 'DataSymbol' but found '{access_type.name}'."
            )

        existing_dag_node = dag.get_dag_node_for(psyir, access_type)
        if existing_dag_node is not None:
            return existing_dag_node
        else:
            return cls.create(dag, psyir, access_type)

    def copy_single_node_to(self, new_dag):
        """Copy this node to new_dag.

        :param dag: The DataFlowDAG in which to create this node.
        :type dag: :py:class:`DataFlowDAG`

        """
        if not isinstance(new_dag, DataFlowDAG):
            raise TypeError("new_dag must be an instance of DataFlowDAG")

        return DataFlowNode(new_dag, self.psyir, self.access_type)

    def copy_or_get_single_node_to(self, new_dag):
        """Copy this node to new_dag or get it if it already exists in it.

        :param new_dag: The DataFlowDAG in which to create this node.
        :type new_dag: :py:class:`DataFlowDAG`

        :raises TypeError: If new_dag is not an instance of DataFlowDAG.

        :returns: The copied or existing DataFlowNode object.
        :rtype: :py:class:`DataFlowNode`"""
        if not isinstance(new_dag, DataFlowDAG):
            raise TypeError(
                f"'new_dag' argument must be of type 'DataFlowDAG' but found "
                f"'{type(new_dag).__name__}'."
            )

        existing_dag_node = new_dag.get_dag_node_for(
            self.psyir, self.access_type
        )
        if existing_dag_node is not None:
            return existing_dag_node
        else:
            return self.copy_single_node_to(new_dag)

    def add_forward_dependence(self, forward_dependence):
        """
        Adds a forward dependence between the current DataFlowNode and the \
        given DataFlowNode.

        :param forward_dependence: The DataFlowNode to add as a forward \
                                   dependence.
        :type forward_dependence: :py:class:`DataFlowNode`

        :raises TypeError: If the forward_dependence is not an instance of \
                           DataFlowNode.
        :raises ValueError: If the forward_dependence is the same as the \
                            current DataFlowNode.
        """

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
        """Adds a backward dependence between the current DataFlowNode and \
        the given DataFlowNode.

        :param backward_dependence: The DataFlowNode to add as a backward \
                                    dependence.
        :type backward_dependence: :py:class:`DataFlowNode`

        :raises TypeError: If the backward_dependence is not an instance of \
                           DataFlowNode.
        :raises ValueError: If the backward_dependence is the same as the \
                            current DataFlowNode.
        """
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
        """The DataFlowDAG that the node belongs to.

        :returns: The DataFlowDAG that the node belongs to.
        :rtype: :py:class:`DataFlowDAG`"""
        return self._dag

    @property
    def psyir(self):
        """The PSyIR node wrapped by the data flow node.

        :returns: The PSyIR node wrapped by the data flow node.
        :rtype: Union[:py:class:`DataNode`, :py:class:`DataSymbol`]
        """
        return self._psyir

    @property
    def forward_dependences(self):
        """The forward dependences of the data flow node, as a list.

        :returns: The forward dependences of the data flow node.
        :rtype: List[:py:class:`DataFlowNode`]
        """
        return self._forward_dependences

    @property
    def backward_dependences(self):
        """The backward dependences of the data flow node, as a list.

        :returns: The backward dependences of the data flow node.
        :rtype: List[:py:class:`DataFlowNode`]
        """
        return self._backward_dependences

    @property
    def access_type(self):
        """The access type of the data flow node.

        :returns: The access type of the data flow node.
        :rtype: :py:class:`psyclone.core.access_type.AccessType`"""
        return self._access_type

    @property
    def is_call_argument_reference(self):
        """If the PSyIR node wrapped by the data flow node is a reference to a \
        call argument.

        :returns: True if the PSyIR node wrapped by the data flow node is a \
                  reference to a call argument, False otherwise.
        :rtype: bool
        """
        return isinstance(
            self.psyir, Reference
        ) and isinstance(  # symbols have no parent
            self.psyir.parent, Call
        )  # should be an arg

    def copy_forward(self, dag_copy=None, originals=None, copies=None):
        """Copy the current node to a new DAG, recursing along the forward \
        dependences.

        :param dag_copy: The new DAG to copy the node to. 
                         If None, a new DAG is created. Defaults to None.
        :type dag_copy: Union[:py:class:`DataFlowDAG`, NoneType]
        :param originals: The list of original nodes. Defaults to an empty list.
        :type originals: List[:py:class:`DataFlowNode`]
        :param copies: The list of copied nodes. Defaults to None.
        :type copies: Union[List[:py:class:`DataFlowNode`], NoneType]

        :raises TypeError: If dag_copy is not an instance of DataFlowDAG, \
                           or originals is not a list or None, \
                           or copies is not a list or None, \
                           or if an element of originals is not an instance \
                           of DataFlowNode, \
                           or if an element of copies is not an instance \
                           of DataFlowNode.
        :raises ValueError: If the lengths of originals and copies are not \
                            equal.

        :returns: The copied node.
        :rtype: :py:class:`DataFlowNode`
        """
        if not isinstance(dag_copy, (DataFlowDAG, NoneType)):
            raise TypeError(
                f"'dag_copy' argument must be of type 'DataFlowDAG' or "
                f"NoneType but found '{type(dag_copy).__name__}'."
            )
        if not isinstance(originals, (list, NoneType)):
            raise TypeError(
                f"'originals' argument must be of type 'list' or 'NoneType' "
                f"but found '{type(originals).__name__}'."
            )
        if not isinstance(copies, (list, NoneType)):
            raise TypeError(
                f"'copies' argument must be of type 'list' or 'NoneType' but "
                f"found '{type(copies).__name__}'."
            )
        if isinstance(originals, list) and isinstance(copies, list):
            if len(originals) != len(copies):
                raise ValueError(
                    f"The lengths of 'originals' and 'copies' must be equal "
                    f"but found {len(originals)} and {len(copies)} respectively."
                )
        if isinstance(originals, list):
            for original in originals:
                if not isinstance(original, DataFlowNode):
                    raise TypeError(
                        f"'originals' argument must be a list of "
                        f"'DataFlowNode' but found an element of type "
                        f"'{type(original).__name__}'."
                    )
        if isinstance(copies, list):
            for copy in copies:
                if not isinstance(copy, DataFlowNode):
                    raise TypeError(
                        f"'copies' argument must be a list of 'DataFlowNode' "
                        f"but found an element of type '{type(copy).__name__}'."
                    )

        if originals is None:
            originals = []
        if copies is None:
            copies = []

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
        """Copy the current node to a new DAG, recursing along the backward \
        dependences.

        :param dag_copy: The new DAG to copy the node to. 
                         If None, a new DAG is created. Defaults to None.
        :type dag_copy: Union[:py:class:`DataFlowDAG`, NoneType]
        :param originals: The list of original nodes. Defaults to an empty list.
        :type originals: List[:py:class:`DataFlowNode`]
        :param copies: The list of copied nodes. Defaults to None.
        :type copies: Union[List[:py:class:`DataFlowNode`], NoneType]

        :raises TypeError: If dag_copy is not an instance of DataFlowDAG, \
                           or originals is not a list, \
                           or copies is not a list, \
                           or if an element of originals is not an instance \
                           of DataFlowNode, \
                           or if an element of copies is not an instance \
                           of DataFlowNode.
        :raises ValueError: If the lengths of originals and copies are not \
                            equal.

        :returns: The copied node.
        :rtype: :py:class:`DataFlowNode`
        """
        if not isinstance(dag_copy, (DataFlowDAG, NoneType)):
            raise TypeError(
                f"'dag_copy' argument must be of type 'DataFlowDAG' or "
                f"'NoneType' but found '{type(dag_copy).__name__}'."
            )
        if not isinstance(originals, (list, NoneType)):
            raise TypeError(
                f"'originals' argument must be of type 'list' or 'NoneType' "
                f"but found '{type(originals).__name__}'."
            )
        if not isinstance(copies, (list, NoneType)):
            raise TypeError(
                f"'copies' argument must be of type 'list' or 'NoneType' but "
                f"found '{type(copies).__name__}'."
            )
        if isinstance(originals, list) and isinstance(copies, list):
            if len(originals) != len(copies):
                raise ValueError(
                    f"The lengths of 'originals' and 'copies' must be equal "
                    f"but found {len(originals)} and {len(copies)} respectively."
                )
        if isinstance(originals, list):
            for original in originals:
                if not isinstance(original, DataFlowNode):
                    raise TypeError(
                        f"'originals' argument must be a list of "
                        f"'DataFlowNode' but found an element of type "
                        f"'{type(original).__name__}'."
                    )
        if isinstance(copies, list):
            for copy in copies:
                if not isinstance(copy, DataFlowNode):
                    raise TypeError(
                        f"'copies' argument must be a list of 'DataFlowNode' "
                        f"but found an element of type '{type(copy).__name__}'."
                    )

        if originals is None:
            originals = []
        if copies is None:
            copies = []

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
        """Recursively get the PSyIR nodes of the forward dependences of the \
        current node and output them as a list.

        :returns: list of all recursively found PSyIR nodes along the forward \
                  dependences.
        :rtype: List[:py:class:`DataNode`]
        """
        psyir_list = [self.psyir]
        for dep in self.forward_dependences:
            dep_psyir_list = dep.to_psyir_list_forward()
            for psyir in dep_psyir_list:
                if psyir not in psyir_list:
                    psyir_list.append(psyir)
            # psyir_list.extend(dep.to_psyir_list_forward())

        return psyir_list

    def to_psyir_list_backward(self):
        """Recursively get the PSyIR nodes of the backward dependences of the \
        current node and output them as a list.

        :returns: list of all recursively found PSyIR nodes along the backward \
                  dependences.
        :rtype: List[:py:class:`DataNode`]
        """
        psyir_list = [self.psyir]
        for dep in self.backward_dependences:
            dep_psyir_list = dep.to_psyir_list_backward()
            for psyir in dep_psyir_list:
                if psyir not in psyir_list:
                    psyir_list.append(psyir)

        return psyir_list

    def __str__(self):
        """Write a string representation of the DataFlowNode.

        :returns: The string representation of the DataFlowNode.
        :rtype: str
        """
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
        """Write a string representation of the DataFlowNode.

        :returns: The string representation of the DataFlowNode.
        :rtype: str
        """
        return str(self)


class DataFlowDAG:
    """A data flow graph representing the data dependencies between PSyIR \
    nodes.
    """

    def __init__(self):
        self._schedule = None
        self._dag_nodes = []

    @property
    def schedule(self):
        """The PSyIR schedule that the data flow graph is based on, \
        if it was created from one, None otherwise.

        :returns: The PSyIR schedule that the data flow graph is based on, \
                  if it was created from one.
        :rtype: Union[:py:class:`Schedule`, NoneType]
        """
        return self._schedule

    @property
    def dag_nodes(self):
        """The list of DAG nodes in the data flow graph.

        :returns: The list of DAG nodes in the data flow graph.
        :rtype: List[:py:class:`DataFlowNode`]
        """
        return self._dag_nodes

    @property
    def in_arguments_nodes(self):
        """The list of nodes in the data flow graph that are intent(in[out]) \
        arguments.

        :returns: The list of nodes in the data flow graph that are \
                  intent(in[out]) arguments.
        :rtype: List[:py:class:`DataFlowNode`]
        """
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
        """The list of nodes in the data flow graph that are intent([in]out) \
        arguments.

        :returns: The list of nodes in the data flow graph that are \
                  intent([in]out) arguments.
        :rtype: List[:py:class:`DataFlowNode`]
        """
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
        """The list of forward leaves in the data flow graph, ie. nodes with \
        no forward dependences.

        :returns: The list of forward leaves in the data flow graph.
        :rtype: List[:py:class:`DataFlowNode`]
        """
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
        """The list of backward leaves in the data flow graph, ie. nodes with \
        no backward dependences.

        :returns: The list of backward leaves in the data flow graph.
        :rtype: List[:py:class:`DataFlowNode`]
        """
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
        """Create a DataFlowDAG from a PSyIR schedule (Schedule or Routine).

        :param schedule: The PSyIR schedule to create the data flow graph from.
        :type schedule: :py:class:`Schedule`

        :raises TypeError: If the schedule is not an instance of Schedule.

        :returns: The created DataFlowDAG.
        :rtype: :py:class:`DataFlowDAG`
        """
        if not isinstance(schedule, Schedule):
            raise TypeError(
                f"'schedule' argument must be of type 'Schedule' but "
                f"found '{type(schedule).__name__}'."
            )

        dag = cls()
        dag._schedule = schedule

        # Add nodes for the intent(in[out]) arguments of the routine (if any)
        if isinstance(schedule, Routine):
            for argument in schedule.symbol_table.argument_list:
                # intent(in[out]) => write-like node at routine start
                if argument.interface.access is not ArgumentInterface.Access.WRITE:
                    DataFlowNode.create(dag, argument, AccessType.WRITE)

        # Transform all statements found in the schedule, recursing if need be
        dag._statement_list_to_dag_nodes(schedule.children)

        # Add nodes for the intent([in]out) arguments of the routine (if any)
        if isinstance(schedule, Routine):
            for argument in schedule.symbol_table.argument_list:
               # intent(out) => read-like node at routine end
                if argument.interface.access is not ArgumentInterface.Access.READ:
                    DataFlowNode.create(dag, argument, AccessType.READ)

        # # Link the (in)out argument output node to the previous writes
        # for out_arg_node in dag.out_arguments_nodes:
        #     out_arg_node.add_backward_dependence_to_last_write()

        return dag

    def get_all_last_writes_to_array_symbol(self, array_symbol):
        """Get all last writes to the array symbol in the data flow graph.
        This is done by getting all writes to the symbol and then removing \
        the ones where indices are the same.

        :param array_symbol: The array symbol to get the last writes to.
        :type array_symbol: :py:class:`DataSymbol`

        :raises TypeError: If array_symbol is not an instance of DataSymbol \
                           or if array_symbol is not an array.

        :returns: The list of all last writes to the array symbol.
        :rtype: List[:py:class:`DataFlowNode`]
        """
        if not isinstance(array_symbol, DataSymbol):
            raise TypeError(
                f"'array_symbol' argument must be of type 'DataSymbol' but "
                f"found '{type(array_symbol).__name__}'."
            )
        if not isinstance(array_symbol.datatype, ArrayType):
            raise TypeError(
                f"'array_symbol' argument must be an array but found "
                f"'{type(array_symbol.datatype).__name__}'."
            )

        all_writes_to_symbol = self.all_writes_to(array_symbol)
        all_last_writes_to_symbol = all_writes_to_symbol.copy()
        # For every write, check if *it's* previous write is in the list
        # and remove it if so
        for write in all_writes_to_symbol:
            previous_write = self.last_write_before(write)
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
        """Get the DAG node for the given PSyIR node and access type from the \
        data flow graph.

        :param psyir: The PSyIR node to get the DAG node for.
        :type psyir: Union[:py:class:`DataNode`, :py:class:`DataSymbol`]
        :param access_type: The access type which the DAG node must have.
        :type access_type: :py:class:`AccessType`

        :raises TypeError: If psyir is not an instance of DataNode or \
                           DataSymbol, \
                           or if access_type is not an instance of AccessType.
        :raises ValueError: If multiple DAG nodes are found for the given \
                            PSyIR node and access type.

        :returns: The DAG node for the given PSyIR node and access type, \
                  or None if not found.
        :rtype: Union[:py:class:`DataFlowNode`, NoneType]
        """

        if not isinstance(psyir, (DataNode, DataSymbol)):
            raise TypeError(
                f"'psyir' argument must be of type 'DataNode' or 'DataSymbol' "
                f"but found '{type(psyir).__name__}'."
            )
        if not isinstance(access_type, AccessType):
            raise TypeError(
                f"'access_type' argument must be of type 'AccessType' but "
                f"found '{type(access_type).__name__}'."
            )

        dag_nodes_for = []
        for dag_node in self.dag_nodes:
            if dag_node.psyir is psyir and dag_node.access_type is access_type:
                dag_nodes_for.append(dag_node)

        if len(dag_nodes_for) == 0:
            return None

        if len(dag_nodes_for) == 1:
            return dag_nodes_for[0]

        raise ValueError(
            f"Multiple DAG nodes found for PSyIR node '{psyir}' with access "
            f"type '{access_type}'."
        )

    def to_psyir_list(self):
        """Get all PSyIR nodes in the data flow graph as a list.

        :returns: The list of all PSyIR nodes in the data flow graph.
        :rtype: List[Union[:py:class:`DataNode`, :py:class:`DataSymbol`]
        """
        psyir_list = []
        for leaf in self.backward_leaves:
            leaf_psyir_list = leaf.to_psyir_list_forward()
            for psyir in leaf_psyir_list:
                if psyir not in psyir_list:
                    psyir_list.append(psyir)
        return psyir_list

    def dataflow_tree_from(self, psyir):
        """Extract a data flow tree starting from the given PSyIR node and \
        going along the forward dependences.

        :param psyir: The PSyIR node to start the data flow tree from.
        :type psyir: Union[:py:class:`DataNode`, :py:class:`DataSymbol`]

        :raises TypeError: If the given PSyIR node is not an instance of \
                           DataNode or DataSymbol.
        :raises ValueError: If the given PSyIR node is not found in the data \
                            flow graph.
        :raises ValueError: If the given PSyIR node is not the unique backward \
                            leaf of the data flow tree.

        :returns: The data flow tree starting from the given PSyIR node.
        :rtype: :py:class:`DataFlowDAG`
        """
        if not isinstance(psyir, (DataNode, DataSymbol)):
            raise TypeError(
                f"'psyir' argument must be of type 'DataNode' or 'DataSymbol' "
                f"but found '{type(psyir).__name__}'."
            )

        from_node = None
        nodes = self.backward_leaves.copy()
        new_nodes = []
        while len(nodes) != 0:
            for node in nodes:
                if node.psyir == psyir:
                    from_node = node
                else:
                    new_nodes.extend(node.forward_dependences)

            nodes = new_nodes.copy()
            new_nodes = []

        if from_node is None:
            raise ValueError(
                "'from_node' argument not found in the data flow graph."
            )

        tree_from_node = DataFlowDAG()
        copy_from_node = from_node.copy_forward(tree_from_node)

        tree_backward_leaves = tree_from_node.backward_leaves
        if (
            len(tree_backward_leaves) != 1
            or tree_backward_leaves[0] is not copy_from_node
        ):
            raise ValueError(
                "'from_node' argument is not the unique backward leaf of the "
                "data flow tree."
            )

        return tree_from_node

    def dataflow_tree_to(self, psyir):
        """Extract a data flow tree ending in the given PSyIR node and going \
        along the backward dependences.

        :param psyir: The PSyIR node to end the data flow tree in.
        :type psyir: Union[:py:class:`DataNode`, :py:class:`DataSymbol`]

        :raises TypeError: If the given PSyIR node is not an instance of \
                           DataNode or DataSymbol.
        :raises ValueError: If the given PSyIR node is not found in the data \
                            flow graph.
        :raises ValueError: If the given PSyIR node is not the unique forward \
                            leaf of the data flow tree.

        :returns: The data flow tree ending in the given PSyIR node.
        :rtype: :py:class:`DataFlowDAG`
        """
        if not isinstance(psyir, (DataNode, DataSymbol)):
            raise TypeError(
                f"'psyir' argument must be of type 'DataNode' or 'DataSymbol' "
                f"but found '{type(psyir).__name__}'."
            )

        to_node = None
        nodes = self.forward_leaves.copy()
        new_nodes = []
        while len(nodes) != 0:
            for node in nodes:
                if node.psyir == psyir:
                    to_node = node
                else:
                    new_nodes.extend(node.backward_dependences)

            nodes = new_nodes.copy()
            new_nodes = []

        if to_node is None:
            raise ValueError(
                "'to_node' argument not found in the data flow graph."
            )

        tree_to_node = DataFlowDAG()
        copy_to_node = to_node.copy_backward(tree_to_node)

        tree_forward_leaves = tree_to_node.forward_leaves
        if (
            len(tree_forward_leaves) != 1
            or tree_forward_leaves[0] is not copy_to_node
        ):
            raise ValueError(
                "'to_node' argument is not the unique forward leaf of the "
                "data flow tree."
            )

        return tree_to_node

    @property
    def all_reads(self):
        """Get all reads in the data flow graph.

        :returns: The list of all reads in the data flow graph.
        :rtype: List[:py:class:`DataFlowNode`]
        """
        return [
            dag_node
            for dag_node in self.dag_nodes
            if dag_node.access_type is AccessType.READ
        ]

    def all_reads_from(self, psyir):
        """Get all reads from the given PSyIR symbol (or symbol of node) \
        in the data flow graph.

        :param psyir: The PSyIR node or symbol to get all reads from.
        :type psyir: Union[:py:class:`DataNode`, :py:class:`DataSymbol`]

        :raises TypeError: If the given PSyIR node is not an instance of \
                           DataNode or DataSymbol.

        :returns: The list of all reads from the given PSyIR symbol.
        :rtype: List[:py:class:`DataFlowNode`]
        """
        if not isinstance(psyir, (Reference, DataSymbol)):
            raise TypeError(
                f"'psyir' argument must be of type 'Reference' or 'DataSymbol' "
                f"but found '{type(psyir).__name__}'."
            )

        all_reads_from = []
        symbol = psyir if isinstance(psyir, DataSymbol) else psyir.symbol

        for read in self.all_reads:
            if isinstance(read.psyir, DataSymbol):
                if read.psyir == symbol:
                    all_reads_from.append(read)
            elif isinstance(read.psyir, Reference):
                if read.psyir.symbol == symbol:
                    all_reads_from.append(read)

        return all_reads_from

    @property
    def all_writes(self):
        """Get all writes in the data flow graph.

        :returns: The list of all writes in the data flow graph.
        :rtype: List[:py:class:`DataFlowNode`]
        """
        return [
            dag_node
            for dag_node in self.dag_nodes
            if dag_node.access_type is AccessType.WRITE
        ]

    def all_writes_to(self, psyir):
        """Get all writes to the given PSyIR symbol (or symbol of node) in the \
        data flow graph.

        :param psyir: The PSyIR node or symbol to get all writes to.
        :type psyir: Union[:py:class:`DataNode`, :py:class:`DataSymbol`]

        :raises TypeError: If the given PSyIR node is not an instance of \
                           DataNode or DataSymbol.

        :returns: The list of all writes to the given PSyIR symbol.
        :rtype: List[:py:class:`DataFlowNode`]
        """
        if not isinstance(psyir, (Reference, DataSymbol)):
            raise TypeError(
                f"'psyir' argument must be of type 'Reference' or 'DataSymbol' "
                f"but found '{type(psyir).__name__}'."
            )

        all_writes_to = []
        symbol = psyir if isinstance(psyir, DataSymbol) else psyir.symbol

        for write in self.all_writes:
            if isinstance(write.psyir, DataSymbol):
                if write.psyir == symbol:
                    all_writes_to.append(write)
            elif isinstance(write.psyir, Reference):
                if write.psyir.symbol == symbol:
                    all_writes_to.append(write)

        return all_writes_to

    @staticmethod
    def node_position(node):
        """Compute the position of a DAG node in the PSyIR tree.
        For DataSymbols, writes are considered to be at the beginning of the \
        tree and reads at the end.

        :param node: The DAG node to compute the position of.
        :type node: :py:class:`DataFlowNode`

        :raises TypeError: If the given node is not an instance of DataFlowNode.

        :returns: The position of the DAG node in the PSyIR tree.
        :rtype: int
        """
        if not isinstance(node, DataFlowNode):
            raise TypeError(
                f"'node' argument must be of type 'DataFlowNode' but found "
                f"'{type(node).__name__}'."
            )

        if isinstance(node.psyir, DataSymbol):
            if node.access_type is AccessType.WRITE:
                return -1
            elif node.access_type is AccessType.READ:
                return 1000000
            else:
                raise ValueError(
                    f"The access type of the node '{node}' should be either "
                    f"'WRITE' or 'READ' but found '{node.access_type.name}'."
                )
        else:
            return node.psyir.abs_position

    def last_write_before(self, dag_node):
        """Get the last write to the PSyIR from dag_node before it, excluding \
        itself, if any, or None if not found.
        This looks at directives to exclude writes in private scopes.
        This also looks at loops to determine if indices should be considered \
        or not.

        :param dag_node: The DAG node to get the last write before.
        :type dag_node: :py:class:`DataFlowNode`

        :raises TypeError: If the given dag_node is not an instance of \
                           DataFlowNode.

        :returns: The last write before dag_node, or None if not found.
        :rtype: Union[:py:class:`DataFlowNode`, NoneType]
        """
        if not isinstance(dag_node, DataFlowNode):
            raise TypeError(
                f"'dag_node' argument must be of type 'DataFlowNode' but "
                f"found '{type(dag_node).__name__}'."
            )

        last_write = None
        all_writes_before = self.all_writes_to(dag_node.psyir)
        # Filter the writes to the same psyir (symbol) according to their
        # abs_position for nodes or made up position for arguments
        all_writes_before = [
            write
            for write in all_writes_before
            if self.node_position(write) < self.node_position(dag_node)
        ]
        # NOTE: abs_position is not enough for assignments
        # as lhs.abs_position < rhs.abs_position
        # nor for calls, so filter these out
        # Otherwise we'd get the lhs of say 'b = b + a' as the last write
        # before 'b' on the rhs, which is wrong.
        if not isinstance(dag_node.psyir, DataSymbol):
            for write in all_writes_before:
                if not isinstance(write.psyir, DataSymbol):
                    if isinstance(write.psyir.parent, (Assignment, Call)):
                        try:
                            dag_node.psyir.path_from(write.psyir.parent)
                        except ValueError:
                            pass
                        else:
                            all_writes_before.remove(write)

        # Sort the remaining writes by descending position
        all_writes_before.sort(key=self.node_position, reverse=True)

        # TODO: extensive testing...
        # Consider all writes in parallel directives with variable scoping
        # - private or firstprivate for OpenMP
        # - copyin for ACC
        # ie. which are not written back outside of scope
        all_writes_in_private_directives = []
        associated_directives = []
        for write in all_writes_before:
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
                                        dag_node.psyir
                                        if isinstance(
                                            dag_node.psyir, DataSymbol
                                        )
                                        else dag_node.psyir.symbol
                                    ):
                                        all_writes_in_private_directives.append(
                                            write
                                        )
                                        associated_directives.append(directive)

        # If looking for the last write to a Symbol, this is an out argument,
        # all these private writes can be removed
        if isinstance(dag_node.psyir, DataSymbol):
            for write in all_writes_in_private_directives:
                all_writes_before.remove(write)
        # Otherwise, remove all the writes for which there is no path from the
        # directive to the psyir, that is for which the psyir and the write
        # are not in the same directive
        else:
            for write, directive in zip(
                all_writes_in_private_directives, associated_directives
            ):
                try:
                    dag_node.psyir.path_from(directive)
                except ValueError:
                    all_writes_before.remove(write)

        if len(all_writes_before) != 0:
            # For array references, we need to consider indices in some cases
            # - if not, we only look at the symbol
            if isinstance(dag_node.psyir, ArrayReference):
                all_writes_to_possibly_same_index = []
                for write in all_writes_before:
                    # - in arguments write to all indices, so yes
                    if isinstance(write.psyir, DataSymbol):
                        all_writes_to_possibly_same_index.append(write)
                    elif isinstance(write.psyir, ArrayReference):
                        psyir_loop_ancestor = dag_node.psyir.ancestor(Loop)
                        write_loop_ancestor = write.psyir.ancestor(Loop)
                        # - if within same loop or not in loops,
                        # we require indices equality
                        if psyir_loop_ancestor is write_loop_ancestor:
                            if dag_node.psyir.indices == write.psyir.indices:
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
                last_write = all_writes_before[0]

        return last_write

    def _statement_list_to_dag_nodes(self, statement_list):
        """Generate the DAG nodes from a list of PSyIR statements.

        :param statement_list: The list of PSyIR statements to generate the \
                               DAG nodes from.
        :type statement_list: List[:py:class:`Statement`]

        :raises TypeError: If statement_list is not a list or if an element of \
                           statement_list is not an instance of Statement.
        :raises NotImplementedError: If a statement is not yet supported.
        """

        if not isinstance(statement_list, list):
            raise TypeError(
                f"'statement_list' argument must be of type 'list' but found "
                f"'{type(statement_list).__name__}'."
            )
        for statement in statement_list:
            if not isinstance(statement, Statement):
                raise TypeError(
                    f"'statement_list' argument must be a list of 'Statement' "
                    f"but found an element of type '{type(statement).__name__}'."
                )

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
                lhs_node = DataFlowNode.create_or_get(
                    self, lhs, AccessType.WRITE
                )
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
                    self._statement_list_to_dag_nodes(
                        statement.else_body.children
                    )

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
                    self._statement_list_to_dag_nodes(
                        statement.dir_body.children
                    )

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
        """Build a string representation of the data flow graph in DOT format.

        :returns: The string representation of the data flow graph in DOT format.
        :rtype: str
        """
        if isinstance(self.schedule, Routine):
            digraph_name = self.schedule.name
        else:
            digraph_name = "G"
        lines = [f"digraph {digraph_name}", "{"]

        # TODO: subgraphs?

        # Add the nodes
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
                label = (
                    f"{dag_node.psyir.name} "
                    f"({dag_node.psyir.interface.access.name})"
                )
                lines.append(
                    f'{id} [label="{label}", shape="invtriangle", '
                    f'color="{color}"]'
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
                    if isinstance(dag_node.psyir.parent, (OMPFirstprivateClause,
                            OMPSharedClause,
                            ACCCopyClause,
                            ACCCopyInClause,
                            OMPDependClause,
                            OMPPrivateClause, ACCCopyOutClause)):
                        label += f"({type(dag_node.psyir.parent).__name__})"
                        lines.append(
                            f'{id} [label="{label}", shape="diamond", '
                            f'color="{color}"]'
                        )
                    else:
                        lines.append(f'{id} [label="{label}", color="{color}"]')
                else:
                    raise ValueError(type(dag_node.psyir).__name__)

        # Add the edges
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
        """Render the data flow graph as a PNG image using pydot.

        :param filename: name of output PNG file, defaults to "graph".
        :type filename: str, optional
        """
        import pydot

        dot_graph = self.to_dot_format()
        (graph,) = pydot.graph_from_dot_data(dot_graph)
        graph.write_png(f"{filename}.png")

    def __str__(self):
        """Get a string representation of the data flow graph.

        :returns: The string representation of the data flow graph.
        :rtype: str"""
        sorted_dag_nodes = self.dag_nodes
        sorted_dag_nodes.sort(key=self.node_position)
        strings = [str(dag_node) for dag_node in sorted_dag_nodes]
        return "\n".join(strings)

    def __repr__(self):
        """Get a string representation of the data flow graph.

        :returns: The string representation of the data flow graph.
        :rtype: str
        """
        return str(self)


if __name__ == "__main__":
    from psyclone.psyir.frontend.fortran import FortranReader

    reader = FortranReader()

    source1 = """
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

    source2 = """
subroutine foo(a, b)
    real, intent(inout) :: a
    real, intent(inout) :: b
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
end subroutine bar"""

    source3 = """
    subroutine foo(a, b, c)
        real, intent(inout) :: a
        real, intent(inout) :: b
        real, intent(inout) :: c

        !$ omp parallel private(b) firstprivate(c)
            a = b + c
            b = 3.0
            c = 4.0
        !$omp end parallel
    end subroutine foo
    """

    source = source3
    psyir = reader.psyir_from_source(source)
    routine = psyir.children[0]

    if source == source3:
        datasymbol_b = routine.symbol_table.lookup("b")
        datasymbol_c = routine.symbol_table.lookup("c")
        directive = OMPParallelDirective.create(routine.pop_all_children())
        private_clause = OMPPrivateClause.create([datasymbol_b])
        ref_b_private = private_clause.children[0]
        firstprivate_clause = OMPFirstprivateClause.create([datasymbol_c])
        ref_c_firstprivate = firstprivate_clause.children[0]
        directive.children[2] = private_clause
        directive.children[3] = firstprivate_clause
        routine.addchild(directive)

    dag = DataFlowDAG.create_from_schedule(routine)

    #     dag_fwd_leaves = dag.forward_leaves
    #     dag_bwd_leaves = dag.backward_leaves

    #     # print("DAG nodes:")
    #     # for node in dag.dag_nodes:
    #     #     print(node.psyir)

    #     # print("==========\nForward leaves:")
    #     # for leaf in dag_fwd_leaves:
    #     #     print(leaf.psyir)

    #     # print("==========\nBackward leaves:")
    #     # for leaf in dag_bwd_leaves:
    #     #     print(leaf.psyir)

    a, b = routine.symbol_table.argument_list[:2]
    flow_from_a = dag.dataflow_tree_from(a)
    flow_to_b = dag.dataflow_tree_to(b)
    list_from_a = flow_from_a.to_psyir_list()
    list_to_b = flow_to_b.to_psyir_list()

    #     bwd_leaves_from_a = flow_from_a.backward_leaves
    #     print("==========\nBackward leaves:")
    #     for leaf in bwd_leaves_from_a:
    #         print(leaf.psyir)

    #     fwd_leaves_to_b = flow_to_b.forward_leaves
    #     print("==========\nForward leaves:")
    #     for leaf in fwd_leaves_to_b:
    #         print(leaf.psyir)

    #     # print("===========\nFwd and bwd deps for b_arg")
    #     # b_out_arg_node = DataFlowNode.create_or_get(dag, b)
    #     # print(b_out_arg_node.forward_dependences,
    #     #        b_out_arg_node.backward_dependences)

    #     # print("=====\n from a:")
    #     # for psyir in list_from_a:
    #     #     print(psyir)
    #     #     print("------------")
    #     # print("=====\n to b:")
    #     # for psyir in list_to_b:
    #     #     print(psyir)
    #     #     print("------------")
    #     # print(flow_to_b.to_psyir_list())

    dag.render_graph("dag")
    flow_from_a.render_graph("from_a")
    flow_to_b.render_graph("to_b")

#     # print(dag)

#     from psyclone.psyir.symbols import REAL_TYPE
#     from psyclone.psyir.nodes import BinaryOperation

#     dag = DataFlowDAG()
#     datasymbol = DataSymbol("a", REAL_TYPE, interface=ArgumentInterface())
#     datasymbol2 = DataSymbol("b", REAL_TYPE, interface=ArgumentInterface())
#     reference = Reference(datasymbol)
#     reference2 = Reference(datasymbol2)
#     operation = BinaryOperation.create(
#         BinaryOperation.Operator.ADD, reference, reference2
#     )
#     read = AccessType.READ
#     write = AccessType.WRITE

#     node = DataFlowNode.create(dag, operation, AccessType.UNKNOWN)

#     print("===")
#     print(dag.dataflow_tree_from(reference))
#     print("===")
#     print(dag.dataflow_tree_from(reference2))
