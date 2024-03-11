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

"""This module provides an abstract class for reverse-mode 
automatic differentiation "taping" (storing and recovering) of different values.
"""

from abc import ABCMeta
from types import NoneType

from psyclone.psyir.nodes import (ArrayReference, Literal, Node, Range,
                                  BinaryOperation, Reference, DataNode,
                                  IntrinsicCall, Loop, IfBlock, Routine,
                                  Schedule)
from psyclone.psyir.symbols import (DataSymbol, INTEGER_TYPE, ScalarType,
                                    ArrayType)
from psyclone.autodiff import one, zero, simplify_node, add, sub, div, mul
from psyclone.psyir.transformations import TransformationError

# class ScopeNodesOffsetsList(list):
#     """Extends the list class to provide a \
#     [scope, [[recorded node, [first index, last index]]] with types \
#     [Schedule, [[DataNode, [DataNode, DataNode]]]] 
#     list for indexing into an automatic differentiation tape array.
#     TODO: describe this more.
#     """
#     def _typecheck_scope(self, scope):
#         """Typeckeck the scope argument.

#         :param scope: scope to typecheck.
#         :type scope: :py:class:`psyclone.psyir.nodes.Schedule`

#         :raises TypeError: if scope is of the wrong type.
#         """
#         if not isinstance(scope, Schedule):
#             raise TypeError(
#                 f"'scope' argument should be of type "
#                 f"'Schedule' but found '{type(scope).__name__}'."
#             )

#     def _check_is_current_scope(self, scope):
#         """Check that the scope argument is the current scope ie. the last of \
#         the list.

#         :param scope: scope to check.
#         :type scope: :py:class:`psyclone.psyir.nodes.Schedule`

#         :raises TypeError: if scope is of the wrong type.
#         :raises ValueError: if scope is not the current ie. last one.
#         """
#         self._typecheck_scope(scope)
#         if self[-1][0] != scope:
#             raise ValueError("'scope' argument should be the current scope ie. \
#                              the last but is not.")

#     def _validate_item(self, item):
#         """Validates the [recorded node, tape offset] item, which should be of \
#         type List[DataNode, DataNode].

#         :param item: item to check.
#         :type item: List[:py:class:`psyclone.psyir.nodes.DataNode`] of length 2.

#         :raises TypeError: if item is of the wrong type.
#         :raises TypeError: if item is of the wrong length.
#         :raises TypeError: if the items of item are of the wrong type.
#         """
#         if not isinstance(item, list):
#             raise TypeError(
#                 f"'item' argument should be of type "
#                 f"'list' but found '{type(item).__name__}'."
#             )
#         if len(item) != 2:
#             raise TypeError(
#                 f"'item' argument should be of length 2 but found "
#                 f"length '{len(item)}'."
#             )
#         first, second = item
#         if not isinstance(first, DataNode):
#             raise TypeError(
#                 f"'item' argument should be of type "
#                 f"'list' with the first element of "
#                 f"type 'DataNode' but found "
#                 f"'{type(first).__name__}'."
#             )
#         if not isinstance(second, DataNode):
#             raise TypeError(
#                 f"'item' argument should be of type "
#                 f"'list' with the second element of "
#                 f"type 'DataNode' but found '{type(second).__name__}'."
#             )
        
    
        
#     def record(self, node):
#         scope = node.ancestor(Schedule)
#         offset = self.compute_node_offset(node)
#         if self[-1][0] == scope:
#             self[-1][1].append([node, offset])
#         else:
#             self.append([scope, [node, offset]])
    

#     def add_to_current_scope_offset(self, scope, offset_increment, 
#                                     simplify, simplify_times):
#         """Add offset_increment to the current scope (which should match scope)
#         and return the new offset.
#         Optionally applies simplifications to the offset expression.

#         :param scope: scope to check.
#         :type scope: Union[:py:class:`psyclone.psyir.nodes.Loop`,
#                            :py:class:`psyclone.psyir.nodes.IfBlock`,
#                            :py:class:`psyclone.psyir.nodes.Routine`]
#         :param offset_increment: increment to be added to the offset.
#         :type offset_increment: :py:class:`psyclone.psyir.nodes.DataNode`
#         :param simplify: whether to simplify.
#         :type simplify: bool
#         :param simplify_times: number of times simplifications rules should be \
#                                applied.
#         :type simplify_times: int

#         :raises ValueError: if scope is not the current ie. last scope of the \
#                             list.
#         :raises TypeError: if offset_increment is the wrong type.

#         :return: new offset, as a PSyIR node.
#         :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
#         """
#         self._check_current_scope(scope)
#         if not isinstance(offset_increment, DataNode):
#             raise TypeError(
#                 f"'offset_increment' argument should be of type "
#                 f"'DataNode' but found "
#                 f"'{type(offset_increment).__name__}'."
#             )
#         current_scope, current_offset = self[-1]
#         new_offset = add(current_offset, offset_increment)
#         if simplify:
#             new_offset = simplify_node(new_offset, simplify_times)

#         self[-1] = [current_scope, new_offset]
#         return new_offset

#     def enter_new_scope(self, scope):
#         # TODO: branch on scope type and append new (scope, offset) item
#         # loop: use iteration based offset
#         # ????
#         raise NotImplementedError("")

#     def exit_current_scope(self, scope):
#         self._check_current_scope(scope)
#         # TODO: drop current scope and concatenate on the left the new offset, depending on scope type
#         # exit if block: max
#         raise NotImplementedError("")

#     ##############################
#     # IN FACT ALL SCOPES ARE SCHEDULES, 
#     # RULES DEPEND ON THE PARENT NODE IF A LOOP/IFBLOCK
#     # OR THE NODE ITSELF IF A ROUTINE
        
#     ##############################
#     # Wouldn't it be better to store a (scope, [nodes]) thingy and compute the
#     # offset from the nodes????
    


#     def _get_all_offsets(self):
#         return [offset for _, offset in self]

#     def compute_current_total_offset(self, scope, simplify, simplify_times):
#         self._check_current_scope(scope)
#         # TODO: sum all scope offsets
#         all_offsets = self._get_all_offsets()
#         if len(all_offsets) == 0:
#             return zero()
#         if len(all_offsets) == 1:
#             return all_offsets[0]
#         total_offset = all_offsets[0]
#         for next_offset in all_offsets[1:]:
#             total_offset = add(total_offset, next_offset)
#         if simplify:
#             total_offset = simplify_node(total_offset, simplify_times)
#         return total_offset


#     def append(self, item):
#         """Extends list append method with item validation.

#         :param item: two element list to append to the list, \
#                      the first item being a Loop, IfBlock or Routine PSyIR \
#                      node to use as a scope, the second item being a PSyIR \
#                      DataNode to use as an index offset.
#         :type item: List[Union[:py:class:`psyclone.psyir.nodes.Loop`,
#                                :py:class:`psyclone.psyir.nodes.IfBlock`,
#                                :py:class:`psyclone.psyir.nodes.Routine`],
#                          :py:class:`psyclone.psyir.nodes.DataNode`]

#         """
#         self._validate_item(item)
#         super().append(item)

#     def __setitem__(self, index, item):
#         """Extends list __setitem__ method with item validation.

#         :param int index: position where to insert the item. Has to be the \
#                           last index.
#         :param item: two element list to insert in the list, \
#                      the first item being a Loop, IfBlock or Routine PSyIR \
#                      node to use as a scope, the second item being a PSyIR \
#                      DataNode to use as an index offset.
#         :type item: List[Union[:py:class:`psyclone.psyir.nodes.Loop`,
#                                :py:class:`psyclone.psyir.nodes.IfBlock`,
#                                :py:class:`psyclone.psyir.nodes.Routine`],
#                          :py:class:`psyclone.psyir.nodes.DataNode`]

#         :raises NotImplementedError: if the index to set is not the last of \
#                                      the list.

#         """
#         if index not in (-1, len(self) - 1):
#             raise NotImplementedError("ScopesOffsetsList only allows setting "
#                                       "the last item of the list.")
#         self._validate_item(item)
#         super().__setitem__(index, item)

#     def __getitem__(self, index_or_scope):
#         """Extends list __getitem__ method with either integer indexing or \
#         dictionnary like access.
#         Returns the (scope, offset) couple.

#         :param index_or_scope: integer index or scope.
#         :type index_or_scope: Union[int, 
#                                     Union[:py:class:`psyclone.psyir.nodes.Loop`,
#                                       :py:class:`psyclone.psyir.nodes.IfBlock`,
#                                       :py:class:`psyclone.psyir.nodes.Routine`]]

#         :raises TypeError: if index_or_scope is of the wrong type.
#         :raises ValueError: if index_or_scope is a scope but cannot be found \
#                             in the list.

#         :return: item, that is (scope, offset) couple.
#         :rtype: List[Union[:py:class:`psyclone.psyir.nodes.Loop`,
#                            :py:class:`psyclone.psyir.nodes.IfBlock`,
#                            :py:class:`psyclone.psyir.nodes.Routine`],
#                      :py:class:`psyclone.psyir.nodes.DataNode`]
#         """
#         if not isinstance(index_or_scope, (int, Loop, IfBlock, Routine)):
#             raise TypeError(
#                 f"'index_or_scope' argument should be of type "
#                 f"'int', 'Loop', 'IfBlock' or 'Routine' "
#                 f"but found '{type(index_or_scope).__name__}'."
#             )
#         if isinstance(index_or_scope, int):
#             return super().__getitem__(index_or_scope)

#         for scope, offset in self:
#             if scope == index_or_scope:
#                 return scope, offset

#         raise ValueError("'index_or_scope' argument was not of type 'int' "
#                          "but no corresponding scope could be found in the "
#                          "list.")

#     def insert(self, index, item):
#         """Not implemented.
#         :raises NotImplementedError: in all cases.
#         """
#         raise NotImplementedError("Inserting in a ScopesOffsetsList is not "
#                                   "implemented.")

#     def extend(self, items):
#         """Extends list extend method with item validation.

#         :param items: list of items to be appened to the list.
#         :type items: List[List[Union[:py:class:`psyclone.psyir.nodes.Loop`,
#                                      :py:class:`psyclone.psyir.nodes.IfBlock`,
#                                      :py:class:`psyclone.psyir.nodes.Routine`],
#                                :py:class:`psyclone.psyir.nodes.DataNode`]]

#         """
#         for item in items:
#             self._validate_item(item)
#         super().extend(items)

#     def __delitem__(self, index):
#         """Not implemented.
#         :raises NotImplementedError: in all cases.
#         """
#         raise NotImplementedError("__delitem__ on a ScopesOffsetsList is not "
#                                   "implemented.")

#     def remove(self, item):
#         """Extends list remove method with item validation.

#         :param item: item to be deleted the list. Has to be the last item.
#         :type item: List[Union[:py:class:`psyclone.psyir.nodes.Loop`,
#                                :py:class:`psyclone.psyir.nodes.IfBlock`,
#                                :py:class:`psyclone.psyir.nodes.Routine`],
#                          :py:class:`psyclone.psyir.nodes.DataNode`]

#         :raises NotImplementedError: if the item to remove is not the last of \
#                                      the list.

#         """
#         self._validate_item(item)
#         if self[-1] != item:
#             raise NotImplementedError("Removing an item different from the "
#                                       "last item of a ScopesOffsetsList is not "
#                                       "implemented.")
#         self.pop()

#     def pop(self):
#         """Extends list pop method, only allowing popping last item.
#         """
#         return super().pop(-1)

#     def sort(self, reverse=False, key=None):
#         """Not implemented.
#         :raises NotImplementedError: in all cases.
#         """
#         raise NotImplementedError("Sorting a ScopesOffsetsList is not "
#                                   "implemented.")


class ADTape(object, metaclass=ABCMeta):
    """An abstract class for taping values in reverse-mode 
    automatic differentiation.
    Based on Fortran static arrays storing a single type of data rather than \
    on a LIFO stack.
    The Python tape is a list \
    [[scope, [[recorded node, [first index, last index]]]]] with type \
    List[List[Schedule, List[List[DataNode, List[DataNode, DataNode]]]].

    :param name: name of the value_tape (after a prefix).
    :type object: str
    :param datatype: datatype of the elements of the value_tape.
    :type datatype: :py:class:`psyclone.psyir.symbols.ScalarType`
    :param is_dynamic_array: whether to make the Fortran array dynamic \
                             (allocatable) or not. Optional, defaults to False.
    :type is_dynamic_array: Optional[bool]

    :raises TypeError: if name is of the wrong type.
    :raises TypeError: if datatype is of the wrong type.
    :raises TypeError: if is_dynamic_array is of the wrong type.
    :raises TypeError: if uses_offset is of the wrong type.
    """
    # pylint: disable=useless-object-inheritance

    # TODO: this is messy?
    _default_allocate_length = Literal('10', INTEGER_TYPE)

    # NOTE: these should be redefined by subclasses
    _node_types = (Node,)
    _tape_prefix = ""

    def __init__(self, name, datatype, is_dynamic_array = False):
        if not isinstance(name, str):
            raise TypeError(
                f"'name' argument should be of type "
                f"'str' but found '{type(name).__name__}'."
            )
        if not isinstance(datatype, ScalarType):
            raise TypeError(
                f"'datatype' argument should be of type "
                f"'ScalarType' but found "
                f"'{type(datatype).__name__}'."
            )
        if not isinstance(is_dynamic_array, bool):
            raise TypeError(
                f"'is_dynamic_array' argument should be of type "
                f"'bool' but found '{type(is_dynamic_array).__name__}'."
            )

        # PSyIR datatype of the elements in this tape
        self.datatype = datatype

        # Whether to make this tape a dynamic (allocatable) array or not
        self.is_dynamic_array = is_dynamic_array

        # Type of the value_tape
        if is_dynamic_array:
            tape_type = ArrayType(datatype, [ArrayType.Extent.DEFERRED])
        else:
            # is a static array, shape will me modified on the go as needed
            tape_type = ArrayType(datatype, [0])

        # Symbols of the tape
        self.symbol = DataSymbol(self._tape_prefix + name, datatype=tape_type)

        # Symbol of the do loop offset
        self.do_offset_symbol = DataSymbol(self._tape_prefix
                                                + name
                                                + "_do_offset",
                                            datatype=INTEGER_TYPE)

        # Symbol of the offset
        self.offset_symbol = DataSymbol(self._tape_prefix
                                             + name
                                             + "_offset",
                                        datatype=INTEGER_TYPE)
        
        # When the offset variable is used, this is the recorded DataNode whose
        # last index in the tape array it gives
        self.offset_is_last_index_of = None

        # Internal list
        self._tape = []

    @property
    def node_type_names(self):
        """Names of the types of nodes that can be stored in the tape.

        :return: list of type names.
        :rtype: List[Str]`
        """
        return [T.__name__ for T in self._node_types]

    @property
    def datatype(self):
        """PSyIR datatype of the tape elements.

        :return: datatype.
        :rtype: :py:class:`psyclone.psyir.symbols.ScalarType`
        """
        return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        if not isinstance(datatype, ScalarType):
            raise TypeError(
                f"'datatype' argument should be of type "
                f"'ScalarType' but found "
                f"'{type(datatype).__name__}'."
            )
        self._datatype = datatype

    @property
    def is_dynamic_array(self):
        """Whether this array is dynamic (ArrayType.Extent.DEFERRED) or static.

        :return: boolean specifying whether the array is dynamic or not.
        :rtype: bool
        """
        return self._is_dynamic_array

    @is_dynamic_array.setter
    def is_dynamic_array(self, is_dynamic_array):
        if not isinstance(is_dynamic_array, bool):
            raise TypeError(
                f"'is_dynamic_array' argument should be of type "
                f"'bool' but found '{type(is_dynamic_array).__name__}'."
            )
        self._is_dynamic_array = is_dynamic_array

    @property
    def symbol(self):
        """Symbol of the tape.

        :return: data symbol.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        if not isinstance(symbol, DataSymbol):
            raise TypeError(
                f"'symbol' argument should be of type "
                f"'DataSymbol' but found '{type(symbol).__name__}'."
            )
        self._symbol = symbol

    @property
    def name(self):
        """PSyIR name of the tape.

        :return: name.
        :rtype: `Str`
        """
        return self.symbol.name

    @property
    def tape(self):
        """The Python tape which is a list \
        [[scope, [[recorded node, [first index, last index]]]]] of type \
        List[List[Schedule, List[List[DataNode, List[DataNode, DataNode]]]].

        :return: tape.
        :rtype: List[\
                  List[\
                    :py:class:`psyclone.psyir.nodes.Schedule`, \
                    List[List[:py:class:`psyclone.psyir.nodes.DataNode`, \
                              List[:py:class:`psyclone.psyir.nodes.DataNode`, \
                                   :py:class:`psyclone.psyir.nodes.DataNode`]]]]
        """
        return self._tape
    
    @property
    def do_offset_symbol(self):
        """Loop index offset PSyIR DataSymbol. 
        Used for indexing in the tape array **inside** a do loop, \
        depending on the value of the loop variable itself, and to avoid too \
        long index expressions.

        :return: index offset DataSymbol for indexing within a loop.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        return self._do_offset_symbol

    @do_offset_symbol.setter
    def do_offset_symbol(self, do_offset_symbol):
        if not isinstance(do_offset_symbol, DataSymbol):
            raise TypeError(
                f"'do_offset_symbol' argument should be of type "
                f"'DataSymbol' but found "
                f"'{type(do_offset_symbol).__name__}'."
            )
        self._do_offset_symbol = do_offset_symbol

    @property
    def do_offset(self):
        """Loop index offset PSyIR Reference. 
        Used for indexing in the tape array **inside** a do loop, \
        depending on the value of the loop variable itself.

        :return: index offset DataSymbol for indexing within a loop.
        :rtype: :py:class:`psyclone.psyir.symbols.Reference`
        """
        return Reference(self.do_offset_symbol)

    @property
    def offset_symbol(self):
        """Index offset PSyIR DataSymbol. 
        Used for indexing into the tape when loops are present.
        This is used to offset array indices by a run-time dependent number, \
        which depends on loops iterations.

        :return: index offset DataSymbol for indexing into the tape.
        :rtype: :py:class:`psyclone.psyir.symbols.DataSymbol`
        """
        return self._offset_symbol

    @offset_symbol.setter
    def offset_symbol(self, offset_symbol):
        if not isinstance(offset_symbol, DataSymbol):
            raise TypeError(
                f"'offset_symbol' argument should be of type "
                f"'DataSymbol' but found "
                f"'{type(offset_symbol).__name__}'."
            )
        self._offset_symbol = offset_symbol

    @property
    def offset(self):
        """Index offset PSyIR Reference. 
        Used for indexing into the tape and to avoid too long index expressions.

        :return: fresh index offset Reference for indexing into the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.Reference`
        """
        return Reference(self.offset_symbol)
    
    def node_was_recorded(self, node):
        """Check if node was recorded in the tape or not.

        :param node: node to look for in the tape.
        :type node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if node is of the wrong type.

        :return: True is node was recorded in the tape, False otherwise.
        :rtype: bool
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )
        for scope_nodes_indices in self.tape:
            for recorded_node, _ in scope_nodes_indices[1]:
                if node == recorded_node:
                    return True
        
        return False
    
    @property
    def offset_is_last_index_of(self):
        """Recorded node whose last index in the Fortran tape array the offset
        variable of the tape currently gives, if it is used, None otherwise.

        :return: recorded node.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.DataNode`, NoneType]
        """
        return self._offset_is_last_index_of
    
    @offset_is_last_index_of.setter
    def offset_is_last_index_of(self, recorded_node):
        if not isinstance(recorded_node, (*self._node_types, NoneType)):
            raise TypeError(
                f"'recorded_node' argument should be of type among "
                f"{self.node_type_names} or 'NoneType' but found "
                f"'{type(recorded_node).__name__}'."
            )
        if recorded_node is None:
            self._offset_is_last_index_of = None
        else:
            if not self.node_was_recorded(recorded_node):
                raise ValueError(
                    "'recorded_node' argument should have been recorded in the "
                    "tape but it was not found in it."
                )
            self._offset_is_last_index_of = recorded_node

    def compute_first_index_of_new_node(self, node):
        """Compute the first index of the tape where node will be recorded.
        The node argument should not have been recorded in the tape yet.

        :param node: new node to be recorded.
        :type node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if node is of the wrong type.
        :raises ValueError: if node has already been recorded in the tape.

        :return: first index of the node in the Fortran tape array, as a PSyIR \
                 DataNode.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )
        if self.node_was_recorded(node):
                raise ValueError(
                    "'node' argument should not have been recorded in the "
                    "tape but it was found in it."
                )
        
        # TODO!
        raise NotImplementedError("TODO")

    def compute_length_of_node(self, node):
        """Compute the length the node (flattened if needed) will occupy in \
        the Fortran tape array.

        :param node: new node to be recorded.
        :type node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if node is of the wrong type.

        :return: length of the node in the Fortran tape array, as a PSyIR \
                 DataNode.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )
        
        # Nodes of ScalarType correspond to one index of the tape
        if isinstance(node.datatype, ScalarType):
            return one()

        # Nodes of ArrayType correspond to a range
        return self._array_size(node)
        

    def compute_indices_of_new_node(self, node):
        """Compute the first and last indices the node will use in the Fortran \
        tape array.
        The node argument should not have been recorded in the tape yet.

        :param node: new node to be recorded.
        :type node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if node is of the wrong type.
        :raises ValueError: if node has already been recorded in the tape.

        :return: list with 2 items [first index, last index] for this node.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )
        if self.node_was_recorded(node):
                raise ValueError(
                    "'node' argument should not have been recorded in the "
                    "tape but it was found in it."
                )
        first_index = self.compute_first_index_of_new_node(node)
        length = self.compute_length_of_node(node)
        if length == one():
            last_index = first_index
        else:
            # [first, first + length - 1]
            last_index = self._add_datanodes([first_index, 
                                              self._substract_datanodes(length, 
                                                                        one())])

        return [first_index, last_index]
    
    def create_array_reference_from_indices(self, indices):
        """Creates an ArrayReference node tape(index) or tape(first:last) \
        from a list of two indices.

        :param indices: list with 2 items [first index, last index].
        :type indices: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :raises TypeError: if indices is of the wrong type.
        :raises TypeError: if indices is not of length 2.
        :raises TypeError: if an index in indices is of the wrong type.

        :return: PSyIR node 'tape(index)' or 'tape(first:last)'.
        :rtype: ArrayReference
        """
        if not isinstance(indices, list):
            raise TypeError(
                f"'indices' argument should be of type among "
                f"'list' but found '{type(indices).__name__}'."
            )
        if len(indices) != 2:
            raise TypeError(
                f"'indices' argument should be a list of length 2 but found "
                f"length {len(indices)}."
            )
        for index in indices:
            if not isinstance(index, DataNode):
                raise TypeError(
                f"'indices' argument should be of a list of 2 elements of type "
                f"'DataNode' but found an element of type "
                f"'{type(indices).__name__}'."
            )
        first, last = indices
        if first == last:
            return ArrayReference.create(self.symbol, 
                                         [first])
        else:
            tape_range = Range.create(first, last)
            return ArrayReference.create(self.symbol, 
                                         [tape_range])
        
    def get_scope_and_indices_of(self, recorded_node):
        """Returns the scope recorded_node was recorded from and the indices \
        it uses in the Fortran tape array.

        :param recorded_node: node to look for in the tape.
        :type recorded_node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if recorded_node is of the wrong type.
        :raises ValueError: if recorded_node was not recorded in the tape.

        :return: scope and indices for this node.
        :rtype: :py:class:`psyclone.psyir.nodes.Schedule`, \
                List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if not isinstance(recorded_node, self._node_types):
            raise TypeError(
                f"'recorded_node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(recorded_node).__name__}'."
            )
        
        for scope_nodes_indices in self.tape:
            scope, nodes_indices = scope_nodes_indices
            for node, indices in nodes_indices:
                if node == recorded_node:
                    return scope, indices

        raise ValueError(
            "'recorded_node' argument should have been recorded in the "
            "tape but it was not found in it."
        )
    
    def get_scope_of(self, recorded_node):
        """Returns the scope recorded_node was recorded from.

        :param recorded_node: node to look for in the tape.
        :type recorded_node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if recorded_node is of the wrong type.
        :raises ValueError: if recorded_node was not recorded in the tape.

        :return: scope this node was recorded from.
        :rtype: :py:class:`psyclone.psyir.nodes.Schedule`
        """
        if not isinstance(recorded_node, self._node_types):
            raise TypeError(
                f"'recorded_node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(recorded_node).__name__}'."
            )
        
        return self.get_scope_and_indices_of(recorded_node)[0]
    
    def get_indices_of(self, recorded_node):
        """Returns the indices recorded_node uses in the Fortran tape array.

        :param recorded_node: node to look for in the tape.
        :type recorded_node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if recorded_node is of the wrong type.
        :raises ValueError: if recorded_node was not recorded in the tape.

        :return: 2 elements list of [first, last] indices for this node.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if not isinstance(recorded_node, self._node_types):
            raise TypeError(
                f"'recorded_node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(recorded_node).__name__}'."
            )
        
        return self.get_scope_and_indices_of(recorded_node)[1]
    
    def create_array_reference_of(self, recorded_node):
        """Returns a PSyIR ArrayReference node tape(index) or tape(first:last) \
        for recorded_node.

        :param recorded_node: node to look for in the tape.
        :type recorded_node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if recorded_node is of the wrong type.
        :raises ValueError: if recorded_node was not recorded in the tape.

        :return: tape(index) or tape(first:last) node.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(recorded_node, self._node_types):
            raise TypeError(
                f"'recorded_node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(recorded_node).__name__}'."
            )
        
        indices = self.get_indices_of(recorded_node)
        return self.create_array_reference_from_indices(indices)
    
    def create_flatten_call(self, recorded_array_reference):
        """Returns a PSyIR IntrinsicCall with Intrinsic RESHAPE that flattens \
        recorded_array_reference for recording it in the tape.

        :param recorded_array_reference: node to look for in the tape.
        :type recorded_array_reference: \
            :py:class:`psyclone.psyir.nodes.ArrayReference`

        :raises TypeError: if recorded_array_reference is of the wrong type.
        :raises ValueError: if recorded_array_reference was not recorded in \
                            the tape.

        :return: PSyIR call to RESHAPE for the recorded array.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        """
        if not isinstance(recorded_array_reference, ArrayReference):
            raise TypeError(
                f"'recorded_array_reference' argument should be of type "
                f"'ArrayReference' but found "
                f"'{type(recorded_array_reference).__name__}'."
            )
        if not self.node_was_recorded(recorded_array_reference):
            raise ValueError(
                    "'recorded_array_reference' argument should have been "
                    "recorded in the tape but it was not found in it."
                )
        fortran_writer = FortranWriter()
        size = self._array_size(recorded_array_reference)
        size_str = fortran_writer(size)
        shape_array = Literal(f"(/ {size_str} /)",
                              ArrayType(INTEGER_TYPE, [1]))
        return IntrinsicCall.create(IntrinsicCall.Intrinsic.RESHAPE,
                                    [recorded_array_reference.copy(), 
                                     shape_array])
    
    def create_unflatten_call(self, recorded_array_reference):
        """Returns a PSyIR IntrinsicCall with Intrinsic RESHAPE that \
        unflattens recorded_array_reference for restoring it from the tape \
        ie. RESHAPES the tape slice to the correct dimensions.

        :param recorded_array_reference: node to look for in the tape.
        :type recorded_array_reference: \
            :py:class:`psyclone.psyir.nodes.ArrayReference`

        :raises TypeError: if recorded_array_reference is of the wrong type.
        :raises ValueError: if recorded_array_reference was not recorded in \
                            the tape.

        :return: PSyIR call to RESHAPE for the recorded array.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        """
        if not isinstance(recorded_array_reference, ArrayReference):
            raise TypeError(
                f"'recorded_array_reference' argument should be of type "
                f"'ArrayReference' but found "
                f"'{type(recorded_array_reference).__name__}'."
            )
        if not self.node_was_recorded(recorded_array_reference):
            raise ValueError(
                    "'recorded_array_reference' argument should have been "
                    "recorded in the tape but it was not found in it."
                )
        fortran_writer = FortranWriter()
        dimensions = self._array_dimensions(recorded_array_reference)
        str_dimensions = [fortran_writer(dim) for dim in dimensions]
        shape_str = "(/ " + ", ".join(str_dimensions) + " /)"
        shape_datatype = ArrayType(INTEGER_TYPE, [len(dimensions)])
        shape_array = Literal(shape_str, shape_datatype)
        tape_array_reference \
            = self.create_array_reference_of(recorded_array_reference)
        return IntrinsicCall.create(IntrinsicCall.Intrinsic.RESHAPE,
                                    [tape_array_reference, shape_array])
    
    def create_record_assignment(self, recorded_node):
        array_ref = self.create_array_reference_of(recorded_node)
        # TODO
        raise NotImplementedError("TODO")
    
    def create_restore_assignment(self, recorded_node):
        array_ref = self.create_array_reference_of(recorded_node)
        # TODO
        raise NotImplementedError("TODO")
    
    def record(self, node):
        """Record node in the tape. This adds the necessary \
        [scope, [node, [first_index, last_index]]] entry in the Python tape \
        and returns the ArrayReference tape(index) or tape(first:last) that \
        corresponds to node in the Fortran tape array.

        :param node: node to be recorded.
        :type node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if node is of the wrong datatype.
        :raises ValueError: if node was already recorded in the tape.
        :raises NotImplementedError: if the scope of node is not a Routine,
                                     Loop body or IfBlock body.
        :raises TransformationError: if the scope of the node is not the last \
                                     ie. current scope in the tape.

        :return: PSyIR ArrayReference 'tape(index)' or 'tape(first:last)' node.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, DataNode):
            raise TypeError(
                f"'node' argument should be of type "
                f"'DataNode' but found "
                f"'{type(node).__name__}'."
            )
        if isinstance(node.datatype, ScalarType):
            if not node.datatype != self.datatype:
                    raise TypeError(
                        f"'node' argument should be of datatype "
                        f"'{self.datatype.__name__}' but found "
                        f"'{type(node.datatype).__name__}'."
                    ) 
        if isinstance(node.datatype, ArrayType):
            if not node.datatype.datatype != self.datatype:
                    raise TypeError(
                        f"'node' argument is of datatype ArrayType and should "
                        f"have elements of datatype "
                        f"'{self.datatype.__name__}' but found "
                        f"'{type(node.datatype.datatype).__name__}'."
                    ) 
        if self.node_was_recorded(node):
            raise ValueError("'node' argument was already recorded to the tape "
                             "but it should be recorded only once.")
        
        # Get the scope, ie. first Schedule ancestor of the node to record
        scope = node.ancestor(Schedule)
        # Either this is a Routine node, or it should be the body of a Loop or 
        # IfBlock node
        if not isinstance(scope, Routine):
            parent = scope.parent
            if not isinstance(parent, (Loop, IfBlock)):
                raise NotImplementedError("Recording nodes that are not in a "
                                          "Routine, Loop or IfBlock has not "
                                          "been implemented yet.")
        
        # Get the current ie. last scope in the tape and check it's the same
        # as the scope of node
        current_scope, nodes_indices = self.tape[-1]
        if scope != current_scope:
            raise TransformationError("The scope of the 'node' argument is not "
                                      "the last scope in the tape. This "
                                      "probably means a call to enter_new_scope"
                                      " or exit_current_scope is missing.")
            
        # Compute the indices this node will use in the Fortran tape array.
        indices = self.compute_indices_of(node)

        # Add the node and its indices to the tape
        nodes_indices.append([node, indices])

        return self.create_array_reference_of(node)
    
    def restore(self, node):
        """Restore node from the tape. This returns the ArrayReference \
        tape(index) or tape(first:last) that corresponds to node in the \
        Fortran tape array.

        :param node: node to be restored.
        :type node: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if node is of the wrong datatype.
        :raises ValueError: if node was not recorded in the tape.
        :raises NotImplementedError: if the last recorded node was from a \
                                     different scope.
        :raises NotImplementedError: if node was not the last recorded node.

        :return: PSyIR ArrayReference 'tape(index)' or 'tape(first:last)' node.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, DataNode):
            raise TypeError(
                f"'node' argument should be of type "
                f"'DataNode' but found "
                f"'{type(node).__name__}'."
            )
        if isinstance(node.datatype, ScalarType):
            if not node.datatype != self.datatype:
                    raise TypeError(
                        f"'node' argument should be of datatype "
                        f"'{self.datatype.__name__}' but found "
                        f"'{type(node.datatype).__name__}'."
                    ) 
        if isinstance(node.datatype, ArrayType):
            if not node.datatype.datatype != self.datatype:
                    raise TypeError(
                        f"'node' argument is of datatype ArrayType and should "
                        f"have elements of datatype "
                        f"'{self.datatype.__name__}' but found "
                        f"'{type(node.datatype.datatype).__name__}'."
                    ) 
        
        # TODO: for now we check this is the last recorded node in the last 
        # scope
        # This is actually not needed but ensures we add records and restores
        # in the same order in the transformations, for now.
        scope = self.get_scope_of(node)
        last_scope, nodes_indices = self.tape[-1]
        if scope != last_scope:
            raise NotImplementedError("The scope of the 'node' argument is not "
                                      "the last scope a node was recorded from "
                                      "in the tape.")
        last_node = nodes_indices[-1][0]
        if node != last_node:
            raise NotImplementedError("The 'node' argument was not the last "
                                      "node recorded in the tape.")
        
        return self.create_array_reference_of(node)

    def compute_loop_total_number_of_iterations(self, loop_body):
        if not isinstance(loop_body, Schedule):
            raise TypeError(
                f"'loop_body' argument should be of type "
                f"'Schedule' but found "
                f"'{type(loop_body).__name__}'."
            )
        if not isinstance(loop_body.parent, Loop):
            raise TypeError(
                f"Parent node of 'loop_body' argument should be of type "
                f"'Loop' but found "
                f"'{type(loop_body.parent).__name__}'."
            )
        
        loop = loop_body.parent
        start = loop.start_expr
        stop = loop.stop_expr
        step = loop.step_expr

        # Number of iterations of a do loop, using integer division
        # could use MOD instead
        # step*((stop - start)/step) + start
        return add(mul(step, div(sub(stop, start), step)), start)
    
    def compute_loop_iteration_number(self, loop_body):
        if not isinstance(loop_body, Schedule):
            raise TypeError(
                f"'loop_body' argument should be of type "
                f"'Schedule' but found "
                f"'{type(loop_body).__name__}'."
            )
        if not isinstance(loop_body.parent, Loop):
            raise TypeError(
                f"Parent node of 'loop_body' argument should be of type "
                f"'Loop' but found "
                f"'{type(loop_body.parent).__name__}'."
            )
        
        loop = loop_body.parent
        var = loop.variable
        start = loop.start_expr
        step = loop.step_expr

        #(var - start)/step
        return div(sub(var, start), step)
    
    def get_all_scope_indices_in_tape(self, scope):
        if not isinstance(scope, Schedule):
            raise TypeError(
                f"'scope' argument should be of type "
                f"'Schedule' but found "
                f"'{type(scope).__name__}'."
            )
        scope_indices_in_tape = []
        for i, (scope, _) in enumerate(self.tape):
            if scope == scope:
                scope_indices_in_tape.append(i)
        return scope_indices_in_tape
    
    def compute_loop_taping_length_per_iteration(self, loop_body):
        if not isinstance(loop_body, Schedule):
            raise TypeError(
                f"'loop_body' argument should be of type "
                f"'Schedule' but found "
                f"'{type(loop_body).__name__}'."
            )
        if not isinstance(loop_body.parent, Loop):
            raise TypeError(
                f"Parent node of 'loop_body' argument should be of type "
                f"'Loop' but found "
                f"'{type(loop_body.parent).__name__}'."
            )
        
        # WRONG! what if recording in loop, then in subscope, then in loop...
        # 

        loop_body_indices = self.get_all_scope_indices_in_tape(loop_body)
        if len(loop_body_indices) == 0:
            return zero()
        if len(loop_body_indices) == 1:
            scope, nodes_indices = self.tape[loop_body_indices[0]]
            nodes_lengths = []
            for node, _ in nodes_indices:
                nodes_lengths.append(self.compute_length_of(node))
            return self._add_datanodes(nodes_lengths)
        else:
            first_scope_index = loop_body_indices[0]
            last_scope_index = loop_body_indices[-1]
            lengths = []
            for scope, nodes_indices \
                in self.tape[first_scope_index:last_scope_index + 1]:
                if scope == loop_body:
                    for node, _ in nodes_indices:
                        lengths.append(self.compute_length_of(node))
                else:
                    if isinstance(scope.parent, Loop):
                        lengths.append(self.compute_loop_total_taping_length(scope))
                    elif isinstance(scope.parent, IfBlock):
                        lengths.append(self.compute_if_block_taping_length(scope))
            # calls???
            return self._add_datanodes(lengths)


    
    def compute_loop_total_taping_length(self, loop_body):
        if not isinstance(loop_body, Schedule):
            raise TypeError(
                f"'loop_body' argument should be of type "
                f"'Schedule' but found "
                f"'{type(loop_body).__name__}'."
            )
        if not isinstance(loop_body.parent, Loop):
            raise TypeError(
                f"Parent node of 'loop_body' argument should be of type "
                f"'Loop' but found "
                f"'{type(loop_body.parent).__name__}'."
            )
        
        return mul(self.compute_loop_iteration_number(loop_body),
                   self.compute_loop_taping_length_per_iteration(loop_body))
    
    def compute_loop_iteration_offset(self, loop_body):
        if not isinstance(loop_body, Schedule):
            raise TypeError(
                f"'loop_body' argument should be of type "
                f"'Schedule' but found "
                f"'{type(loop_body).__name__}'."
            )
        if not isinstance(loop_body.parent, Loop):
            raise TypeError(
                f"Parent node of 'loop_body' argument should be of type "
                f"'Loop' but found "
                f"'{type(loop_body.parent).__name__}'."
            )
        raise NotImplementedError("TODO")
    
    def create_loop_offset_assignment(self, loop_body):
        if not isinstance(loop_body, Schedule):
            raise TypeError(
                f"'loop_body' argument should be of type "
                f"'Schedule' but found "
                f"'{type(loop_body).__name__}'."
            )
        if not isinstance(loop_body.parent, Loop):
            raise TypeError(
                f"Parent node of 'loop_body' argument should be of type "
                f"'Loop' but found "
                f"'{type(loop_body.parent).__name__}'."
            )
        raise NotImplementedError("TODO")
    
    def compute_if_or_else_body_taping_length(self, if_or_else_body):
        if not isinstance(if_or_else_body, Schedule):
            raise TypeError(
                f"'if_or_else_body' argument should be of type "
                f"'Schedule' but found "
                f"'{type(if_or_else_body).__name__}'."
            )
        if not isinstance(if_or_else_body.parent, IfBlock):
            raise TypeError(
                f"Parent node of 'if_or_else_body' argument should be of type "
                f"'IfBlock' but found "
                f"'{type(if_or_else_body.parent).__name__}'."
            )
        raise NotImplementedError("TODO")
    
    def compute_if_block_taping_length(self, if_block):
        if not isinstance(if_block, IfBlock):
            raise TypeError(
                f"'if_block' argument should be of type "
                f"'IfBlock' but found "
                f"'{type(if_block.parent).__name__}'."
            )
        # Max
        raise NotImplementedError("TODO")
    
    def __str__(self):
        fortran_writer = FortanWriter()
        string = f"Tape '{self.name}' of length {self.length}\n"
        for scope, nodes_indices in self.tape:
            if isinstance(scope, Routine):
                string += f"\tRoutine '{scope.name}':\n"
            else:
                if isinstance(scope.parent, Loop):
                    loop = scope.parent
                    start_str = fortran_writer(loop.start_expr)
                    stop_str = fortran_writer(loop.stop_expr)
                    step_str = fortran_writer(loop.step_expr)
                    loop_header = f"do {loop.variable.name} = {start_str}, {stop_str}, {step_str}"
                    string += f"\tLoop '{loop_header}':\n"
                elif isinstance(scope.parent, IfBlock):
                    if_block = scope.parent
                    string += f"\tIfBlock {fortran_writer(if_block.condition)} "
                    if scope == if_block.if_body:
                        string += "if body:\n"
                    else:
                        string += "else body:\n"
                        
            for node, (first, last) in nodes_indices:
                string += f"\t\t{fortran_writer(node)} at indices {fortran_writer(first)}:{fortran_writer(last)}"

        return string
    
    def __repr__(self):
        return self.__str__()


    
    # TODO: record, restore, print
    # TODO: compute_loop_iterations_count, compute_loop_taping_length
    # TODO: compute_loop_iteration_offset
    # TODO: loop_offset_assignment
    # TODO: compute_if_block_tape_length
    # etc
    # don't??? exit_current_scope, enter_new_scope, etc.

    def _typecheck_list_of_int_literals(self, int_literals):
        """Check that the argument is a list of scalar integer literals.

        :param int_literals: list of scalar integer literals.
        :type int_literals: List[:py:class:`psyclone.psyir.nodes.Literal`]

        :raises TypeError: if int_literals is of the wrong type.
        :raises TypeError: if an element of int_literals is of the wrong type.
        :raises ValueError: if an element of int_literals is not of datatype \
                            ScalarType.
        :raises ValueError: if an element of int_literals is not of intrinsic \
                            ScalarType.Intrinsic.INTEGER.
        """
        if not isinstance(int_literals, list):
            raise TypeError(f"'int_literals' argument should be of type 'list' "
                            f"but found '{type(int_literals).__name__}'.")
        for literal in int_literals:
            if not isinstance(literal, Literal):
                raise TypeError(f"'int_literals' argument should be a 'list' "
                                f"of elements of type 'Literal' "
                                f"but found '{type(literal).__name__}'.")
            if not isinstance(literal.datatype, ScalarType):
                raise ValueError(f"'int_literals' argument should be a 'list' "
                                 f"of elements of datatype 'ScalarType' but "
                                 f"found '{type(literal.datatype).__name__}'.")
            if literal.datatype.intrinsic is not ScalarType.Intrinsic.INTEGER:
                raise ValueError(f"'int_literals' argument should be a 'list' "
                            f"of elements of intrinsic "
                            f"'ScalarType.Intrinsic.INTEGER' but found "
                            f"'{type(literal.datatype.intrinsic).__name__}'.")

    def _typecheck_list_of_datanodes(self, datanodes):
        """Check that the argument is a list of datanodes.

        :param datanodes: list of datanodes.
        :type datanodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :raises TypeError: if datanodes is of the wrong type.
        :raises TypeError: if an element of datanodes is of the wrong type.
        """
        if not isinstance(datanodes, list):
            raise TypeError(f"'datanodes' argument should be of type 'list' "
                            f"but found '{type(datanodes).__name__}'.")
        for datanode in datanodes:
            if not isinstance(datanode, DataNode):
                raise TypeError(f"'datanodes' argument should be a 'list' "
                                f"of elements of type 'DataNode' "
                                f"but found '{type(datanode).__name__}'.")

    def _separate_int_literals(self, datanodes):
        """| Separates the datanodes from a list into:
        | - a list of scalar integer Literals,
        | - a list of other datanodes. 

        :param datanodes: list of datanodes to separate.
        :type datanodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :return: list of integer literals, list of other datanodes
        :rtype: List[:py:class:`psyclone.psyir.nodes.Literal`], \
                List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        self._typecheck_list_of_datanodes(datanodes)

        int_literals = []
        other_datanodes = []
        for datanode in datanodes:
            if (isinstance(datanode, Literal)
                and isinstance(datanode.datatype, ScalarType)
                and (datanode.datatype.intrinsic
                     is ScalarType.Intrinsic.INTEGER)):
                int_literals.append(datanode)
            else:
                other_datanodes.append(datanode)

        return int_literals, other_datanodes

    def _add_int_literals(self, int_literals):
        """Add the int Literals from a list, summing in Python and returning \
        a new Literal.

        :param datanodes: list of literals.
        :type datanodes: List[:py:class:`psyclone.psyir.nodes.Literal`]

        :return: sum, as a Literal.
        :rtype: :py:class:`psyclone.psyir.nodes.Literal`
        """
        self._typecheck_list_of_int_literals(int_literals)

        result = 0
        for literal in int_literals:
            result += int(literal.value)

        return Literal(str(result), INTEGER_TYPE)

    def _add_datanodes(self, datanodes):
        """Add the datanodes from a list, dealing with Literals in Python \
        and others in BinaryOperations.

        :param datanodes: list of datanodes.
        :type datanodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :return: sum, as a Literal or BinaryOperation.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`, \
                      :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """
        self._typecheck_list_of_datanodes(datanodes)

        int_literals, other_datanodes = self._separate_int_literals(datanodes)
        int_sum = self._add_int_literals(int_literals)

        if int_sum.value != "0":
            other_datanodes.append(int_sum)

        result = zero()
        if len(other_datanodes) != 0:
            result = other_datanodes[0]
            if len(other_datanodes) > 1:
                for datanode in other_datanodes[1:]:
                    result = BinaryOperation.create(
                                BinaryOperation.Operator.ADD,
                                result.copy(),
                                datanode.copy())

        return result

    def _substract_datanodes(self, lhs, rhs):
        """Substract the datanodes from two lists, dealing with int Literals \
        in Python and others in BinaryOperations.

        :param lhs: list of datanodes to sum as lhs of '-'.
        :type lhs: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        :param rhs: list of datanodes to sum as rhs of '-'.
        :type rhs: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :return: substraction, as a Literal or BinaryOperation.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`, \
                      :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """
        self._typecheck_list_of_datanodes(lhs)
        self._typecheck_list_of_datanodes(rhs)

        lhs_int_literals, lhs_others = self._separate_int_literals(lhs)
        rhs_int_literals, rhs_others = self._separate_int_literals(rhs)

        lhs_int_sum = self._add_int_literals(lhs_int_literals)
        rhs_int_sum = self._add_int_literals(rhs_int_literals)

        int_literal = Literal(str(int(lhs_int_sum.value)
                                  - int(rhs_int_sum.value)),
                              INTEGER_TYPE)

        if int_literal.value != "0":
            lhs_others.append(int_literal)

        result = zero()
        if len(lhs_others) != 0:
            result = lhs_others[0]
            if len(lhs_others) > 1:
                for datanode in lhs_others[1:]:
                    result = BinaryOperation.create(
                                BinaryOperation.Operator.ADD,
                                result.copy(),
                                datanode.copy())

        if len(rhs_others) != 0:
            substract = rhs_others[0]
            if len(rhs_others) > 1:
                for datanode in rhs_others[1:]:
                    substract = BinaryOperation.create(
                                    BinaryOperation.Operator.ADD,
                                    substract.copy(),
                                    datanode.copy())
            result = BinaryOperation.create(BinaryOperation.Operator.SUB,
                                            result.copy(),
                                            substract.copy())

        return result

    def _multiply_int_literals(self, int_literals):
        """Multiply the int literals from a list.
        Performs the multiplication in Python and returns a new Literal.

        :param datanodes: list of literals.
        :type datanodes: List[:py:class:`psyclone.psyir.nodes.Literal`]

        :return: multiplication, as a Literal.
        :rtype: :py:class:`psyclone.psyir.nodes.Literal`
        """
        self._typecheck_list_of_int_literals(int_literals)

        result = 1
        for literal in int_literals:
            result *= int(literal.value)

        return Literal(str(result), INTEGER_TYPE)

    def _multiply_datanodes(self, datanodes):
        """Multiply the datanodes from a list, dealing with Literals in Python \
        and others in BinaryOperations.

        :param datanodes: list of datanodes.
        :type datanodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :return: multiplication, as a Literal or BinaryOperation.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`, \
                      :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """
        self._typecheck_list_of_datanodes(datanodes)

        int_literals, other_datanodes = self._separate_int_literals(datanodes)
        int_mul = self._multiply_int_literals(int_literals)

        if int_mul.value != "1":
            other_datanodes.append(int_mul)

        result = one()
        if len(other_datanodes) != 0:
            result = other_datanodes[0]
            if len(other_datanodes) > 1:
                for datanode in other_datanodes[1:]:
                    result = BinaryOperation.create(
                                BinaryOperation.Operator.MUL,
                                result.copy(),
                                datanode.copy())

        return result

    def length(self):
        """Total length of the tape (Fortran) array ie. last index of the \
        last recorded element.

        :return: length of the tape, as a Literal or sum (BinaryOperation).
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`, \
                      :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """

        # TODO: make sure that if last node is within a do loop or if/else body
        # this returns the total length independently of the branch or loop var

        raise NotImplementedError("TODO")

#        lengths = []
#        for node, multiplicity in self.recorded_nodes:
#            if isinstance(node.datatype, ScalarType):
#                lengths.append(multiplicity)
#            else:
#                lengths.append(self._multiply_datanodes([self._array_size(node),
#                                                         multiplicity]))
#
#        # Within a do loop, indexing in the tape array uses the do offset
#        # variable, which depends on the loop index. Add it if necessary.
#        if do_loop:
#            lengths.append(self.do_offset)
#
#        # If the index offset is used (to offset indexing after exiting a loop
#        # body), add it
#        if self.offset is not None:
#            lengths.append(self.offset)
#
#        return self._add_datanodes(lengths)

    def first_index_of_last_element(self, do_loop = False):
        """Gives the first index of the last element that was recorded.

        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if do_loop is of the wrong type.

        :return: Literal or BinaryOperation giving the index.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`,
                      :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        lengths = []
        for node, multiplicity in self.recorded_nodes[:-1]:
            if isinstance(node.datatype, ScalarType):
                lengths.append(multiplicity)
            else:
                lengths.append(self._multiply_datanodes([self._array_size(node),
                                                         multiplicity]))

        # Within a do loop, indexing in the tape array uses the do offset
        # variable, which depends on the loop index. Add it if necessary.
        if do_loop:
            lengths.append(self.do_offset)

        # If the index offset is used (to offset indexing after exiting a loop
        # body), add it
        if self.offset is not None:
            lengths.append(self.offset)

        lengths.append(one())

        return self._add_datanodes(lengths)

    def _array_size(self, array):
        """Returns the BinaryOperation giving the size of the array.

        TODO: NEW_ISSUE this should simply be a SIZE operation with no second \
        argument but this can't be done yet.

        :param array: array.
        :type array: py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if array is of the wrong type.
        :raises ValueError: if the datatype of array is not an ArrayType.

        :return: size of the array, as a MUL BinaryOperation.
        :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
        """
        if not isinstance(array, Reference):
            raise TypeError(f"'array' argument should be of type 'Reference' "
                            f"but found '{type(array).__name__}'.")
        if not isinstance(array.datatype, ArrayType):
            raise ValueError(f"'array' argument should be of datatype "
                             f"'ArrayType' but found "
                             f"'{type(array.datatype).__name__}'.")

        # NEW_ISSUE: SIZE has optional second argument, this should be fixed
        # in FortranWriter & co.
        # return BinaryOperation.create(BinaryOperation.Operator.SIZE,
        #                               array.copy(),
        #                               None)

        dimensions = self._array_dimensions(array)

        return self._multiply_datanodes(dimensions)

    def _array_dimensions(self, array):
        """Returns a list of Literals or BinaryOperations giving the \
        dimensions of the array.

        :param array: array.
        :type array: py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if array is of the wrong type.
        :raises ValueError: if the datatype of array is not an ArrayType.

        :return: dimensions of the array as a list of Literals or \
                 BinaryOperations.
        :rtype: List[Union[:py:class:`psyclone.psyir.nodes.Literal`,
                           :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """
        if not isinstance(array, Reference):
            raise TypeError(f"'array' argument should be of type 'Reference' "
                            f"but found '{type(array).__name__}'.")
        if not isinstance(array.datatype, ArrayType):
            raise ValueError(f"'array' argument should be of datatype "
                             f"'ArrayType' but found "
                             f"'{type(array.datatype).__name__}'.")

        # Go through the shape attribute of the ArrayType
        dimensions = []
        for dim, shape in enumerate(array.datatype.shape):
            # For array bounds, compute upper + 1 - lower
            if isinstance(shape, ArrayType.ArrayBounds):
                plus = [shape.upper, one()]
                minus = [shape.lower]
                dimensions.append(self._substract_datanodes(plus, minus))
            else:
                # For others, compute SIZE(array, dim)
                size_operation = IntrinsicCall.create(
                                    IntrinsicCall.Intrinsic.SIZE,
                                    [array.copy(),
                                     Literal(str(dim), INTEGER_TYPE)])
                dimensions.append(size_operation)

        # from psyclone.psyir.backend.fortran import FortranWriter
        # fortran_writer = FortranWriter()
        # dim_str = [fortran_writer(dim) for dim in dimensions]
        # print(f"Array {fortran_writer(array)}, found dimensions {dim_str}")

        return dimensions

    def _has_last(self, node):
        """Check that the last node recorded to the tape is the one passed \
        as argument.

        :param node: node to be checked.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :raises TypeError: if node is of the wrong type.
        :raises ValueError: if the node is not the last element of the tape.
        """

        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )
        if self.recorded_nodes[-1][0] != node:
            raise ValueError(
                f"node argument named {node.name} was not "
                f"stored as last element of the value_tape."
            )

    def record(self, node, do_loop = False):
        """Add the node as last element of the tape and return the \
        ArrayReference node of the tape.

        :param node: node whose prevalue should be recorded.
        :type node: :py:class:`psyclone.psyir.nodes.Reference`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if the intrinsic of node's datatype is not the \
                           same as the intrinsic of the value_tape's \
                           elements datatype.
        :raises TypeError: if do_loop is of the wrong type.

        :return: the array node to the last element of the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )

        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        # Nodes of ScalarType correspond to one index of the tape
        if isinstance(node.datatype, ScalarType):
            self.recorded_nodes.append([node, one()])
            # If static array, reshape to take the new length into account
            if not self.is_dynamic_array:
                self.reshape()
            # This is the Fortran index, starting at 1
            tape_ref = ArrayReference.create(self.symbol, 
                                             [self.length(do_loop)])

        # Nodes of ArrayType correspond to a range
        else:
            self.recorded_nodes.append([node, one()])
            # If static array, reshape to take the new length into account
            if not self.is_dynamic_array:
                self.reshape()
            tape_range = Range.create(self.first_index_of_last_element(do_loop),
                                      self.length(do_loop))
            tape_ref = ArrayReference.create(self.symbol, [tape_range])

        return tape_ref

    def restore(self, node, do_loop = False):
        """Check that node is the last element of the tape and return an \
        ArrayReference to it in the tape.

        :param node: node restore.
        :type node: :py:class:`psyclone.psyir.nodes.Node`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if do_loop is of the wrong type.

        :return: an ArrayReference node to the last element of the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        self._has_last(node)

        # Nodes of ScalarType correspond to one index of the tape
        if isinstance(node.datatype, ScalarType):
            # This is the Fortran index, starting at 1
            tape_ref = ArrayReference.create(self.symbol, 
                                             [self.length(do_loop)])

        # Nodes of ArrayType correspond to a range
        else:
            tape_range = Range.create(self.first_index_of_last_element(do_loop),
                                      self.length(do_loop))
            tape_ref = ArrayReference.create(self.symbol, [tape_range])

        return tape_ref

    def reshape(self, do_loop = False):
        """Change the static length of the tape array in its datatype.

        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if do_loop is of the wrong type.
        """
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        value_tape_type = ArrayType(self.datatype,
                                    [self.length(do_loop)])
        self.symbol.datatype = value_tape_type

    def allocate(self, length = _default_allocate_length):
        """Generates a PSyIR IntrinsicCall ALLOCATE node for the tape, with \
        the given length.

        :param length: length to allocate.
        :type length: Union[int, :py:class:`psyclone.psyir.nodes.Literal`]

        :raises TypeError: if length is of the wrong type.
        :raises ValueError: if length is a Literal but not of ScalarType \
                            or not an integer.

        :return: the ALLOCATE statement as a PSyIR IntrinsicCall node.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        """
        if not isinstance(length, (int, Literal)):
            raise TypeError(
                f"'length' argument should be of type "
                f"'int' or 'Literal' but found "
                f"'{type(length).__name__}'."
            )
        if isinstance(length, int):
            length = Literal(str(length), INTEGER_TYPE)
        elif (isinstance(length, Literal) and
              (not isinstance(length.datatype, ScalarType)
               or length.datatype.intrinsic is not INTEGER_TYPE.intrinsic)):
            raise ValueError(f"'length' argument is a 'Literal' but either its "
                             f"datatype is not 'ScalarType' or it is not an "
                             f"integer. Found {length}.")

        return IntrinsicCall.create(IntrinsicCall.Intrinsic.ALLOCATE,
                                    [ArrayReference.create(self.symbol,
                                                           [length.copy()])])

    def deallocate(self):
        """Generates a PSyIR IntrinsicCall ALLOCATE node for the tape.

        :return: the DEALLOCATE statement as a PSyIR IntrinsicCall node.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        """
        return IntrinsicCall.create(IntrinsicCall.Intrinsic.DEALLOCATE,
                                    [Reference(self.symbol)])

    def extend(self, tape):
        """Extends the tape with the recorded nodes of the 'tape' argument, \
        which must be of the same type.

        :param tape: tape to combine.
        :type tape: :py:class:`psyclone.autodiff.ADTape`, same as self.

        :raises TypeError: if tape is of the wrong type.
        :raises TypeError: if the tape datatype is different.
        """
        if not isinstance(tape, type(self)):
            raise TypeError(
                f"'tape' argument should be of type "
                f"'{type(self).__name__}' but found "
                f"'{type(tape).__name__}'."
            )
        if tape.datatype != self.datatype:
            raise TypeError(
                f"'tape' argument should have elements of datatype "
                f"'{self.datatype}' but found "
                f"'{tape.datatype}'."
            )

        self.recorded_nodes.extend(tape.recorded_nodes)

        # If static array, reshape to take the new length into account
        if not self.is_dynamic_array:
            self.reshape()

    def extend_and_slice(self, tape, do_loop = False):
        """Extends the tape by the 'tape' argument and return \
        the ArrayReference corresponding to the correct slice.

        :param tape: tape to extend with.
        :type tape: :py:class:`psyclone.autodiff.tapes.ADTape`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if tape is not of the same type as self.
        :raises ValueError: if the datatype of tape is not the same as \
            the datatype of self. 
        :raises TypeError: if do_loop is of the wrong type.

        :return: slice of the tape array that corresponds \
            to the tape it was extended with.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """

        if not isinstance(tape, type(self)):
            raise TypeError(
                f"'tape' argument should be of type "
                f"'{type(self).__name__}' but found "
                f"'{type(tape).__name__}'."
            )
        if tape.datatype != self.datatype:
            raise TypeError(
                f"'tape' argument should have elements of datatype "
                f"'{self.datatype}' but found "
                f"'{tape.datatype}'."
            )
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )
        # First index of the slice corresponding to the "new" tape
        first_index = self._add_datanodes([self.length(do_loop),
                                           one()])
        # Extend the parent value_tape with the new value_tape
        self.extend(tape)
        # Last index of the slice
        last_index = self.length(do_loop)
        # Slice of the parent value_tape
        value_tape_range = Range.create(first_index, last_index)

        return ArrayReference.create(self.symbol, [value_tape_range])

    def length_of_last_recorded_nodes(self, nodes, multiplicity):
        """Computes the total length of the last recorded nodes, which should \
        match the ones in 'nodes'.
        This takes into account the individual lengths of recorded arrays and \
        applies 'multiplicity' as a common factor.
        Used both in computing the (internal) do loop offset and the 'global' \
        tape offset.

        :param nodes: list of nodes whose length to compute.
        :type nodes: List[:py:class:`psyclone.psyir.nodes.Node`]
        :param multiplicity: multiplicity to be used, as a PSyIR node.
        :type multiplicity: :py:class:`psyclone.psyir.nodes.DataNode`

        :raises TypeError: if nodes is of the wrong type.
        :raises TypeError: if an element of nodes is of the wrong type.
        :raises TypeError: if multiplicity is of the wrong type.
        :raises ValueError: if a recorded node doesn't match the corresponding \
                            one in 'nodes'.

        :return: length of the recorded nodes in the tape, with multiplicity \
                 taken into account, as a PSyIR node.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """

        if not isinstance(nodes, list):
            raise TypeError(
                f"'nodes' argument should be of type "
                f"'list[Node]' but found '{type(nodes).__name__}'."
            )
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError(
                    f"'nodes' argument should be of type "
                    f"'list[Node]' but found an element of type "
                    f"'{type(node).__name__}'."
                )
        if not isinstance(multiplicity, DataNode):
            raise TypeError(
                f"'multiplicity' argument should be of type "
                f"'DataNode' but found "
                f"'{type(multiplicity).__name__}'."
            )

        # TODO: get rid of the stored multiplicity?
        # Go through both the recorded nodes (end of the tape) and the nodes to
        # edit, check they match
        for index, ((recorded_node, _),
                    (node)) in enumerate(zip(self.recorded_nodes[-len(nodes):],
                                             nodes)):
            if recorded_node != node:
                raise ValueError(
                    f"'nodes' list contains a different node at index {index} "
                    f"than the one recorded in the tape."
                )

        lengths = []
        for node in nodes:
            if isinstance(node.datatype, ScalarType):
                lengths.append(one())
            else:
                lengths.append(self._array_size(node))

        length = self._add_datanodes(lengths)

        return self._multiply_datanodes([multiplicity, length])

    def change_last_nodes_multiplicity(self, nodes, multiplicity):
        """Change the multiplicity of the last recorded nodes, which should \
        match the ones in 'nodes'.
        Used after transforming a loop so that the tape offset is correct \
        based on the number of iterations that were performed.

        :param nodes: list of nodes whose multiplicities should be changed.
        :type nodes: List[:py:class:`psyclone.psyir.nodes.Node`]
        :param multiplicity: new multiplicity to be used, as a PSyIR node.
        :type multiplicity: Union[:py:class:`psyclone.psyir.nodes.Literal`,\
                               :py:class:`psyclone.psyir.nodes.Reference`,\
                               :py:class:`psyclone.psyir.nodes.BinaryOperation`]

        :raises TypeError: if nodes is of the wrong type.
        :raises TypeError: if an element of nodes is of the wrong type.
        :raises TypeError: if multiplicity is of the wrong type.
        :raises ValueError: if a recorded node doesn't match the corresponding \
                            one in 'nodes'.
        :raises ValueError: if the recorded node already has multiplicity \
                            different from 1.
        """
        if not isinstance(nodes, list):
            raise TypeError(
                f"'nodes' argument should be of type "
                f"'list[Node]' but found '{type(nodes).__name__}'."
            )
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError(
                    f"'nodes' argument should be of type "
                    f"'list[Node]' but found an element of type "
                    f"'{type(node).__name__}'."
                )
        if not isinstance(multiplicity, (Literal, Reference, BinaryOperation)):
            raise TypeError(
                f"'multiplicity' argument should be of type "
                f"'Literal', 'Reference' or 'Binary Operation' but found "
                f"'{type(multiplicity).__name__}'."
            )

        # Go through both the recorded nodes (end of the tape) and the nodes to
        # edit, check they match, check the recorded nones don't have
        # multiplicities different than 1 already, then update the multiplicity
        for index, ((recorded_node, recorded_multiplicity),
                    (node)) in enumerate(zip(self.recorded_nodes[-len(nodes):],
                                             nodes)):
            if recorded_node != node:
                raise ValueError(
                    f"'nodes' list contains a different node at index {index} "
                    f"than the one recorded in the tape."
                )
            if recorded_multiplicity != one():
                raise ValueError(
                    f"The recorded node in the tape corresponding to the one "
                    f"at index {index} in the 'nodes' list already has "
                    f"multiplicity different from one."
                )
            self.recorded_nodes[-len(nodes) + index][1] = multiplicity
