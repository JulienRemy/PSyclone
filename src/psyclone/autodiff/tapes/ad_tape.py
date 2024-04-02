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

from psyclone.psyir.nodes import (
    ArrayReference,
    Literal,
    Node,
    Range,
    Operation,
    Reference,
    DataNode,
    IntrinsicCall,
    Call,
    Routine,
    Assignment,
    Loop,
)
from psyclone.psyir.symbols import (
    DataSymbol,
    INTEGER_TYPE,
    ScalarType,
    ArrayType,
    SymbolTable,
)
from psyclone.autodiff import (
    one,
    zero,
    increment,
    assign,
    mul,
    add,
    add_datanodes,
    substract_datanodes,
    multiply_datanodes,
)


class ADTape(object, metaclass=ABCMeta):
    """An abstract class for taping values in reverse-mode 
    automatic differentiation. 
    Based on static arrays storing a single type of data rather than a LIFO 
    stack. 

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
    """

    # pylint: disable=useless-object-inheritance

    # NOTE: these should be redefined by subclasses
    _node_types = (Node,)
    _tape_prefix = ""

    def __init__(self, name, datatype, is_dynamic_array=False):
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

        # Symbol of the do loop offset (iteration dependent)
        self.do_offset_symbol = DataSymbol(
            self._tape_prefix + name + "_do_offset", datatype=INTEGER_TYPE
        )

        # Symbol of the tape offset
        self.offset_symbol = DataSymbol(
            self._tape_prefix + name + "_offset", datatype=INTEGER_TYPE
        )

        # Mask not to count element lengths if already taken into account in
        # the offset
        self._offset_mask = []

        # Internal list of multiplicities for the recorded nodes
        self._multiplicities = []

        # Offset value to be used, after exiting a (nested) loop(s) body
        self.current_offset_value = zero()

        # Internal list of recorded nodes
        self._recorded_nodes = []

        # Internsal ists of recording/restorings to/from the the tape
        self._recordings = []
        self._restorings = []

        self._usefully_recorded_flags = []

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

    # @name.setter
    # def name(self, name):
    #    if not isinstance(name, str):
    #        raise TypeError(
    #            f"'name' argument should be of type "
    #            f"'str' but found '{type(name).__name__}'."
    #        )
    #    tape_type = ArrayType(self.datatype, [self.length(do_loop)])
    #    self.symbol = DataSymbol(self._tape_prefix + name, datatype=tape_type)

    def _check_internal_lists_are_all_same_length(self):
        for name, lst in (
            ("_multiplicities", self._multiplicities),
            ("_offset_mask", self._offset_mask),
            ("_recordings", self._recordings),
            ("_restorings", self._restorings),
            ("_usefully_recorded_flags", self._usefully_recorded_flags),
        ):
            if len(lst) != len(self._recorded_nodes):
                raise ValueError(
                    f"The length of the {name} list should "
                    f"always be equal to that of the _recorded_nodes "
                    f"list but found respectively {len(lst)} "
                    f"and {len(self._recorded_nodes)}"
                )

    def _only_keep_useful_items(self, internal_list):
        if internal_list not in (
            self._recorded_nodes,
            self._offset_mask,
            self._multiplicities,
            self._recordings,
            self._restorings,
        ):
            raise ValueError(
                "'internal_list' should be one of "
                "'_recorded_nodes', '_offset_mask', "
                "'_multiplicities', '_recordings' or '_restorings'"
                " but is not."
            )
        self._check_internal_lists_are_all_same_length()

        useful_items = [
            item
            for i, item in enumerate(internal_list)
            if self.usefully_recorded_flags[i] is True
        ]

        return useful_items

    @property
    def recorded_nodes(self):
        """List of recorded PSyIR nodes. 
        This is filtered according to the usefully_recorded_flags list. \
        For an unfiltered version, access _recorded_nodes instead.

        :return: list of nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        self._check_internal_lists_are_all_same_length()

        # return self._only_keep_useful_items(self._recorded_nodes)

        return self._recorded_nodes

    @property
    def offset_mask(self):
        """List of booleans, indicating whether the length of the associated \
        recorded node should be taken into account in the length or not. \
        True means not masked (use the element length), False means masked.
        This is filtered according to the usefully_recorded_flags list. \
        For an unfiltered version, access _offset_mask instead.

        :return: list of booleans.
        :rtype: List[bool]
        """
        self._check_internal_lists_are_all_same_length()

        # return self._only_keep_useful_items(self._offset_mask)

        return self._offset_mask

    @property
    def multiplicities(self):
        """List of datanodes, indicating the multiplicities of the recorded \
        nodes. Used for nodes that are recorded in a loop body, after the loop \
        is exited and for computing the total length of the static array.
        This is filtered according to the usefully_recorded_flags list. \
        For an unfiltered version, access _multiplicities instead.

        :return: list of multiplicities as PSyIR datanodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        self._check_internal_lists_are_all_same_length()

        # return self._only_keep_useful_items(self._multiplicities)

        return self._multiplicities

    @property
    def recordings(self):
        """List of recordings to the tape, found in the recording routine.
        This is filtered according to the usefully_recorded_flags list. \
        For an unfiltered version, access _recordings instead.

        :return: list of recordings as PSyIR nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        self._check_internal_lists_are_all_same_length()

        # return self._only_keep_useful_items(self._recordings)

        return self._recordings

    @property
    def restorings(self):
        """List of restorings from the tape, found in the restoring routine.
        This is filtered according to the usefully_recorded_flags list. \
        For an unfiltered version, access _restorings instead.

        :return: list of restorings as PSyIR nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        self._check_internal_lists_are_all_same_length()

        # return self._only_keep_useful_items(self._restorings)

        return self._restorings

    @property
    def usefully_recorded_flags(self):
        """List of booleans indicating whether the node was usefully recorded \
        or not. Used to filter the tape elements.
        
        :return: list of booleans indicating usefulness.
        :rtype: List[bool]"""

        return self._usefully_recorded_flags

    @property
    def offset_symbol(self):
        """Index offset PSyIR DataSymbol. 
        Used for indexing into the tape when loops are present, represents the \
        number of elements stored within previous loops.\
        Not the same as the do_loop_offset symbol which is iteration dependent \
        and used within loops bodies.

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
        Used for indexing into the tape when loops are present, represents the \
        number of elements stored within previous loops.\
        Not the same as the do_loop_offset symbol which is iteration dependent \
        and used within loops bodies.

        :return: fresh index offset Reference for indexing into the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.Reference`
        """
        return Reference(self.offset_symbol)

    @property
    def current_offset_value(self):
        """Current (ie. last) value of the offset symbol. This is incremented \
        when exiting a (nested) do loop body.

        :return: index offset value for indexing into the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        return self._current_offset_value

    @current_offset_value.setter
    def current_offset_value(self, current_offset_value):
        if not isinstance(current_offset_value, DataNode):
            raise TypeError(
                f"'current_offset_value' argument should be of type "
                f"'DataNode' but found "
                f"'{type(current_offset_value).__name__}'."
            )
        self._current_offset_value = current_offset_value

    @property
    def offset_assignment(self):
        """Returns an assignment of the current_offset_value to the offset \
        variable.

        :return: assignment.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        return assign(self.offset, self.current_offset_value)

    @property
    def do_offset_symbol(self):
        """Loop index offset PSyIR DataSymbol. 
        Used for indexing in the tape array in the presence of do loops, \
        depending on the value of the loop variables.

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

    def _tape_offsets(self, do_loop=False):
        """Returns a list of tape offsets to be used for indexing. \
        Within a do loop, adds the iteration dependent do_offset, after \
        loops adds the offset.

        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if do_loop is of the wrong type.

        :return: list of offsets to be added for indexing.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        offsets = []
        # Add the general offset if not null
        if self.current_offset_value != zero():
            offsets.append(self.offset)

        # Within a do loop body, also used the iteration dependent do_offset
        if do_loop:
            offsets.append(self.do_offset)

        return offsets

    def update_offset_and_mask(
        self, number_of_iterations, new_nodes, new_multiplicities
    ):
        """Given the total number of iterations of a loop, \
        some new nodes taped in it and their associated multiplicities \ 
        current_offset_value attribute and the offset_mask list.
        Returns the assignment of the new offset value to the tape offset.

        :param number_of_iterations: total number of iterations of the loop.
        :type number_of_iterations: :py:class:`psyclone.psyir.nodes.DataNode`
        :param new_nodes: list of nodes recorded in the loop.
        :type new_nodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        :param new_multiplicities: list of multiplicites of the nodes.
        :type new_multiplicities: List[\
                                      :py:class:`psyclone.psyir.nodes.DataNode`]

        :raises TypeError: if number_of_iterations is of the wrong type.
        :raises TypeError: if new_nodes is of the wrong type.
        :raises TypeError: if a node in new_nodes is of the wrong type.
        :raises TypeError: if new_multiplicities is of the wrong type.
        :raises TypeError: if an item in new_multiplicities is of the wrong \
                           type.
        :raises ValueError: if new_nodes and new_multiplicities are not of the \
                            same lengths.

        :return: assignment of the new offset value to the tape offset.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        if not isinstance(number_of_iterations, DataNode):
            raise TypeError(
                f"'number_of_iterations' argument should be of type "
                f"'DataNode' but found "
                f"'{type(number_of_iterations).__name__}'."
            )
        if not isinstance(new_nodes, list):
            raise TypeError(
                f"'new_nodes' argument should be of type "
                f"'list' but found '{type(new_nodes).__name__}'."
            )
        for new_node in new_nodes:
            if not isinstance(new_node, self._node_types):
                raise TypeError(
                    f"'new_nodes' argument should be a list of nodes of types "
                    f"among {self.node_type_names} but found an element of "
                    f"type '{type(new_node).__name__}'."
                )
        if not isinstance(new_multiplicities, list):
            raise TypeError(
                f"'new_multiplicities' argument should be of type "
                f"'list' but found '{type(new_multiplicities).__name__}'."
            )
        for new_multiplicity in new_multiplicities:
            if not isinstance(new_multiplicity, DataNode):
                raise TypeError(
                    f"'new_multiplicities' argument should be a list of nodes "
                    f"of type 'DataNode' but found an element of "
                    f"type '{type(new_multiplicity).__name__}'."
                )
        if len(new_nodes) != 0 and (
            self._recorded_nodes[-len(new_nodes) :] != new_nodes
        ):
            raise ValueError(
                "'new_nodes' argument should match the last "
                "recorded nodes but does not."
            )

        # Compute the length of all new nodes, with their associated
        # multiplicities
        new_lengths = self.length_of_nodes_with_multiplicities(
            new_nodes, new_multiplicities
        )

        # Multiply with the number of iterations of the loop and add to the
        # previous offset value
        length_increment = mul(number_of_iterations, new_lengths)
        self.current_offset_value = add(
            self.current_offset_value, length_increment
        )

        # Set the new nodes to be masked and their multiplicities to be
        # the number of iterations of the loop times their multiplicity in the
        # loop
        for i in range(len(new_nodes)):
            self.offset_mask[-len(new_nodes) + i] = False
            self.multiplicities[-len(new_nodes) + i] = mul(
                number_of_iterations, new_multiplicities[i]
            )

        # Return the assignement to the offset
        return assign(self.offset, self.current_offset_value)

    def _list_of_lengths_of_nodes(self, nodes):
        """Returns a list of lengths of the nodes.

        :param nodes: list of nodes.
        :type nodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :raises TypeError: if nodes is of the wrong type.
        :raises TypeError: if a node in nodes is of the wrong type.

        :return: list of lengths.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if not isinstance(nodes, list):
            if not isinstance(nodes, list):
                raise TypeError(
                    f"'nodes' argument should be of type "
                    f"'list' but found '{type(nodes).__name__}'."
                )
        for node in nodes:
            if not isinstance(node, self._node_types):
                raise TypeError(
                    f"'nodes' argument should be a list of nodes of types "
                    f"among {self.node_type_names} but found an element of "
                    f"type '{type(node).__name__}'."
                )

        lengths = []
        for node in nodes:
            # Special case for split reversal schedule tape extension,
            # the nodes taped inside the called routine are replaced with a
            # None item in the tape, with associated multiplicity equal to
            # the called subroutine tape length.
            if node is None:
                lengths.append(one())
            elif isinstance(node.datatype, ScalarType):
                lengths.append(one())
            else:
                lengths.append(self._array_size(node))

        return lengths

    def length_of_nodes(self, nodes):
        """Return the total length of the nodes.

        :param nodes: list of nodes.
        :type nodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :return: total length of the nodes.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        return add_datanodes(self._list_of_lengths_of_nodes(nodes))

    def _list_of_lengths_of_nodes_with_multiplicities(
        self, nodes, multiplicities
    ):
        """Returns a list of lengths of the nodes times their respective \
        multiplicities.

        :param nodes: list of nodes.
        :type nodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        :param nodes: list of multiplicities.
        :type nodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :raises TypeError: if nodes is of the wrong type.
        :raises TypeError: if a node in nodes is of the wrong type.
        :raises TypeError: if multiplicities is of the wrong type.
        :raises TypeError: if an item in multiplicities is of the wrong type.
        :raises ValueError: if nodes and multiplicities are of different \
                            lengths.

        :return: list of lengths.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if not isinstance(nodes, list):
            if not isinstance(nodes, list):
                raise TypeError(
                    f"'nodes' argument should be of type "
                    f"'list' but found '{type(nodes).__name__}'."
                )
        for node in nodes:
            if not isinstance(node, self._node_types):
                raise TypeError(
                    f"'nodes' argument should be a list of nodes of types "
                    f"among {self.node_type_names} but found an element of "
                    f"type '{type(node).__name__}'."
                )
        if not isinstance(multiplicities, list):
            if not isinstance(multiplicities, list):
                raise TypeError(
                    f"'multiplicities' argument should be of type "
                    f"'list' but found '{type(multiplicities).__name__}'."
                )
        for multiplicity in multiplicities:
            if not isinstance(multiplicity, DataNode):
                raise TypeError(
                    f"'multiplicities' argument should be a list of items of "
                    f"type 'DataNode' but found an element of "
                    f"type '{type(multiplicity).__name__}'."
                )
        if len(nodes) != len(multiplicities):
            raise ValueError(
                f"'nodes' and 'multiplicities' arguments should be lists of "
                f"same lengths but found {len(nodes)} and "
                f"{len(multiplicities)}."
            )

        lengths = self._list_of_lengths_of_nodes(nodes)
        mult_lengths = []
        for length, multiplicity in zip(lengths, multiplicities):
            mult_lengths.append(multiply_datanodes([length, multiplicity]))

        return mult_lengths

    def length_of_nodes_with_multiplicities(self, nodes, multiplicities):
        """Return the total length of the nodes times their respective \
        multiplicities.

        :param nodes: list of nodes.
        :type nodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        :param nodes: list of multiplicities.
        :type nodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

        :return: total length of the nodes times multiplicities.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        return add_datanodes(
            self._list_of_lengths_of_nodes_with_multiplicities(
                nodes, multiplicities
            )
        )

    @property
    def total_length(self):
        """Return the total length of the nodes in the tape, times their \
        respective multiplicities. This ignores the mask and is used to get \
        the length of the tape to use eg. in the tape declaration.

        :return: total length of all nodes times their multiplicities.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        return self.length_of_nodes_with_multiplicities(
            self.recorded_nodes, self.multiplicities
        )

    def length(self, do_loop=False):
        """Length of the tape (Fortran) array, which is the sum of the sizes \
        of its elements plus the offsets.
        If currently in a do loop, adds the do loop offset based on the loop \
        variable.

        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if do_loop is of the wrong type.

        :return: length of the tape, as a Literal or sum (BinaryOperation).
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`, \
                      :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        lengths = []

        # Add the offsets if necessary
        lengths.extend(self._tape_offsets(do_loop))

        unmasked_nodes = []
        unmasked_multiplicities = []
        for node, mult, mask in zip(
            self.recorded_nodes, self.multiplicities, self.offset_mask
        ):
            if mask:
                unmasked_nodes.append(node)
                unmasked_multiplicities.append(mult)

        lengths.extend(
            self._list_of_lengths_of_nodes_with_multiplicities(
                unmasked_nodes, unmasked_multiplicities
            )
        )

        return add_datanodes(lengths).copy()

    def first_index_of_last_element(self, do_loop=False):
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

        # Add the offsets if necessary
        lengths.extend(self._tape_offsets(do_loop))

        unmasked_nodes = []
        unmasked_multiplicities = []
        for node, mult, mask in zip(
            self.recorded_nodes[:-1],
            self.multiplicities[:-1],
            self.offset_mask[:-1],
        ):
            if mask:
                unmasked_nodes.append(node)
                unmasked_multiplicities.append(mult)

        lengths.extend(
            self._list_of_lengths_of_nodes_with_multiplicities(
                unmasked_nodes, unmasked_multiplicities
            )
        )

        lengths.append(one())

        return add_datanodes(lengths).copy()

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
            raise TypeError(
                f"'array' argument should be of type 'Reference' "
                f"but found '{type(array).__name__}'."
            )
        if not isinstance(array.datatype, ArrayType):
            raise ValueError(
                f"'array' argument should be of datatype "
                f"'ArrayType' but found "
                f"'{type(array.datatype).__name__}'."
            )

        # NEW_ISSUE: SIZE has optional second argument, this should be fixed
        # in FortranWriter & co.
        # return BinaryOperation.create(BinaryOperation.Operator.SIZE,
        #                               array.copy(),
        #                               None)

        dimensions = self._array_dimensions(array)

        return multiply_datanodes(dimensions)

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
            raise TypeError(
                f"'array' argument should be of type 'Reference' "
                f"but found '{type(array).__name__}'."
            )
        if not isinstance(array.datatype, ArrayType):
            raise ValueError(
                f"'array' argument should be of datatype "
                f"'ArrayType' but found "
                f"'{type(array.datatype).__name__}'."
            )

        # Go through the shape attribute of the ArrayType
        dimensions = []
        for dim, shape in enumerate(array.datatype.shape):
            # For array bounds, compute upper + 1 - lower
            if isinstance(shape, ArrayType.ArrayBounds):
                plus = [shape.upper, one()]
                minus = [shape.lower]
                dimensions.append(substract_datanodes(plus, minus))
            else:
                # For others, compute SIZE(array, dim)
                size_operation = IntrinsicCall.create(
                    IntrinsicCall.Intrinsic.SIZE,
                    [array.copy(), Literal(str(dim), INTEGER_TYPE)],
                )
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

        if isinstance(self._recorded_nodes[-1], IntrinsicCall):
            if self._recorded_nodes[-1].children[0] != node:
                raise ValueError(
                    f"node argument named {node.name} was not "
                    f"stored as last element of the value_tape."
                )
        else:
            if self._recorded_nodes[-1] != node:
                raise ValueError(
                    f"node argument named {node.name} was not "
                    f"stored as last element of the value_tape."
                )

    def record(self, node, do_loop=False):
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

        self._recorded_nodes.append(node)
        self._offset_mask.append(True)
        self._multiplicities.append(one())
        self._usefully_recorded_flags.append(True)
        self._recordings.append(None)
        self._restorings.append(None)

        # If static array, reshape to take the new length into account
        if not self.is_dynamic_array:
            self.reshape()

        # Nodes of ScalarType correspond to one index of the tape
        if isinstance(node.datatype, ScalarType):
            # This is the Fortran index, starting at 1
            tape_ref = ArrayReference.create(
                self.symbol, [self.length(do_loop)]
            )
        # Nodes of ArrayType correspond to a range
        else:
            # If static array, reshape to take the new length into account
            tape_range = Range.create(
                self.first_index_of_last_element(do_loop), self.length(do_loop)
            )
            tape_ref = ArrayReference.create(self.symbol, [tape_range])

        self._recordings[-1] = tape_ref

        return tape_ref

    def restore(self, node, do_loop=False):
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
            tape_ref = ArrayReference.create(
                self.symbol, [self.length(do_loop)]
            )

        # Nodes of ArrayType correspond to a range
        else:
            tape_range = Range.create(
                self.first_index_of_last_element(do_loop), self.length(do_loop)
            )
            tape_ref = ArrayReference.create(self.symbol, [tape_range])

        self._restorings[-1] = tape_ref

        return tape_ref

    def reshape(self, do_loop=False):
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

        value_tape_type = ArrayType(self.datatype, [self.total_length])
        self.symbol.datatype = value_tape_type

    def allocate(self, length):
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
        elif isinstance(length, Literal) and (
            not isinstance(length.datatype, ScalarType)
            or length.datatype.intrinsic is not INTEGER_TYPE.intrinsic
        ):
            raise ValueError(
                f"'length' argument is a 'Literal' but either its "
                f"datatype is not 'ScalarType' or it is not an "
                f"integer. Found {length}."
            )

        return IntrinsicCall.create(
            IntrinsicCall.Intrinsic.ALLOCATE,
            [ArrayReference.create(self.symbol, [length])],
        )

    def deallocate(self):
        """Generates a PSyIR IntrinsicCall ALLOCATE node for the tape.

        :return: the DEALLOCATE statement as a PSyIR IntrinsicCall node.
        :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
        """
        return IntrinsicCall.create(
            IntrinsicCall.Intrinsic.DEALLOCATE, [Reference(self.symbol)]
        )

    def _substitute_non_argument_references_in_length(self, tape, call):
        """Given a tape returned by a called subroutine (call being its primal \
        call), substitutes all the references in its length which are not \
        arguments in the routine in which call is found, so that the tape \
        might be declared as a static array.

        :param tape: tape whose length to substitute in.
        :type tape: :py:class:`psyclone.autodiff.tapes.ADTape`
        :param call: primal call to the subroutine which returned the tape.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: _type_
        """

        # Length expression to substitute in
        length = tape.total_length

        # Get the calling routine
        # All references in the tape length should ultimately be arguments of it
        # or literal values
        calling_routine = call.ancestor(Routine)

        # Arguments names (the symbols being different between the length
        # expression and the the symbol table of the calling routine)
        calling_routine_arguments_names = [
            sym.name for sym in calling_routine.symbol_table.argument_list
        ]

        # Sort through the references in length to get the ones that are not
        # arguments of the calling routine and should be substituted
        all_refs_in_length = length.walk(Reference)
        non_argument_refs_in_length = [
            ref
            for ref in all_refs_in_length
            if ref.name not in calling_routine_arguments_names
        ]
        non_argument_refs_names_in_length = [
            ref.name for ref in non_argument_refs_in_length
        ]

        # Get the assignments in the calling routine, the position of the call,
        # sort to get the assignments before the call, make sure there is only
        # one per symbol and build a dictionary {name: value}
        all_assignments = calling_routine.walk(Assignment)
        call_pos = call.abs_position
        non_arg_ref_values = dict()
        for non_arg_ref_name in non_argument_refs_names_in_length:
            values = [
                assignment.rhs
                for assignment in all_assignments
                if (
                    assignment.lhs.name == non_arg_ref_name
                    and assignment.abs_position < call_pos
                )
            ]
            if len(values) == 0:
                raise NotImplementedError(
                    f"Could not find a value to substitute for {non_arg_ref_name}."
                )
            if len(values) > 1:
                raise NotImplementedError(
                    "Substitution only supports references which are on the LHS of a single assignement, for now."
                )
            non_arg_ref_values[non_arg_ref_name] = values[0]

        # Substitute until all references are arguments
        while len(non_argument_refs_in_length) != 0:

            # Some might have been detached, ignore them
            for ref in non_argument_refs_in_length:
                if ref.parent is None:
                    continue

                # Get the associated value in the calling routine and
                # substitute
                value = non_arg_ref_values[ref.name]
                ref.replace_with(value.copy())

                # If substituting by an operation, a call, etc., the new value
                # might itself contain non argument references, so "recurse"
                # and get the associated values of these new ones
                if not isinstance(value, Reference):
                    for new_ref in value.walk(Reference):
                        if new_ref.name not in calling_routine_arguments_names:
                            values = [
                                assignment.rhs
                                for assignment in all_assignments
                                if (
                                    assignment.lhs.name == new_ref.name
                                    and assignment.abs_position < call_pos
                                )
                            ]
                            if len(values) == 0:
                                raise NotImplementedError(
                                    f"Could not find a value to substitute for "
                                    f"{new_ref.name}."
                                )
                            if len(values) > 1:
                                raise NotImplementedError(
                                    "Substitution only supports references "
                                    "which are on the LHS of a single "
                                    "assignemnt, for now."
                                )
                            non_arg_ref_values[new_ref.name] = values[0]

            # Build the list again, loop
            non_argument_refs_in_length = [
                ref
                for ref in length.walk(Reference)
                if ref.name not in calling_routine_arguments_names
            ]

        return length

    def extend(self, tape, call):
        """Extends the tape with the recorded nodes of the 'tape' argument, \
        which must be of the same type.
        This is used in split reversal schedule mode.
        The recorded nodes of the 'tape' arguments are **not** actually added \
        to the tape being extended. The 'call' node is added instead, with \
        an associated multiplicity equal to the length of the new nodes.

        :param tape: tape to combine.
        :type tape: :py:class:`psyclone.autodiff.ADTape`, same as self.
        :param call: call to the routine that returned 'tape'.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :raises TypeError: if tape is of the wrong type.
        :raises TypeError: if the tape datatype is different.
        :raises TypeError: if call is of the wrong type.
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
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of type "
                f"'Call' but found "
                f"'{type(call).__name__}'."
            )

        # Substitute the references which are not arguments of the calling
        # routine so that the tape may be declared as a static array
        length = self._substitute_non_argument_references_in_length(tape, call)

        # Add None, unmasked, with the length of the new nodes/tape
        # as associated multiplicity
        self._recorded_nodes.append(None)
        self._offset_mask.append(True)
        self._multiplicities.append(length)

        # TODO: property?
        self._recordings.append(None)
        self._restorings.append(None)

        # If static array, reshape to take the new length into account
        if not self.is_dynamic_array:
            self.reshape()

    def extend_and_slice(self, tape, call, do_loop=False):
        """Extends the tape by the 'tape' argument and return \
        the ArrayReference corresponding to the correct slice.
        This is used in split reversal schedule mode.
        The recorded nodes of the 'tape' arguments are **not** actually added \
        to the tape being extended. The 'call' node is added instead, with \
        an associated multiplicity equal to the length of the new nodes.

        :param tape: tape to extend with.
        :type tape: :py:class:`psyclone.autodiff.tapes.ADTape`
        :param call: call to the routine that returned 'tape'.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]
        :param call: call to the routine that returned 'tape'.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

        :raises TypeError: if tape is not of the same type as self.
        :raises ValueError: if the datatype of tape is not the same as \
            the datatype of self. 
        :raises TypeError: if do_loop is of the wrong type.
        :raises TypeError: if call is of the wrong type.

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
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of type "
                f"'Call' but found "
                f"'{type(call).__name__}'."
            )
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )
        # First index of the slice corresponding to the "new" tape
        first_index = add_datanodes([self.length(do_loop), one()])
        # Extend the parent value_tape with the new value_tape
        self.extend(tape, call)
        # Last index of the slice
        last_index = self.length(do_loop)
        # Slice of the parent value_tape
        value_tape_range = Range.create(first_index, last_index)

        return ArrayReference.create(self.symbol, [value_tape_range])

    def simplify_length_expression_with_sympy(self, symbol_table):
        """Simplify the length expression of the Fortran tape using sympy.
        Takes a symbol table where all symbols in the length expression can be \
        found as argument, for the sympy reader. 
        The length is modified in the shape attribute of the tape datatype \
        itself.

        :param symbol_table: table where symbols in the length expression can \
                             be found.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :raises TypeError: if symbol_table is of the wrong type.
        """
        # Importing here to keep reliance on sympy entirely optional
        # pylint: disable=import-outside-toplevel
        from psyclone.psyir.backend.sympy_writer import SymPyWriter
        from psyclone.psyir.frontend.sympy_reader import SymPyReader
        from sympy import simplify

        if not isinstance(symbol_table, SymbolTable):
            raise TypeError(
                f"'symbol_table' argument should be of type "
                f"'SymbolTable' but found "
                f"'{type(symbol_table).__name__}'."
            )

        sympywriter = SymPyWriter()
        sympyreader = SymPyReader(sympywriter)
        # Transform the upper bound of the array shape
        fortran_expr = self.symbol.datatype.shape[0].upper
        sympy_expr = sympywriter(fortran_expr)
        new_sympy_expr = simplify(sympy_expr)
        length = sympyreader.psyir_from_expression(new_sympy_expr, symbol_table)

        # Edit the tape datatype accordingly
        self.symbol.datatype = ArrayType(
            self.symbol.datatype.datatype, [length]
        )

    def simplify_expression_with_sympy(self, expression, symbol_table):
        """Simplify the expression using sympy.
        Takes a symbol table where all symbols in the length expression can be \
        found as argument, for the sympy reader. 

        :param expression: PSyIR expression to simplify.
        :type expression: :py:class:`psyclone.psyir.nodes.DataNode`
        :param symbol_table: table where symbols in the length expression can \
                             be found.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`

        :raises TypeError: if expression is of the wrong type.
        :raises TypeError: if symbol_table is of the wrong type.

        :return: simplified expression, as a copy of the argument.
        :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
        """
        # Importing here to keep reliance on sympy entirely optional
        # pylint: disable=import-outside-toplevel
        from psyclone.psyir.backend.sympy_writer import SymPyWriter
        from psyclone.psyir.frontend.sympy_reader import SymPyReader
        from sympy import simplify

        if not isinstance(expression, DataNode):
            raise TypeError(
                f"'expression' argument should be of type "
                f"'DataNode' but found "
                f"'{type(expression).__name__}'."
            )
        if not isinstance(symbol_table, SymbolTable):
            raise TypeError(
                f"'symbol_table' argument should be of type "
                f"'SymbolTable' but found "
                f"'{type(symbol_table).__name__}'."
            )

        sympywriter = SymPyWriter()
        sympyreader = SymPyReader(sympywriter)

        simplified_expression = expression.copy()
        sympy_expr = sympywriter(simplified_expression)
        return sympyreader.psyir_from_expression(
            simplify(sympy_expr), symbol_table
        )

    def get_all_recorded_symbols_names(self):
        """Returns a list of unique symbols names that were recorded to the \
        tape.

        :return: list of unique symbols names in the tape.
        :rtype: List[str]
        """
        recorded_symbols = []
        for recorded_node in self._recorded_nodes:
            # Special case: None for tape extension in split mode
            if recorded_node is None:
                continue
            if recorded_node.symbol.name not in recorded_symbols:
                recorded_symbols.append(recorded_node.symbol.name)
        return recorded_symbols

    def get_recorded_symbols_to_restorings_map(self):
        """Returns a dictionnary with recorded symbol names as keys and lists \
        of restorings as values.

        :return: {symbol_name : [restoring, restoring, ...]}
        :rtype: Dict[str,
                     List[:py:class:`psyclone.psyir.nodes.Node`]]
        """
        self._check_internal_lists_are_all_same_length()

        symbols_map = dict()

        for recorded_node, restoring in zip(
            self._recorded_nodes, self._restorings
        ):
            if recorded_node is None:
                continue
            if recorded_node.symbol.name in symbols_map:
                symbols_map[recorded_node.symbol.name].append(restoring)
            else:
                symbols_map[recorded_node.symbol.name] = [restoring]

        for restorings in symbols_map.values():
            restorings.sort(key=(lambda x: x.abs_position))
        return symbols_map

    def get_all_recorded_symbols_reads_in_routine(self, routine):
        """Returns a dictionnary with recorded symbol names as keys and lists \
        of read References to these symbols in the routine argument as values.
        Looks in the rhs of assignments, the indices of array references on \
        the lhs of assignments and the call arguments.

        :param routine: routine to look in.
        :type routine: :py:class:`psyclone.psyir.nodes.Routine`

        :raises TypeError: if routine is of the wrong type.

        :return: {symbol_name : [read, read, ...]}
        :rtype: Dict[str,
                     List[:py:class:`psyclone.psyir.nodes.Reference`]]
        """
        if not isinstance(routine, Routine):
            raise TypeError(
                f"'routine' argument should be of type "
                f"'Routine' but found "
                f"'{type(routine).__name__}'."
            )
        recorded_symbols_names = self.get_all_recorded_symbols_names()
        recorded_symbols_reads = dict()

        # Look in the rhs of assignments, the arguments of calls, the indices
        # on the lhs of assignments, the bounds of loops
        all_assignments = routine.walk(Assignment)
        all_assignments_rhs = [assignment.rhs for assignment in all_assignments]
        all_assignments_lhs_indices = []
        for assignment in all_assignments:
            if isinstance(assignment.lhs, ArrayReference):
                all_assignments_lhs_indices.extend(assignment.lhs.indices)
        all_call_arguments = []
        for call in routine.walk(Call):
            all_call_arguments.extend(call.children)
        all_loop_bounds = []
        for loop in routine.walk(Loop):
            all_loop_bounds.extend(loop.children)

        # For all recorded symbols, get references to them and add to the dict
        for recorded_symbol_name in recorded_symbols_names:
            all_reads = []
            for expr in (
                all_assignments_rhs
                + all_assignments_lhs_indices
                + all_call_arguments
                + all_loop_bounds
            ):
                refs_to_this_symbol = [
                    ref
                    for ref in expr.walk(Reference)
                    if ref.symbol.name == recorded_symbol_name
                ]
                all_reads.extend(refs_to_this_symbol)
            recorded_symbols_reads[recorded_symbol_name] = all_reads

        return recorded_symbols_reads

    def update_usefully_recorded_flags(self, useful_restorings):
        """Update the usefully_recorded_flags list based on a list of useful 
        restorings.

        :param useful_restorings: list of useful restorings, corresponding to \
                                  nodes that should be kept unmasked.
        :type useful_restorings: List[:py:class:`psyclone.psyir.nodes.Node`]

        :raises TypeError: if useful_restorings is of the wrong type.
        :raises TypeError: if an element of useful_restorings is of the wrong \
                           type.
        :raises ValueError: if an element of useful_restorings is not found in \
                            the internal _restorings list.
        """
        if not isinstance(useful_restorings, list):
            raise TypeError(
                f"'useful_restorings' argument should be of type "
                f"'list' but found "
                f"'{type(useful_restorings).__name__}'."
            )
        for restoring in useful_restorings:
            if not isinstance(restoring, Node):
                raise TypeError(
                    f"'useful_restorings' argument should be a list of items "
                    f"of type 'Node' but found "
                    f"'{type(restoring).__name__}'."
                )
            if restoring not in self._restorings:
                raise ValueError(
                    f"The items of 'useful_restorings' should be "
                    f"restorings but found "
                    f"{restoring.debug_string()} which is not in "
                    f"_restorings."
                )

        self._check_internal_lists_are_all_same_length()

        for i, restoring in enumerate(self._restorings):
            if restoring in useful_restorings:
                self._usefully_recorded_flags[i] = True
            else:
                self._usefully_recorded_flags[i] = False

    def get_useless_recordings(self):
        """Get all useless recordings based on the usefully_recorded_flags list.
        Used to detach them from the AST in post-processing TBR analysis.

        :return: list of useless recording nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        return [
            recording
            for i, recording in enumerate(self._recordings)
            if self.usefully_recorded_flags[i] is False
        ]

    def get_useless_restorings(self):
        """Get all useless restorings based on the usefully_recorded_flags list.
        Used to detach them from the AST in post-processing TBR analysis.

        :return: list of useless restorings nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        return [
            restoring
            for i, restoring in enumerate(self._restorings)
            if self.usefully_recorded_flags[i] is False
        ]
