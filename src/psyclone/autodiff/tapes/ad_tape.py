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
)
from psyclone.psyir.symbols import (
    DataSymbol,
    INTEGER_TYPE,
    ScalarType,
    ArrayType,
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

        # List of multiplicities for the recorded nodes
        self._multiplicities = []

        # Offset value to be used, after exiting a (nested) loop(s) body
        self.current_offset_value = zero()

        # Internal list of recorded nodes
        self._recorded_nodes = []

        # TODO: describe
        self._recordings = []
        self._restorings = []

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

    @property
    def recorded_nodes(self):
        """List of recorded PSyIR nodes.

        :return: list of nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        if len(self._offset_mask) != len(self._recorded_nodes):
            raise ValueError(
                f"The length of the offset_mask list should "
                f"always be equal to that of the recorded_nodes "
                f"list but found {len(self._offset_mask)} "
                f"and {len(self._recorded_nodes)}"
            )
        if len(self._multiplicities) != len(self._recorded_nodes):
            raise ValueError(
                f"The length of the multiplicities list should "
                f"always be equal to that of the recorded_nodes "
                f"list but found {len(self._multiplicities)} "
                f"and {len(self._recorded_nodes)}"
            )
        # if len(self._recordings) != len(self._recorded_nodes):
        #     raise ValueError(
        #         f"The length of the recordings list should "
        #         f"always be equal to that of the recorded_nodes "
        #         f"list but found {len(self._recordings)} "
        #         f"and {len(self._recorded_nodes)}"
        #     )
        # if len(self._restorings) != len(self._recorded_nodes):
        #     raise ValueError(
        #         f"The length of the restorings list should "
        #         f"always be equal to that of the recorded_nodes "
        #         f"list but found {len(self._restorings)} "
        #         f"and {len(self._recorded_nodes)}"
        #     )

        return self._recorded_nodes

    @property
    def offset_mask(self):
        """List of booleans, indicating whether the length of the associated \
        recorded node should be taken into account in the length or not. \
        True means not masked (use the element length), False means masked.

        :return: list of booleans.
        :rtype: List[bool]
        """
        if len(self._offset_mask) != len(self._recorded_nodes):
            raise ValueError(
                f"The length of the offset_mask list should "
                f"always be equal to that of the recorded_nodes "
                f"list but found {len(self._offset_mask)} "
                f"and {len(self._recorded_nodes)}"
            )
        return self._offset_mask

    @property
    def multiplicities(self):
        """List of datanodes, indicating the multiplicities of the recorded \
        nodes. Used for nodes that are recorded in a loop body, after the loop \
        is exited and for computing the total length of the static array.

        :return: list of multiplicities as PSyIR datanodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.DataNode`]
        """
        if len(self._multiplicities) != len(self._recorded_nodes):
            raise ValueError(
                f"The length of the multiplicities list should "
                f"always be equal to that of the recorded_nodes "
                f"list but found {len(self._multiplicities)} "
                f"and {len(self._recorded_nodes)}"
            )
        return self._multiplicities

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

    def update_offset_and_mask(self, number_of_iterations, new_nodes):
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
            # if new_node.datatype is not self.datatype or (
            #     new_node.datatype is ArrayType
            #     and new_node.datatype.datatype is not self.datatype
            # ):
            #     raise TypeError(
            #         f"'new_nodes' argument should be a list of nodes of "
            #         f"datatype {self.datatype} but found an element of "
            #         f"datatype '{new_node.datatype}'."
            #     )
        if len(new_nodes) != 0 and (
            self._recorded_nodes[-len(new_nodes) :] != new_nodes
        ):
            raise ValueError(
                "'new_nodes' argument should match the last "
                "recorded nodes but does not."
            )

        # lengths = []
        # for new_node in new_nodes:
        #     if isinstance(new_node.datatype, ScalarType):
        #         lengths.append(one())
        #     else:
        #         lengths.append(self._array_size(new_node))

        new_lengths = self.length_of_nodes(new_nodes)

        length_increment = mul(number_of_iterations, new_lengths)
        self.current_offset_value = add(
            self.current_offset_value, length_increment
        )

        for i in range(len(new_nodes)):
            self.offset_mask[-len(new_nodes) + i] = False
            self.multiplicities[-len(new_nodes) + i] = number_of_iterations

        return assign(self.offset, self.current_offset_value)

    def _list_of_lengths_of_nodes(self, nodes):
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
            # if node.datatype is not self.datatype or (
            #     node.datatype is ArrayType
            #     and node.datatype.datatype is not self.datatype
            # ):
            #     raise TypeError(
            #         f"'nodes' argument should be a list of nodes of "
            #         f"datatype {self.datatype} but found an element of "
            #         f"datatype '{node.datatype}'."
            #     )

        lengths = []
        for node in nodes:
            if isinstance(node.datatype, ScalarType):
                lengths.append(one())
            else:
                lengths.append(self._array_size(node))

        return lengths

    def length_of_nodes(self, nodes):
        return add_datanodes(self._list_of_lengths_of_nodes(nodes))

    def _list_of_lengths_of_nodes_with_multiplicities(
        self, nodes, multiplicities
    ):
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
            # if node.datatype is not self.datatype or (
            #     node.datatype is ArrayType
            #     and node.datatype.datatype is not self.datatype
            # ):
            #     raise TypeError(
            #         f"'nodes' argument should be a list of nodes of "
            #         f"datatype {self.datatype} but found an element of "
            #         f"datatype '{node.datatype}'."
            # )
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
        return add_datanodes(
            self._list_of_lengths_of_nodes_with_multiplicities(
                nodes, multiplicities
            )
        )

    @property
    def total_length(self):
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
        # for node, mask in zip(self.recorded_nodes, self.offset_mask):
        #     if mask:
        #         if isinstance(node.datatype, ScalarType):
        #             lengths.append(one())
        #         else:
        #             lengths.append(self._array_size(node))

        # return add_datanodes(lengths).copy()

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

        # # Add the offsets if necessary
        # lengths.extend(self._tape_offsets(do_loop))

        # for node, mask in zip(self.recorded_nodes[:-1], self.offset_mask[:-1]):
        #     if mask:
        #         if isinstance(node.datatype, ScalarType):
        #             lengths.append(one())
        #         else:
        #             lengths.append(self._array_size(node))

        # lengths.append(one())

        # return add_datanodes(lengths)

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
        # if node.datatype is not self.datatype or (
        #     node.datatype is ArrayType
        #     and node.datatype.datatype is not self.datatype
        # ):
        #     raise TypeError(
        #         f"'node' argument should be of "
        #         f"datatype {self.datatype} but found "
        #         f"datatype '{node.datatype}'."
        #     )
        # FIXME: dirty hack to tape integers
        # if self.recorded_nodes[-1] != node:
        #     raise ValueError(
        #         f"node argument named {node.name} was not "
        #         f"stored as last element of the value_tape."
        #     )

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
        # if node.datatype is not self.datatype or (
        #     node.datatype is ArrayType
        #     and node.datatype.datatype is not self.datatype
        # ):
        #     raise TypeError(
        #         f"'node' argument should be of "
        #         f"datatype {self.datatype} but found "
        #         f"datatype '{node.datatype}'."
        #     )

        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        self._recorded_nodes.append(node)
        self._offset_mask.append(True)
        self._multiplicities.append(one())

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

        self._recordings.append(tape_ref)

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
        # if node.datatype is not self.datatype or (
        #     node.datatype is ArrayType
        #     and node.datatype.datatype is not self.datatype
        # ):
        #     raise TypeError(
        #         f"'node' argument should be of "
        #         f"datatype {self.datatype} but found "
        #         f"datatype '{node.datatype}'."
        #     )
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

        self._restorings.append(tape_ref)

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
        calling_routine = call.ancestor(Routine)

        length = tape.total_length

        print(
            f"Adding tape {tape.name} to calling routine {calling_routine.name}"
        )

        calling_routine_arguments_names = [
            sym.name for sym in calling_routine.symbol_table.argument_list
        ]
        all_refs_in_length = length.walk(Reference)
        non_argument_refs_in_length = [
            ref
            for ref in all_refs_in_length
            if ref.name not in calling_routine_arguments_names
        ]
        non_argument_refs_names_in_length = [
            ref.name for ref in non_argument_refs_in_length
        ]

        print(f"Non arg refs are {non_argument_refs_names_in_length}")

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
        # all_assignments_to_non_arg_refs = [assignment for assignment in all_assignments if assignment.lhs.name in non_argument_refs_names_in_length]

        # all_lhs_names = {assignment.lhs.name for assignment in all_assignments if assignment.abs_position < call_pos}

        # last_values_before_call = dict()
        # for name in all_lhs_names:
        #     values = [
        #         assignment.rhs
        #         for assignment in all_assignments
        #         if assignment.lhs.name == name
        #     ]
        #     last_values_before_call[name] = values[-1]

        while len(non_argument_refs_in_length) != 0:
            refs_being_substituted = []
            new_non_arg_refs = []
            for ref in non_argument_refs_in_length:
                if ref.parent is None:
                    continue

                value = non_arg_ref_values[ref.name]
                print(
                    f"Trying to replace {ref.name} with {value.debug_string()}"
                )
                ref.replace_with(value.copy())
                refs_being_substituted.append(ref)
                if not isinstance(value, Reference):
                    for new_ref in value.walk(Reference):
                        if new_ref.name not in calling_routine_arguments_names:
                            new_non_arg_refs.append(new_ref)
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
                                    f"Could not find a value to substitute for {new_ref.name}."
                                )
                            if len(values) > 1:
                                raise NotImplementedError(
                                    "Substitution only supports references which are on the LHS of a single assignement, for now."
                                )
                            non_arg_ref_values[new_ref.name] = values[0]
                            # raise NotImplementedError("Recursive substitution is not implemented yet.")
            for ref in refs_being_substituted:
                non_argument_refs_in_length.remove(ref)
            for ref in new_non_arg_refs:
                non_argument_refs_in_length.append(ref)

        print(f"Substitution yielded {length.debug_string()}")

        return length

    def extend(self, tape, call):
        """Extends the tape with the recorded nodes of the 'tape' argument, \
        which must be of the same type.

        :param tape: tape to combine.
        :type tape: :py:class:`psyclone.autodiff.ADTape`, same as self.
        :param call: call to the routine that returned 'tape'.
        :type call: :py:class:`psyclone.psyir.nodes.Call`

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
        if not isinstance(call, Call):
            raise TypeError(
                f"'call' argument should be of type "
                f"'Call' but found "
                f"'{type(call).__name__}'."
            )

        # length = self._substitute_non_argument_references_in_length(tape, call)

        # self._recorded_nodes.extend(call)
        # self._offset_mask.extend(True)
        # self._multiplicities.extend(length)

        self._recorded_nodes.extend(tape.recorded_nodes)
        self._offset_mask.extend(tape.offset_mask)
        self._multiplicities.extend(tape.multiplicities)
        # TODO: property
        self._recordings.extend(tape._recordings)
        self._restorings.extend(tape._restorings)

        # If static array, reshape to take the new length into account
        if not self.is_dynamic_array:
            self.reshape()

    def extend_and_slice(self, tape, call, do_loop=False):
        """Extends the tape by the 'tape' argument and return \
        the ArrayReference corresponding to the correct slice.

        :param tape: tape to extend with.
        :type tape: :py:class:`psyclone.autodiff.tapes.ADTape`
        :param call: call to the routine that returned 'tape'.
        :type call: :py:class:`psyclone.psyir.nodes.Call`
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

        :param symbol_table: table where symbols in the length expression can \
                             be found.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        # Importing here to keep reliance on sympy entirely optional
        #pylint: disable=import-outside-toplevel
        from psyclone.psyir.backend.sympy_writer import SymPyWriter
        from psyclone.psyir.frontend.sympy_reader import SymPyReader
        from sympy import simplify

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

    def simplify_assignments_with_sympy(self, symbol_table):
        """Simplify the expressions in the recording and restoring assignments \
        to and from the tape..
        Takes a symbol table where all symbols in the length expression can be \
        found as argument, for the sympy reader. 

        :param symbol_table: table where symbols in the length expression can \
                             be found.
        :type symbol_table: :py:class:`psyclone.psyir.symbols.SymbolTable`
        """
        # Importing here to keep reliance on sympy entirely optional
        #pylint: disable=import-outside-toplevel
        from psyclone.psyir.backend.sympy_writer import SymPyWriter
        from psyclone.psyir.frontend.sympy_reader import SymPyReader
        from sympy import simplify

        sympywriter = SymPyWriter()
        sympyreader = SymPyReader(sympywriter)
        for assignment in self._recordings + self._restorings:
            simplified_operations = []
            for operation in assignment.walk(Operation):
                # If the operation's ancestor was already simplified, there is
                # no need to deal with it
                if (
                    operation.ancestor(Operation, include_self=False)
                    not in simplified_operations
                ):
                    simplified_operations.append(operation)

                    sympy_expr = sympywriter(operation)
                    new_expr = sympyreader.psyir_from_expression(
                        simplify(sympy_expr), symbol_table
                    )
                    operation.replace_with(new_expr)
                else:
                    simplified_operations.append(operation)

    # def change_last_nodes_multiplicity(self, nodes, multiplicity):
    #     """Change the multiplicity of the last recorded nodes, which should \
    #     match the ones in 'nodes'.
    #     Used after transforming a loop so that the tape offset is correct \
    #     based on the number of iterations that were performed.

    #     :param nodes: list of nodes whose multiplicities should be changed.
    #     :type nodes: List[:py:class:`psyclone.psyir.nodes.Node`]
    #     :param multiplicity: new multiplicity to be used, as a PSyIR node.
    #     :type multiplicity: Union[:py:class:`psyclone.psyir.nodes.Literal`,\
    #                            :py:class:`psyclone.psyir.nodes.Reference`,\
    #                            :py:class:`psyclone.psyir.nodes.BinaryOperation`]

    #     :raises TypeError: if nodes is of the wrong type.
    #     :raises TypeError: if an element of nodes is of the wrong type.
    #     :raises TypeError: if multiplicity is of the wrong type.
    #     :raises ValueError: if a recorded node doesn't match the corresponding \
    #                         one in 'nodes'.
    #     :raises ValueError: if the recorded node already has multiplicity \
    #                         different from 1.
    #     """
    #     if not isinstance(nodes, list):
    #         raise TypeError(
    #             f"'nodes' argument should be of type "
    #             f"'list[Node]' but found '{type(nodes).__name__}'."
    #         )
    #     for node in nodes:
    #         if not isinstance(node, Node):
    #             raise TypeError(
    #                 f"'nodes' argument should be of type "
    #                 f"'list[Node]' but found an element of type "
    #                 f"'{type(node).__name__}'."
    #             )
    #     if not isinstance(multiplicity, (Literal, Reference, BinaryOperation)):
    #         raise TypeError(
    #             f"'multiplicity' argument should be of type "
    #             f"'Literal', 'Reference' or 'Binary Operation' but found "
    #             f"'{type(multiplicity).__name__}'."
    #         )

    #     # Go through both the recorded nodes (end of the tape) and the nodes to
    #     # edit, check they match, check the recorded nones don't have
    #     # multiplicities different than 1 already, then update the multiplicity
    #     for index, (
    #         (recorded_node, recorded_multiplicity),
    #         (node),
    #     ) in enumerate(zip(self.recorded_nodes[-len(nodes) :], nodes)):
    #         if recorded_node != node:
    #             raise ValueError(
    #                 f"'nodes' list contains a different node at index {index} "
    #                 f"than the one recorded in the tape."
    #             )
    #         if recorded_multiplicity != one():
    #             raise ValueError(
    #                 f"The recorded node in the tape corresponding to the one "
    #                 f"at index {index} in the 'nodes' list already has "
    #                 f"multiplicity different from one."
    #             )
    #         self.recorded_nodes[-len(nodes) + index][1] = multiplicity
