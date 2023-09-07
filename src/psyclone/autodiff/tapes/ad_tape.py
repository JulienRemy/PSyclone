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

from psyclone.psyir.nodes import (ArrayReference, Literal, Node, Range,
                                  BinaryOperation, Reference, DataNode)
from psyclone.psyir.symbols import (DataSymbol, INTEGER_TYPE, ScalarType,
                                    ArrayType)


class ADTape(object, metaclass=ABCMeta):
    """An abstract class for taping values in reverse-mode 
    automatic differentiation. 
    Based on static arrays storing a single type of data rather than a LIFO 
    stack. 

    :param name: name of the value_tape (after a prefix).
    :type object: str
    :param datatype: datatype of the elements of the value_tape.
    :type datatype: :py:class:`psyclone.psyir.symbols.ScalarType`

    :raises TypeError: if name is of the wrong type.
    :raises TypeError: if datatype is of the wrong type.
    """
    # pylint: disable=useless-object-inheritance

    _node_types = (Node,)

    _tape_prefix = "tape_"

    def __init__(self, name, datatype):
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

        # PSyIR datatype of the elements in this tape
        self.datatype = datatype

        # Type of the value_tape, shape will be modified as needed
        tape_type = ArrayType(datatype, [0])

        # Symbol of the value_tape
        self.symbol = DataSymbol(self._tape_prefix + name, datatype=tape_type)

        # Internal list of recorded nodes
        self._recorded_nodes = []

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

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError(
                f"'name' argument should be of type "
                f"'str' but found '{type(name).__name__}'."
            )
        tape_type = ArrayType(self.datatype, [self.length])
        self.symbol = DataSymbol(self._tape_prefix + name, datatype=tape_type)

    @property
    def recorded_nodes(self):
        """List of recorded PSyIR nodes.

        :return: list of nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        return self._recorded_nodes

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

        result = Literal("0", INTEGER_TYPE)
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

        result = Literal("0", INTEGER_TYPE)
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

        result = Literal("1", INTEGER_TYPE)
        if len(other_datanodes) != 0:
            result = other_datanodes[0]
            if len(other_datanodes) > 1:
                for datanode in other_datanodes[1:]:
                    result = BinaryOperation.create(
                                BinaryOperation.Operator.MUL,
                                result.copy(),
                                datanode.copy())

        return result

    @property
    def length(self):
        """Length of the tape (Fortran) array, which is the sum of the sizes \
        of its elements.

        :return: length of the tape, as a Literal or sum (BinaryOperation).
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`, \
                      :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """

        lengths = []
        for node in self.recorded_nodes:
            if isinstance(node.datatype, ScalarType):
                lengths.append(Literal("1", INTEGER_TYPE))
            else:
                lengths.append(self._array_size(node))

        return self._add_datanodes(lengths)

    @property
    def first_index_of_last_element(self):
        """Gives the first index of the last element that was recorded.

        :return: Literal or BinaryOperation giving the index.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`,
                      :py:class:`psyclone.psyir.nodes.BinaryOperation`]
        """
        lengths = []
        for node in self.recorded_nodes[:-1]:
            if isinstance(node.datatype, ScalarType):
                lengths.append(Literal("1", INTEGER_TYPE))
            else:
                lengths.append(self._array_size(node))

        lengths.append(Literal("1", INTEGER_TYPE))

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
                plus = [shape.upper, Literal("1", INTEGER_TYPE)]
                minus = [shape.lower]
                dimensions.append(self._substract_datanodes(plus, minus))
            else:
                # For others, compute SIZE(array, dim)
                size_operation = BinaryOperation.create(
                                    BinaryOperation.Operator.SIZE,
                                    array.copy(),
                                    Literal(str(dim), INTEGER_TYPE))
                dimensions.append(size_operation)

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
        if self.recorded_nodes[-1] != node:
            raise ValueError(
                f"node argument named {node.name} was not "
                f"stored as last element of the value_tape."
            )

    def record(self, node):
        """Add the node as last element of the tape and return the \
        ArrayReference node of the tape.

        :param node: node whose prevalue should be recorded.
        :type node: :py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if the intrinsic of node's datatype is not the \
                           same as the intrinsic of the value_tape's \
                           elements datatype.

        :return: the array node to the last element of the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )

        # Nodes of ScalarType correspond to one index of the tape
        if isinstance(node.datatype, ScalarType):
            self.recorded_nodes.append(node)
            self.reshape()
            # This is the Fortran index, starting at 1
            tape_ref = ArrayReference.create(self.symbol, [self.length])

        # Nodes of ArrayType correspond to a range
        else:
            self.recorded_nodes.append(node)
            self.reshape()
            tape_range = Range.create(self.first_index_of_last_element,
                                      self.length)
            tape_ref = ArrayReference.create(self.symbol, [tape_range])

        return tape_ref

    def restore(self, node):
        """Check that node is the last element of the tape and return an \
        ArrayReference to it in the tape.

        :param node: node restore.
        :type node: :py:class:`psyclone.psyir.nodes.Node`

        :raises TypeError: if node is of the wrong type.

        :return: an ArrayReference node to the last element of the tape.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type among "
                f"{self.node_type_names} but found "
                f"'{type(node).__name__}'."
            )

        self._has_last(node)

        # Nodes of ScalarType correspond to one index of the tape
        if isinstance(node.datatype, ScalarType):
            # This is the Fortran index, starting at 1
            tape_ref = ArrayReference.create(self.symbol, [self.length])

        # Nodes of ArrayType correspond to a range
        else:
            tape_range = Range.create(self.first_index_of_last_element, 
                                      self.length)
            tape_ref = ArrayReference.create(self.symbol, [tape_range])

        return tape_ref

    def reshape(self):
        """Change the static length of the tape array in its datatype.
        """
        value_tape_type = ArrayType(self.datatype, [self.length])
        self.symbol.datatype = value_tape_type

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

        self.reshape()

    def extend_and_slice(self, tape):
        """Extends the tape by the 'tape' argument and return \
        the ArrayReference corresponding to the correct slice.

        :param tape: tape to extend with.
        :type tape: :py:class:`psyclone.autodiff.tapes.ADTape`

        :raises TypeError: if tape is not of the same type as self.
        :raises ValueError: if the datatype of tape is not the same as \
            the datatype of self. 

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
        # First index of the slice corresponding to the "new" tape
        first_index = self._add_datanodes([self.length,
                                           Literal("1", INTEGER_TYPE)])
        # Extend the parent value_tape with the new value_tape
        self.extend(tape)
        # Last index of the slice
        last_index = self.length
        # Slice of the parent value_tape
        value_tape_range = Range.create(first_index, last_index)

        return ArrayReference.create(self.symbol, [value_tape_range])
