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
automatic differentiation "taping" (storing and recovering) of different values."""

from abc import ABCMeta

from psyclone.psyir.nodes import ArrayReference, Literal, Node, Range
from psyclone.psyir.symbols import DataSymbol, INTEGER_TYPE, DataType, ArrayType


class ADTape(object, metaclass=ABCMeta):
    """An abstract class for taping values in reverse-mode 
    automatic differentiation. 
    Based on static arrays storing a single type of data rather than a LIFO 
    stack. 

    :param name: name of the value_tape (after a prefix).
    :type object: str
    :param datatype: datatype of the elements of the value_tape.
    :type datatype: :py:class:`psyclone.psyir.symbols.DataType`

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
        if not isinstance(datatype, DataType):
            raise TypeError(
                f"'datatype' argument should be of type "
                f"'DataType' but found "
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
        :rtype: :py:class:`psyclone.psyir.symbols.DataType`
        """
        return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        if not isinstance(datatype, DataType):
            raise TypeError(
                f"'datatype' argument should be of type "
                f"'ScalarType' or 'ArrayType' but found "
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

    @property
    def length(self):
        """Length of the tape ie. number of recorded nodes.

        :return: length of the tape.
        :rtype: int
        """
        return len(self.recorded_nodes)

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

        self.recorded_nodes.append(node)
        self.reshape()

        # This is the Fortran index, starting at 1
        tape_index_literal = Literal(str(self.length), INTEGER_TYPE)
        tape_ref = ArrayReference.create(self.symbol, [tape_index_literal])

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

        # This is the Fortran index, starting at 1
        tape_index_literal = Literal(str(self.length), INTEGER_TYPE)
        tape_ref = ArrayReference.create(self.symbol, [tape_index_literal])

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
        first_index = self.length + 1
        # Extend the parent value_tape with the new value_tape
        self.extend(tape)
        # Last index of the slice
        last_index = self.length
        # as literals
        first_literal = Literal(str(first_index), INTEGER_TYPE)
        last_literal = Literal(str(last_index), INTEGER_TYPE)
        # Slice of the parent value_tape
        value_tape_range = Range.create(first_literal, last_literal)

        return ArrayReference.create(self.symbol, [value_tape_range])
