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

"""This module provides a class for reverse-mode automatic differentiation 
"taping" (storing and recovering) of control flow boolean values.
"""

from psyclone.psyir.nodes import (
    Assignment,
    Reference,
    Operation,
    UnaryOperation,
    BinaryOperation,
)
from psyclone.psyir.symbols import ScalarType

from psyclone.autodiff.tapes import ADTape

###################################
###################################
# TODO: this should also accept Call nodes with return_symbol of boolean type
###################################
###################################


class ADControlTape(ADTape):
    """A class for recording and recovering values of control flow conditions \
    in reverse-mode automatic differentiation. \
    The recorded nodes are either boolean Reference nodes \
    or boolean Operation ones.
    Based on static arrays storing a single type of data rather than a LIFO \
    stack. \
    Provides methods to create the PSyIR assignments for recording and \
    restoring operations.

    :param name: name of the tape (after a prefix).
    :type object: str
    :param datatype: datatype of the elements of the value_tape.
    :type datatype: :py:class:`psyclone.psyir.symbols.ScalarType`

    :raises TypeError: if name is of the wrong type.
    :raises TypeError: if datatype is of the wrong type.
    :raises TypeError: if datatype's intrinsic is not BOOLEAN.
    """

    _node_types = (Reference, Operation)
    _tape_prefix = "ctrl_tape_"

    def __init__(self, name, datatype):
        # if not isinstance(name, str):
        #    raise TypeError(f"'name' argument should be of type "
        #                    f"'str' but found '{type(name).__name__}'.")
        if not isinstance(datatype, ScalarType):
            raise TypeError(
                f"'datatype' argument should be of type "
                f"'ScalarType' but found "
                f"'{type(datatype).__name__}'."
            )
        if datatype.intrinsic is not ScalarType.Intrinsic.BOOLEAN:
            raise TypeError(
                f"Only BOOLEAN values can be taped "
                f"for now but found DataType "
                f"'{datatype}' instead."
            )

        control_tape_name = self._tape_prefix + name

        super().__init__(control_tape_name, datatype)

    def record(self, node):
        """Add the boolean reference or operation result as last element of \
        the tape and return the Assignment node to record it to the tape.

        :param node: node whose prevalue should be recorded.
        :type reference: :py:class:`psyclone.psyir.nodes.Reference`

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if the intrinsic of node's datatype is not the \
                           same as the intrinsic of the value_tape's \
                           elements datatype.
        :raises NotImplementedError: if the reference's datatype is ArrayType.

        :return: an Assignment node for recording.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        if not isinstance(node, (Reference, Operation)):
            raise TypeError(
                f"'node' argument should be of type "
                f"'Reference' or 'Operation' but found "
                f"'{type(node).__name__}'."
            )

        if isinstance(node, Reference):
            if not isinstance(node.datatype, (ScalarType)):
                raise TypeError(
                    f"'node' argument should have datatype "
                    f"'ScalarType' if it is of type Reference but found "
                    f"'{type(node.datatype).__name__}'."
                )

            if node.datatype.intrinsic is not ScalarType.Intrinsic.BOOLEAN:
                raise TypeError(
                    f"'node' argument should have datatype "
                    f"'ScalarType' with intrinsic BOOLEAN "
                    f"if it is of type Reference but found "
                    f"'{node.datatype.intrinsic}'."
                )

        if isinstance(node, Operation):
            if node.operator not in (
                UnaryOperation.Operator.NOT,
                BinaryOperation.Operator.GE,
                BinaryOperation.Operator.GT,
                BinaryOperation.Operator.AND,
                BinaryOperation.Operator.OR,
                BinaryOperation.Operator.EQ,
                BinaryOperation.Operator.LE,
                BinaryOperation.Operator.LT,
                BinaryOperation.Operator.NE,
            ):
                raise TypeError(
                    f"'node' argument should have a logical operator "
                    f"if it is of type Operation but found "
                    f"'{node.operator}'."
                )

        value_tape_ref = super().record(node)

        return Assignment.create(value_tape_ref, node.copy())

    def restore(self, node):
        """Restore the boolean reference or operation result \
         and return the Assignment node to restore it from the tape.

        :param node: node whose prevalue should be restored from the value_tape.
        :type node: :py:class:`psyclone.psyir.symbols.DataSymbol`

        :raises TypeError: if node is of the wrong type.

        :return: an Assignment node for restoring the prevalue of the node \
                    from the last element of the value_tape.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        if not isinstance(node, (Reference, Operation)):
            raise TypeError(
                f"'node' argument should be of type "
                f"'Reference' or 'Operation' but found "
                f"'{type(node).__name__}'."
            )

        value_tape_ref = super().restore(node)

        return Assignment.create(node.copy(), value_tape_ref)
