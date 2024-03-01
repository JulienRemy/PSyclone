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
    Literal,
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
    The recorded nodes can be boolean Literal, Reference or Operation nodes.
    Based on static arrays storing a single type of data rather than a LIFO \
    stack. \
    Provides methods to create the PSyIR assignments for recording and \
    restoring operations.

    :param name: name of the tape (after a prefix).
    :type object: str
    :param datatype: datatype of the elements of the control tape.
    :type datatype: :py:class:`psyclone.psyir.symbols.ScalarType`
    :param is_dynamic_array: whether to make the Fortran array dynamic \
                             (allocatable) or not. Optional, defaults to False.
    :type is_dynamic_array: Optional[bool]

    :raises TypeError: if name is of the wrong type.
    :raises TypeError: if datatype is of the wrong type.
    :raises TypeError: if datatype's intrinsic is not BOOLEAN.
    :raises TypeError: if is_dynamic_array is of the wrong type.
    """

    _node_types = (Reference, Operation, Literal)
    _tape_prefix = "ctrl_tape_"

    def __init__(self, name, datatype, is_dynamic_array = False):
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

        super().__init__(name, datatype, is_dynamic_array)

    def record(self, node, do_loop = False):
        """Add the boolean reference or operation result as last element of \
        the tape and return the Assignment node to record it to the tape.

        :param node: node whose boolean value should be recorded.
        :type reference: Union[:py:class:`psyclone.psyir.nodes.Reference`,
                               :py:class:`psyclone.psyir.nodes.Operation`,
                               :py:class:`psyclone.psyir.nodes.Literal`]
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if the intrinsic of node's datatype is not the \
                           same as the intrinsic of the control tape's \
                           elements datatype.
        :raises TypeError: if do_loop is of the wrong type.
        :raises NotImplementedError: if the reference's datatype is ArrayType.

        :return: an Assignment node for recording.
        :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type "
                f"'Reference', 'Operation' or 'Literal' but found "
                f"'{type(node).__name__}'."
            )

        if isinstance(node, (Reference, Literal)):
            if not isinstance(node.datatype, ScalarType):
                raise TypeError(
                    f"'node' argument should have datatype "
                    f"'ScalarType' but found "
                    f"'{type(node.datatype).__name__}'."
                )

            if node.datatype.intrinsic is not ScalarType.Intrinsic.BOOLEAN:
                raise TypeError(
                    f"'node' argument should have datatype "
                    f"'ScalarType' with intrinsic BOOLEAN "
                    f" but found '{node.datatype.intrinsic}'."
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
                BinaryOperation.Operator.EQV,
                BinaryOperation.Operator.NEQV,
            ):
                raise TypeError(
                    f"'node' argument should have a logical operator "
                    f"if it is of type Operation but found "
                    f"'{node.operator}'."
                )
        
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        control_tape_ref = super().record(node, do_loop)

        return Assignment.create(control_tape_ref, node.copy())

    def restore(self, node, do_loop = False):
        """Restore the boolean reference or operation result  from the tape.

        :param node: node whose value should be restored from the control tape.
        :type node: :py:class:`psyclone.psyir.symbols.DataSymbol`
        :param do_loop: whether currently transforming a do loop. \
                        Optional, defaults to False.
        :type do_loop: Optional[bool]

        :raises TypeError: if node is of the wrong type.
        :raises TypeError: if do_loop is of the wrong type.

        :return: a reference to the element of the control tape.
        :rtype: :py:class:`psyclone.psyir.nodes.ArrayReference`
        """
        if not isinstance(node, self._node_types):
            raise TypeError(
                f"'node' argument should be of type "
                f"'Reference', 'Operation' or 'Literal' but found "
                f"'{type(node).__name__}'."
            )
        
        if not isinstance(do_loop, bool):
            raise TypeError(
                f"'bool' argument should be of type "
                f"'bool' but found '{type(do_loop).__name__}'."
            )

        return super().restore(node, do_loop)
