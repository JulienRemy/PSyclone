# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2017-2023, Science and Technology Facilities Council.
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
# Authors R. W. Ford, A. R. Porter and S. Siso, STFC Daresbury Lab
#         I. Kavcic, Met Office
#         J. Henrichs, Bureau of Meteorology
# Modified: J. Remy, Université Grenoble Alpes, Inria
# -----------------------------------------------------------------------------

''' This module contains the implementation of the Operation class and its
sub-classes.'''

from abc import ABCMeta, abstractmethod
from enum import Enum
import re

from psyclone.errors import GenerationError
from psyclone.psyir.nodes import DataNode, Literal

from psyclone.psyir.symbols.datatypes import (INTEGER_TYPE, BOOLEAN_TYPE,
                                              REAL_TYPE, ScalarType, ArrayType)


class Operation(DataNode, metaclass=ABCMeta):
    '''
    Abstract base class for PSyIR nodes representing operators.

    :param operator: the operator used in the operation.
    :type operator: :py:class:`psyclone.psyir.nodes.UnaryOperation.Operator` \
        or :py:class:`psyclone.psyir.nodes.BinaryOperation.Operator` or \
        :py:class:`psyclone.psyir.nodes.NaryOperation.Operator`
    :param parent: the parent node of this Operation in the PSyIR.
    :type parent: :py:class:`psyclone.psyir.nodes.Node`

    :raises TypeError: if the supplied operator is not an instance of \
                       self.Operator.

    '''
    # Must be overridden in sub-class to hold an Enumeration of the Operators
    # that it can represent.
    Operator = object
    _non_elemental_ops = []
    # Textual description of the node.
    _text_name = "Operation"
    _colour = "blue"

    def __init__(self, operator, parent=None):
        super().__init__(parent=parent)

        if not isinstance(operator, self.Operator):
            raise TypeError(
                f"{type(self).__name__} operator argument must be of type "
                f"{type(self).__name__}.Operator but found "
                f"{type(operator).__name__}.")
        self._operator = operator
        self._argument_names = []

    def append_named_arg(self, name, arg):
        '''Append a named argument to this operation.

           :param name: the argument name.
           :type name: Optional[str]
           :param arg: the argument expression.
           :type arg: :py:class:`psyclone.psyir.nodes.DataNode`

           :raises ValueError: if the name argument is already used \
               for an existing argument.

        '''
        self._validate_name(name)
        if name is not None:
            for check_name in self.argument_names:
                if check_name and check_name.lower() == name.lower():
                    raise ValueError(
                        f"The value of the name argument ({name}) in "
                        f"'append_named_arg' in the 'Operator' node is "
                        f"already used for a named argument.")
        self._argument_names.append((id(arg), name))
        self.children.append(arg)

    def insert_named_arg(self, name, arg, index):
        '''Insert a named argument to the operation.

           :param name: the argument name.
           :type name: Optional[str]
           :param arg: the argument expression.
           :type arg: :py:class:`psyclone.psyir.nodes.DataNode`
           :param int index: where in the argument list to insert the \
               named argument.

           :raises ValueError: if the name argument is already used \
               for an existing argument.
           :raises TypeError: if the index argument is the wrong type.

        '''
        self._validate_name(name)
        if name is not None:
            for check_name in self.argument_names:
                if check_name and check_name.lower() == name.lower():
                    raise ValueError(
                        f"The value of the name argument ({name}) in "
                        f"'insert_named_arg' in the 'Operator' node is "
                        f"already used for a named argument.")
        if not isinstance(index, int):
            raise TypeError(
                f"The 'index' argument in 'insert_named_arg' in the "
                f"'Operator' node should be an int but found "
                f"{type(index).__name__}.")
        self._argument_names.insert(index, (id(arg), name))
        self.children.insert(index, arg)

    def replace_named_arg(self, existing_name, arg):
        '''Replace one named argument node with another node keeping the
        same name.

           :param str existing_name: the argument name.
           :param arg: the argument expression.
           :type arg: :py:class:`psyclone.psyir.nodes.DataNode`

           :raises TypeError: if the name argument is the wrong type.
           :raises ValueError: if the name argument is already used \
               for an existing argument.
           :raises TypeError: if the index argument is the wrong type.

        '''
        if not isinstance(existing_name, str):
            raise TypeError(
                f"The 'name' argument in 'replace_named_arg' in the "
                f"'Operation' node should be a string, but found "
                f"{type(existing_name).__name__}.")
        index = 0
        for _, name in self._argument_names:
            if name is not None and name.lower() == existing_name:
                break
            index += 1
        else:
            raise ValueError(
                f"The value of the existing_name argument ({existing_name}) "
                f"in 'replace_named_arg' in the 'Operation' node was not found"
                f" in the existing arguments.")
        self.children[index] = arg
        self._argument_names[index] = (id(arg), existing_name)

    @staticmethod
    def _validate_name(name):
        '''Utility method that checks that the supplied name has a valid
        format.

        :param name: the name to check.
        :type name: Optional[str]

        :raises TypeError: if the name is not a string or None.
        :raises ValueError: if this is not a valid name.

        '''
        if name is None:
            return
        if not isinstance(name, str):
            raise TypeError(
                f"A name should be a string or None, but found "
                f"{type(name).__name__}.")
        if not re.match(r'^[a-zA-Z]\w*$', name):
            raise ValueError(
                f"Invalid name '{name}' found.")

    def __eq__(self, other):
        '''Checks whether two Operations are equal. Operations are equal
        if they are the same type, have the same operator and if the inherited
        equality is True.

        :param object other: the object to check equality to.

        :returns: whether other is equal to self.
        :rtype: bool
        '''
        is_eq = super().__eq__(other)
        is_eq = is_eq and self.operator == other.operator
        is_eq = is_eq and self.argument_names == other.argument_names

        return is_eq

    @property
    def operator(self):
        '''
        Return the operator.

        :returns: Enumerated type capturing the operator.
        :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation.Operator` or \
                :py:class:`psyclone.psyir.nodes.BinaryOperation.Operator` or \
                :py:class:`psyclone.psyir.nodes.NaryOperation.Operator`

        '''
        return self._operator

    def node_str(self, colour=True):
        '''
        Construct a text representation of this node, optionally with control
        codes for coloured display in a suitable terminal.

        :param bool colour: whether or not to include colour control codes.

        :returns: description of this PSyIR node.
        :rtype: str
        '''
        return self.coloured_name(colour) + \
            "[operator:'" + self._operator.name + "']"

    @property
    def argument_names(self):
        '''
        :returns: a list containing the names of named arguments. If the \
            entry is None then the argument is a positional argument.
        :rtype: List[Optional[str]]
        '''
        self._reconcile()
        return [entry[1] for entry in self._argument_names]

    def _reconcile(self):
        '''update the _argument_names values in case child arguments have been
        removed, or added.

        '''
        new_argument_names = []
        for child in self.children:
            for arg in self._argument_names:
                if id(child) == arg[0]:
                    new_argument_names.append(arg)
                    break
            else:
                new_argument_names.append((id(child), None))
        self._argument_names = new_argument_names

    @property
    def is_elemental(self):
        '''
        :returns: whether this operation is elemental (provided with an input \
            array it will apply the operation individually to each of the \
            array elements and return an array with the results).
        :rtype: bool
        '''
        return self.operator not in self._non_elemental_ops

    def __str__(self):
        result = f"{self.node_str(False)}\n"
        for idx, entity in enumerate(self._children):
            if self.argument_names[idx]:
                result += f"{self.argument_names[idx]}={str(entity)}\n"
            else:
                result += f"{str(entity)}\n"

        # Delete last line break
        if result[-1] == "\n":
            result = result[:-1]
        return result

    def copy(self):
        '''Return a copy of this node. This is a bespoke implementation for
        Operation nodes that ensures that any internal id's are
        consistent before and after copying.

        :returns: a copy of this node and its children.
        :rtype: :py:class:`psyclone.psyir.node.Node`

        '''
        # ensure _argument_names is consistent with actual arguments
        # before copying.
        # pylint: disable=protected-access
        self._reconcile()
        # copy
        new_copy = super().copy()
        # Fix invalid id's in _argument_names after copying.
        new_list = []
        for idx, child in enumerate(new_copy.children):
            my_tuple = (id(child), new_copy._argument_names[idx][1])
            new_list.append(my_tuple)
        new_copy._argument_names = new_list

        return new_copy

    @property
    @abstractmethod
    def datatype(self):
        """
        :return: PSyIR datatype of the result of this operation.
        :rtype: :py:class:`psyclone.psyir.symbols.DataType`
        """


class UnaryOperation(Operation):
    '''
    Node representing a UnaryOperation expression. As such it has one operand
    as child 0, and an attribute with the operator type.
    '''
    # Textual description of the node.
    _children_valid_format = "DataNode"
    _text_name = "UnaryOperation"

    Operator = Enum('Operator', [
        # Arithmetic Operators
        'MINUS', 'PLUS', 'SQRT', 'EXP', 'LOG', 'LOG10',
        # Logical Operators
        'NOT',
        # Trigonometric Operators
        'COS', 'SIN', 'TAN', 'ACOS', 'ASIN', 'ATAN',
        # Other Maths Operators
        'ABS', 'CEIL', 'FLOOR', 'TRANSPOSE',
        # Casting Operators
        'REAL', 'INT', 'NINT'
        ])

    _non_elemental_ops = []

    @property
    def datatype(self):
        """
        :return: PSyIR datatype of the result of this operation.
        :rtype: :py:class:`psyclone.psyir.symbols.DataType`
        """
        operand = self.children[0]

        # REAL [, INTEGER for INT] => default INTEGER
        if self.operator in (UnaryOperation.Operator.CEIL,
                             UnaryOperation.Operator.FLOOR,
                             UnaryOperation.Operator.NINT,
                             UnaryOperation.Operator.INT):
            scalar_type = INTEGER_TYPE

        # {INTEGER, REAL} => default REAL
        elif self.operator is UnaryOperation.Operator.REAL:
            scalar_type = REAL_TYPE

        else:
            scalar_type = ScalarType(operand.datatype.intrinsic,
                                     operand.datatype.precision)

        if isinstance(operand.datatype, ScalarType):
            return scalar_type

        # TRANSPOSE : type_kind_ixj => type_kind_jxi
        shape = operand.datatype.shape.copy()

        if self.operator is UnaryOperation.Operator.TRANSPOSE:
            if len(shape) != 2:
                raise TypeError(f"Only matrices can be transposed but found "
                                f"an array with shape attribute of length "
                                f"{len(shape)}.")
            shape.reverse()

        return ArrayType(scalar_type, shape)


    @staticmethod
    def _validate_child(position, child):
        '''
        :param int position: the position to be validated.
        :param child: a child to be validated.
        :type child: :py:class:`psyclone.psyir.nodes.Node`

        :return: whether the given child and position are valid for this node.
        :rtype: bool

        '''
        return position == 0 and isinstance(child, DataNode)

    @staticmethod
    def create(operator, operand):
        '''Create a UnaryOperation instance given an operator and operand.

        :param operator: the specified operator.
        :type operator: \
            :py:class:`psyclone.psyir.nodes.UnaryOperation.Operator`
        :param operand: the PSyIR node that oper operates on, or a tuple \
            containing the name of the argument and the PSyIR node.
        :type operand: Union[:py:class:`psyclone.psyir.nodes.Node` | \
            Tuple[str, :py:class:``psyclone.psyir.nodes.Node``]]

        :returns: a UnaryOperation instance.
        :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation`

        :raises GenerationError: if the arguments to the create method \
            are not of the expected type.

        '''
        if not isinstance(operator, Enum) or \
           operator not in UnaryOperation.Operator:
            raise GenerationError(
                f"operator argument in create method of UnaryOperation class "
                f"should be a PSyIR UnaryOperation Operator but found "
                f"'{type(operator).__name__}'.")

        unary_op = UnaryOperation(operator)
        name = None
        if isinstance(operand, tuple):
            if not len(operand) == 2:
                raise GenerationError(
                    f"If the argument in the create method of "
                    f"UnaryOperation class is a tuple, it's length "
                    f"should be 2, but it is {len(operand)}.")
            if not isinstance(operand[0], str):
                raise GenerationError(
                    f"If the argument in the create method of "
                    f"UnaryOperation class is a tuple, its first "
                    f"argument should be a str, but found "
                    f"{type(operand[0]).__name__}.")
            name, operand = operand

        unary_op.append_named_arg(name, operand)
        return unary_op


class BinaryOperation(Operation):
    '''
    Node representing a BinaryOperation expression. As such it has two operands
    as children 0 and 1, and an attribute with the operator type.

    '''
    Operator = Enum('Operator', [
        # Arithmetic Operators. ('REM' is remainder AKA 'MOD' in Fortran.)
        'ADD', 'SUB', 'MUL', 'DIV', 'REM', 'POW',
        # Relational Operators
        'EQ', 'NE', 'GT', 'LT', 'GE', 'LE',
        # Logical Operators
        'AND', 'OR', 'EQV', 'NEQV',
        # Other Maths Operators
        'SIGN', 'MIN', 'MAX',
        # Casting operators
        'REAL', 'INT', 'CAST',
        # Array Query Operators
        'SIZE', 'LBOUND', 'UBOUND',
        # Matrix and Vector Operators
        'MATMUL', 'DOT_PRODUCT'
        ])
    _non_elemental_ops = [Operator.MATMUL, Operator.SIZE,
                          Operator.LBOUND, Operator.UBOUND,
                          Operator.DOT_PRODUCT]
    '''Arithmetic operators:

    .. function:: POW(arg0, arg1) -> type(arg0)

       :returns: `arg0` raised to the power of `arg1`.

    Array query operators:

    .. function:: SIZE(array, index) -> int

       :returns: the size of the `index` dimension of `array`.

    .. function:: LBOUND(array, index) -> int

       :returns: the value of the lower bound of the `index` dimension of \
                 `array`.

    .. function:: UBOUND(array, index) -> int

       :returns: the value of the upper bound of the `index` dimension of \
                 `array`.

    Casting Operators:

    .. function:: REAL(arg0, precision)

       :returns: `arg0` converted to a floating point number of the \
                 specified precision.

    .. function:: INT(arg0, precision)

       :returns: `arg0` converted to an integer number of the specified \
                  precision.

    .. function:: CAST(arg0, mold)

       :returns: `arg0` with the same bitwise representation but interpreted \
                 with the same type as the specified `mold` argument.

    Matrix and Vector Operators:

    .. function:: MATMUL(array1, array2) -> array

       :returns: the result of performing a matrix multiply with a \
                 matrix (`array1`) and a matrix or a vector
                 (`array2`).

    .. note:: `array1` must be a 2D array. `array2` may be a 2D array
        or a 1D array (vector). The size of the second dimension of
        `array1` must be the same as the first dimension of
        `array1`. If `array2` is 2D then the resultant array will be
        2D with the size of its first dimension being the same as the
        first dimension of `array1` and the size of its second
        dimension being the same as second dimension of `array2`. If
        `array2` is a vector then the resultant array is a vector with
        the its size being the size of the first dimension of
        `array1`.

    .. note:: The type of data in `array1` and `array2` must be the
        same and the resultant data will also have the same
        type. Currently only REAL data is supported.

    .. function:: DOT_PRODUCT(vector1, vector2) -> scalar

       :returns: the result of performing a dot product on two equal \
           sized vectors.

    .. note:: The type of data in `vector1` and `vector2` must be the
        same and the resultant data will also have the same
        type. Currently only REAL data is supported.

    '''
    # Textual description of the node.
    _children_valid_format = "DataNode, DataNode"
    _text_name = "BinaryOperation"

    @staticmethod
    def _promote_scalar_datatypes(first, second):
        """Promote the scalar datatypes passed as arguments.
        Implemented from \
        https://docs.oracle.com/cd/E19957-01/805-4939/z400073a2265/index.html
        for the PSyIR datatypes currently implemented.
        NOTE: treats undefined as single precision, 
        which may not always be true (compiler flags)

        :param first: first scalar datatype to promote.
        :type first: :py:class:`psyclone.psyir.symbols.ScalarType`
        :param second: second scalar datatype to promote.
        :type second: :py:class:`psyclone.psyir.symbols.ScalarType`

        :raises TypeError: if first is of the wrong type.
        :raises TypeError: if second is of the wrong type.
        :raises NotImplementedError: if first is not in the listed datatypes.
        :raises NotImplementedError: if second is not in the listed datatypes.

        :return: the promoted scalar datatype.
        :rtype: :py:class:`psyclone.psyir.symbols.ScalarType`
        """
        if not isinstance(first, ScalarType):
            raise TypeError(f"'first' argument should be of type 'ScalarType' "
                            f"but found '{type(first).__name__}'.")
        if not isinstance(second, ScalarType):
            raise TypeError(f"'second' argument should be of type 'ScalarType' "
                            f"but found '{type(second).__name__}'.")

        # How it should (?) be, from
        # https://docs.oracle.com/cd/E19957-01/805-4939/z400073a2265/index.html
        # NOTE: treating undefined as single precision,
        # which may not always be true (compiler flags)
        # pylint: disable=pointless-string-statement
        """
        datatypes_ranks = {(ScalarType.Intrinsic.BOOLEAN,
                            ScalarType.Precision.SINGLE)      : 3,
                           (ScalarType.Intrinsic.BOOLEAN, 4)  : 3,
                           (ScalarType.Intrinsic.BOOLEAN,
                            ScalarType.Precision.UNDEFINED)   : 3,
                        
                           (ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.SINGLE)      : 5,
                           (ScalarType.Intrinsic.INTEGER, 4)  : 5,
                           (ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.UNDEFINED)   : 5,

                           (ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.DOUBLE)      : 6,
                           (ScalarType.Intrinsic.INTEGER, 8)  : 6,

                           (ScalarType.Intrinsic.BOOLEAN,
                            ScalarType.Precision.DOUBLE)      : 6,
                           (ScalarType.Intrinsic.BOOLEAN, 8)  : 6,

                           (ScalarType.Intrinsic.REAL,
                            ScalarType.Precision.SINGLE)      : 6,
                           (ScalarType.Intrinsic.REAL, 4)     : 6,
                           (ScalarType.Intrinsic.REAL,
                            ScalarType.Precision.UNDEFINED)   : 6,

                           (ScalarType.Intrinsic.REAL,
                            ScalarType.Precision.DOUBLE)      : 7,
                           (ScalarType.Intrinsic.REAL, 8)     : 7}"""

        # gfortran actually does this
        datatypes_ranks = {(ScalarType.Intrinsic.BOOLEAN,
                            ScalarType.Precision.SINGLE)      : 3,
                           (ScalarType.Intrinsic.BOOLEAN, 4)  : 3,
                           (ScalarType.Intrinsic.BOOLEAN,
                            ScalarType.Precision.UNDEFINED)   : 3,
                           (ScalarType.Intrinsic.BOOLEAN,
                            ScalarType.Precision.DOUBLE)      : 3,
                           #(ScalarType.Intrinsic.BOOLEAN, 8)  : 3,

                           (ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.SINGLE)      : 5,
                           (ScalarType.Intrinsic.INTEGER, 4)  : 5,
                           (ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.UNDEFINED)   : 5,
                           (ScalarType.Intrinsic.INTEGER,
                            ScalarType.Precision.DOUBLE)      : 5,

                           (ScalarType.Intrinsic.INTEGER, 8)  : 6,
                           (ScalarType.Intrinsic.BOOLEAN, 8)  : 6,

                           (ScalarType.Intrinsic.REAL,
                            ScalarType.Precision.SINGLE)      : 7,
                           (ScalarType.Intrinsic.REAL, 4)     : 7,
                           (ScalarType.Intrinsic.REAL,
                            ScalarType.Precision.UNDEFINED)   : 7,

                           (ScalarType.Intrinsic.REAL,
                            ScalarType.Precision.DOUBLE)      : 8,
                           (ScalarType.Intrinsic.REAL, 8)     : 8}

        first_key = (first.intrinsic, first.precision)
        second_key = (second.intrinsic, second.precision)
        if first_key not in datatypes_ranks:
            raise NotImplementedError(f"'first' argument is '{first}', whose "
                                      f"promotion rules are not implemented "
                                      f"yet.")
        if second_key not in datatypes_ranks:
            raise NotImplementedError(f"'second' argument is '{second}', whose "
                                      f"promotion rules are not implemented "
                                      f"yet.")

        if datatypes_ranks[first_key] > datatypes_ranks[second_key]:
            return first

        return second

    @property
    def datatype(self):
        """PSyIR datatype of the result of this operation.

        :raises NotImplementedError: if the operator is CAST and the second \
            argument is an array.
        :raises NotImplementedError: if the operator is REAL or INT and the \
            second argument is not a literal.
        :raises NotImplementedError: if the broadcasting rules failed.

        :return: PSyIR datatype of the result of this operation.
        :rtype: :py:class:`psyclone.psyir.symbols.DataType`
        """
        # SIZE, LBOUND, UNBOUND : (array, dim) => default INTEGER
        if self.operator in (BinaryOperation.Operator.SIZE,
                             BinaryOperation.Operator.LBOUND,
                             BinaryOperation.Operator.UBOUND):
            return INTEGER_TYPE

        # CAST : (arg, scalar_type_kind) => scalar_type_kind
        if self.operator is BinaryOperation.Operator.CAST:
            if isinstance(self.children[1].datatype, ScalarType):
                return self.children[1].datatype
            else:
                raise NotImplementedError("Arrays as second arguments of "
                                          "CAST/TRANSFER binary operations "
                                          "are not implemented yet.")

        # MATMUL : (array_ixj, array_jxk) => array_ixk
        if self.operator is BinaryOperation.Operator.MATMUL:
            first = ScalarType(self.children[0].datatype.intrinsic,
                               self.children[0].datatype.precision)
            second = ScalarType(self.children[1].datatype.intrinsic,
                                self.children[1].datatype.precision)

            if (first.intrinsic is ScalarType.Intrinsic.BOOLEAN) \
                and (second.intrinsic is ScalarType.Intrinsic.BOOLEAN):
                scalar_type = ScalarType(ScalarType.Intrinsic.BOOLEAN,
                              ScalarType.Precision.UNDEFINED)
            else:
                scalar_type = self._promote_scalar_datatypes(first, second)
            # Second arg is a matrix
            if len(self.children[1].datatype.shape) == 2:
                return ArrayType(scalar_type,
                                 [self.children[0].datatype.shape[0],
                                  self.children[1].datatype.shape[1]])

            # Second arg is a vector
            return ArrayType(scalar_type, [self.children[0].datatype.shape[0]])

        # DOT_PRODUCT : (vector_i, vector_i) => scalar
        if self.operator is BinaryOperation.Operator.DOT_PRODUCT:
            first = ScalarType(self.children[0].datatype.intrinsic,
                               self.children[0].datatype.precision)
            second = ScalarType(self.children[1].datatype.intrinsic,
                                self.children[1].datatype.precision)
            return self._promote_scalar_datatypes(first, second)

        # REAL, INT : (arg, k) => REAL*k/INT*k
        if self.operator in (BinaryOperation.Operator.REAL,
                             BinaryOperation.Operator.INT):
            if not isinstance(self.children[1], Literal):
                raise NotImplementedError(f"Second argument of operation "
                                          f"{self.view()} is not a Literal, "
                                          f"this is not supported yet.")

            if self.operator is BinaryOperation.Operator.REAL:
                intrinsic = ScalarType.Intrinsic.REAL
            else:
                intrinsic = ScalarType.Intrinsic.INTEGER

            precision = int(self.children[1].value)

            scalar_type = ScalarType(intrinsic, precision)

        # ADD, SUB, MUL, DIV, POW : (arg1, arg2) => promote
        if self.operator in (BinaryOperation.Operator.ADD,
                             BinaryOperation.Operator.SUB,
                             BinaryOperation.Operator.MUL,
                             BinaryOperation.Operator.DIV,
                             BinaryOperation.Operator.POW):
            first = ScalarType(self.children[0].datatype.intrinsic,
                               self.children[0].datatype.precision)
            second = ScalarType(self.children[1].datatype.intrinsic,
                                self.children[1].datatype.precision)
            scalar_type = self._promote_scalar_datatypes(first, second)

        # https://gcc.gnu.org/onlinedocs/gfortran/MOD.html
        if self.operator is BinaryOperation.Operator.REM:
            if (isinstance(self.children[0].datatype.precision,
                           DataNode) \
                    and not isinstance(self.children[0].datatype.precision,
                                       Literal)) \
                or (isinstance(self.children[1].datatype.precision,
                               DataNode) \
                    and not isinstance(self.children[1].datatype.precision,
                                       Literal)):
                raise NotImplementedError("Precision attribute of operand "
                                          "datatype is a DataNode but not a "
                                          "Literal, this is not supported yet.")

            precision = self.children[0].datatype.precision

            # NOTE: (gfortran) kind promotion rules are different for
            # REAL and INTEGER...
            if (self.children[0].datatype.intrinsic
                is ScalarType.Intrinsic.REAL) \
                and ((self.children[0].datatype.precision
                      is ScalarType.Precision.DOUBLE) \
                    or (self.children[1].datatype.precision
                        is ScalarType.Precision.DOUBLE)):
                precision = 8

            # NOT an elif
            if (self.children[0].datatype.precision == 8) \
                or (isinstance(self.children[0].datatype.precision, Literal) \
                    and (self.children[0].datatype.precision.value == "8")) \
                or (self.children[1].datatype.precision == 8) \
                or (isinstance(self.children[1].datatype.precision, Literal) \
                    and (self.children[1].datatype.precision.value == "8")):
                precision = 8

            scalar_type = ScalarType(self.children[0].datatype.intrinsic,
                                     precision)

        if self.operator in (BinaryOperation.Operator.EQ,
                             BinaryOperation.Operator.NE,
                             BinaryOperation.Operator.GT,
                             BinaryOperation.Operator.LT,
                             BinaryOperation.Operator.GE,
                             BinaryOperation.Operator.LE):
            scalar_type = BOOLEAN_TYPE

        if self.operator in (BinaryOperation.Operator.AND,
                             BinaryOperation.Operator.OR,
                             BinaryOperation.Operator.EQV,
                             BinaryOperation.Operator.NEQV):
            if (self.children[0].datatype.precision == 8) \
                or (isinstance(self.children[0].datatype.precision, Literal) \
                    and (self.children[0].datatype.precision.value == "8")) \
                or (self.children[1].datatype.precision == 8) \
                or (isinstance(self.children[1].datatype.precision, Literal) \
                    and (self.children[1].datatype.precision.value == "8")):
                scalar_type = ScalarType(ScalarType.Intrinsic.BOOLEAN, 8)
            else:
                scalar_type = BOOLEAN_TYPE

        if isinstance(self.children[0].datatype, ScalarType) \
            and isinstance(self.children[1].datatype, ScalarType):
            return scalar_type

        if isinstance(self.children[0].datatype, ArrayType) \
            and isinstance(self.children[1].datatype, ArrayType):
            # TODO: should this check whether the shapes are the same?
            # Fortran broadcasting only seems to apply to scalars?
            return ArrayType(scalar_type, self.children[0].datatype.shape)

        if isinstance(self.children[0].datatype, ArrayType) \
            and isinstance(self.children[1].datatype, ScalarType):
            return ArrayType(scalar_type, self.children[0].datatype.shape)

        if isinstance(self.children[0].datatype, ScalarType) \
            and isinstance(self.children[1].datatype, ArrayType):
            return ArrayType(scalar_type, self.children[1].datatype.shape)

        raise NotImplementedError("This shouldn't have happened.")


    @staticmethod
    def _validate_child(position, child):
        '''
        :param int position: the position to be validated.
        :param child: a child to be validated.
        :type child: :py:class:`psyclone.psyir.nodes.Node`

        :return: whether the given child and position are valid for this node.
        :rtype: bool

        '''
        return position in (0, 1) and isinstance(child, DataNode)

    @staticmethod
    def create(operator, lhs, rhs):
        '''Create a BinaryOperator instance given an operator and lhs and rhs
        child instances with optional names.

        :param operator: the operator used in the operation.
        :type operator: \
            :py:class:`psyclone.psyir.nodes.BinaryOperation.Operator`
        :param lhs: the PSyIR node containing the left hand side of \
            the assignment, or a tuple containing the name of the \
            argument and the PSyIR node.
        :type lhs: Union[:py:class:`psyclone.psyir.nodes.Node`, \
            Tuple[str, :py:class:`psyclone.psyir.nodes.Node`]]
        :param rhs: the PSyIR node containing the right hand side of \
            the assignment, or a tuple containing the name of the \
            argument and the PSyIR node.
        :type rhs: Union[:py:class:`psyclone.psyir.nodes.Node`, \
            Tuple[str, :py:class:`psyclone.psyir.nodes.Node`]]

        :returns: a BinaryOperator instance.
        :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`

        :raises GenerationError: if the arguments to the create method \
            are not of the expected type.

        '''
        if not isinstance(operator, Enum) or \
           operator not in BinaryOperation.Operator:
            raise GenerationError(
                f"operator argument in create method of BinaryOperation class "
                f"should be a PSyIR BinaryOperation Operator but found "
                f"'{type(operator).__name__}'.")
        for name, arg in [("lhs", lhs), ("rhs", rhs)]:
            if isinstance(arg, tuple):
                if not len(arg) == 2:
                    raise GenerationError(
                        f"If the {name} argument in create method of "
                        f"BinaryOperation class is a tuple, it's length "
                        f"should be 2, but it is {len(arg)}.")
                if not isinstance(arg[0], str):
                    raise GenerationError(
                        f"If the {name} argument in create method of "
                        f"BinaryOperation class is a tuple, its first "
                        f"argument should be a str, but found "
                        f"{type(arg[0]).__name__}.")

        lhs_name = None
        if isinstance(lhs, tuple):
            lhs_name, lhs = lhs
        rhs_name = None
        if isinstance(rhs, tuple):
            rhs_name, rhs = rhs

        binary_op = BinaryOperation(operator)
        binary_op.append_named_arg(lhs_name, lhs)
        binary_op.append_named_arg(rhs_name, rhs)
        return binary_op

    def reference_accesses(self, var_accesses):
        '''Get all reference access information from this node.
        If the 'COLLECT-ARRAY-SHAPE-READS' options is set, it
        will not report array accesses used as first parameter
        in `lbound`, `ubound`, or `size` as 'read' accesses.

        :param var_accesses: VariablesAccessInfo instance that stores the \
            information about variable accesses.
        :type var_accesses: \
            :py:class:`psyclone.core.VariablesAccessInfo`

        '''
        if not var_accesses.options("COLLECT-ARRAY-SHAPE-READS") \
                and self.operator in [BinaryOperation.Operator.LBOUND,
                                      BinaryOperation.Operator.UBOUND,
                                      BinaryOperation.Operator.SIZE]:
            # If shape accesses are not considered reads, ignore the first
            # child (which is always the array being read)
            for child in self._children[1:]:
                child.reference_accesses(var_accesses)
            return
        for child in self._children:
            child.reference_accesses(var_accesses)


class NaryOperation(Operation):
    '''Node representing a n-ary operation expression. The n operands are
    the stored as the 0 - n-1th children of this node and the type of
    the operator is held in an attribute.

    '''
    # Textual description of the node.
    _children_valid_format = "[DataNode]+"
    _text_name = "NaryOperation"

    Operator = Enum('Operator', [
        # Arithmetic Operators
        'MAX', 'MIN'
        ])
    _non_elemental_ops = []

    @property
    def datatype(self):
        """
        :return: PSyIR datatype of the result of this operation.
        :rtype: :py:class:`psyclone.psyir.symbols.DataType`
        """
        # NOTE: this should be the type and kind of the first argument according
        # to https://gcc.gnu.org/onlinedocs/gfortran/MIN.html
        # but it's not.

        if (self.children[0].datatype.precision == 8) \
            or (isinstance(self.children[0].datatype.precision, Literal) \
                and (self.children[0].datatype.precision.value == "8")) \
            or (self.children[1].datatype.precision == 8) \
            or (isinstance(self.children[1].datatype.precision, Literal) \
                and (self.children[1].datatype.precision.value == "8")) \
            or ((self.children[0].datatype.intrinsic
                 is ScalarType.Intrinsic.REAL) \
                and ((self.children[0].datatype.precision \
                      is ScalarType.Precision.DOUBLE) \
                     or (self.children[1].datatype.precision \
                         is ScalarType.Precision.DOUBLE))):
            return ScalarType(self.children[0].datatype.intrinsic, 8)

        return self.children[0].datatype

    @staticmethod
    def _validate_child(position, child):
        '''
        :param int position: the position to be validated.
        :param child: a child to be validated.
        :type child: :py:class:`psyclone.psyir.nodes.Node`

        :return: whether the given child and position are valid for this node.
        :rtype: bool

        '''
        # pylint: disable=unused-argument
        return isinstance(child, DataNode)

    @staticmethod
    def create(operator, operands):
        '''Create an NaryOperator instance given an operator and a list of
        Node (or name and Node tuple) instances.

        :param operator: the operator used in the operation.
        :type operator: :py:class:`psyclone.psyir.nodes.NaryOperation.Operator`
        :param operands: a list containing PSyIR nodes and/or 2-tuples \
            which contain an argument name and a PSyIR node, that the \
            operator operates on.
        :type operands: List[Union[:py:class:`psyclone.psyir.nodes.Node`, \
            Tuple[str, :py:class:`psyclone.psyir.nodes.DataNode`]]]

        :returns: an NaryOperator instance.
        :rtype: :py:class:`psyclone.psyir.nodes.NaryOperation`

        :raises GenerationError: if the arguments to the create method \
            are not of the expected type.

        '''
        if not isinstance(operator, Enum) or \
           operator not in NaryOperation.Operator:
            raise GenerationError(
                f"operator argument in create method of NaryOperation class "
                f"should be a PSyIR NaryOperation Operator but found "
                f"'{type(operator).__name__}'.")
        if not isinstance(operands, list):
            raise GenerationError(
                f"operands argument in create method of NaryOperation class "
                f"should be a list but found '{type(operands).__name__}'.")

        nary_op = NaryOperation(operator)
        for operand in operands:
            name = None
            if isinstance(operand, tuple):
                if not len(operand) == 2:
                    raise GenerationError(
                        f"If an element of the operands argument in create "
                        f"method of NaryOperation class is a tuple, it's "
                        f"length should be 2, but found {len(operand)}.")
                if not isinstance(operand[0], str):
                    raise GenerationError(
                        f"If an element of the operands argument in create "
                        f"method of NaryOperation class is a tuple, "
                        f"its first argument should be a str, but found "
                        f"{type(operand[0]).__name__}.")
                name, operand = operand
            nary_op.append_named_arg(name, operand)
        return nary_op


# For automatic API documentation generation
__all__ = ["Operation", "UnaryOperation", "BinaryOperation", "NaryOperation"]
