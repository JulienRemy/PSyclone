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

"""This file contains various utility functions used to create common
Operation and Assignment nodes.
"""


from psyclone.psyir.nodes import (
    Reference,
    DataNode,
    Assignment,
    Literal,
    UnaryOperation,
    BinaryOperation,
    Routine,
)
from psyclone.psyir.symbols import (
    DataSymbol,
    ScalarType,
    ArrayType,
    REAL_TYPE,
    INTEGER_TYPE,
)


def own_routine_symbol(routine):
    """Get the RoutineSymol of routine, ie. the symbol tagged \
    'own_routine_symbol' from the routine argument's SymbolTable.

    :param routine: routine whose symbol to return.
    :type routine: :py:class:`psyclone.psyir.nodes.Routine`

    :raises TypeError: if routine is of the wrong type.

    :return: symbol of the routine
    :rtype: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
    """
    if not isinstance(routine, Routine):
        raise TypeError(
            f"'routine' argument should be of "
            f"type 'Routine' but found "
            f"'{type(routine).__name__}'."
        )

    return routine.symbol_table.lookup_with_tag("own_routine_symbol")


def datanode(sym_or_datanode):
    """This function creates a Reference from a DataSymbol, \
    copies a DataNode if it's attached or returns it otherwise.

    :param sym_or_ref: symbol or datanode.
    :type sym_or_ref: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                            :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :raises TypeError: if sym_or_datanode is of the wrong type.

    :return: datanode.
    :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
    """
    if not isinstance(sym_or_datanode, (DataNode, DataSymbol)):
        raise TypeError(
            f"The sym_or_datanode argument of datanode must be of "
            f"type 'DataNode' or 'DataSymbol' but found "
            f"'{type(sym_or_datanode).__name__}'."
        )

    if isinstance(sym_or_datanode, DataSymbol):
        return Reference(sym_or_datanode)

    if sym_or_datanode.parent is None:
        return sym_or_datanode

    return sym_or_datanode.copy()


def assign(variable, value):
    """This function creates an Assignment Node between two References or \
    DataSymbols.

    :param variable: LHS of Assignment.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.Reference`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :param value: RHS of Assignment.
    :type value: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                       :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a Reference or \
                       DataSymbol instance
    :raises TypeError: if the the value argument is not a DataNode or \
                       DataSymbol instance

    :return: an Assignement node `variable = value`.
    :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
    """
    if not isinstance(variable, (Reference, DataSymbol)):
        raise TypeError(
            f"The variable argument of assign must be of "
            f"type 'Reference' or 'DataSymbol' but found "
            f"'{type(variable).__name__}'."
        )
    if not isinstance(value, (DataNode, DataSymbol)):
        raise TypeError(
            f"The value argument of assign must be of "
            f"type 'DataNode' or 'DataSymbol' but found "
            f"'{type(value).__name__}'."
        )

    ref = datanode(variable)
    val = datanode(value)

    return Assignment.create(ref, val)


def assign_zero(variable):
    """This function creates an Assignment Node with zero on the RHS, \
    respecting the LHS datatype.

    :param variable: LHS of Assignment.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.Reference`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a Reference or \
                       DataSymbol instance

    :return: an Assignement node `variable = 0`.
    :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
    """
    if not isinstance(variable, (Reference, DataSymbol)):
        raise TypeError(
            f"The variable argument of assign_zero must be of "
            f"type 'Reference' or 'DataSymbol' but found "
            f"'{type(variable).__name__}'."
        )

    if isinstance(variable.datatype, ScalarType):
        datatype = variable.datatype
    else:
        datatype = variable.datatype.datatype

    return assign(variable, zero(datatype))


def increment(variable, value):
    """This function creates an Assignment Node corresponding to an \
    incrementation of variable by value (ie. C++ style `variable += value`).

    :param variable: LHS of Assignment, to be incremented.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.Reference`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :param value: incrementation value.
    :type value: Union[:py:class:`psyclone.psyir.nodes.Reference`, \
                       :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a Reference or \
                       DataSymbol instance.
    :raises TypeError: if the the value argument is not a Reference or \
                       DataSymbol instance.

    :return: an Assignement node `variable = variable + value`.
    :rtype: :py:class:`psyclone.psyir.nodes.Assignment`
    """
    if not isinstance(variable, (Reference, DataSymbol)):
        raise TypeError(
            f"The variable argument of increment must be of "
            f"type 'Reference' or 'DataSymbol' but found "
            f"'{type(variable).__name__}'."
        )
    if not isinstance(value, (DataNode, DataSymbol)):
        raise TypeError(
            f"The value argument of increment must be of "
            f"type 'DataNode' or 'DataSymbol' but found "
            f"'{type(value).__name__}'."
        )

    operation = add(variable, value)

    return assign(variable, operation)


def zero(datatype=INTEGER_TYPE):
    """ This function creates a Literal Node with value 0 of a given datatype.

    :param datatype: datatype of the returned Literal Node, \
                     defaults to INTEGER_TYPE.
    :type datatype: Optional[\
                        Union[:py:class:`psyclone.psyir.symbols.ScalarType`, \
                              :py:class:`psyclone.psyir.symbols.ArrayType`]\
                    ]

    :raises TypeError: if datatype is not an instance of `ArrayType` or \
                       `ScalarType`.
    :raises NotImplementedError: if the intrinsic of the 'ScalarType` is \
                                 neither `INTEGER` nor `REAL`.

    :return: a Literal Node with value 0 of correct datatype.
    :rtype: :py:class:`psyclone.psyir.nodes.Literal`
    """
    if not isinstance(datatype, (ScalarType, ArrayType)):
        raise TypeError(
            f"The datatype argument of zero should be of type "
            f"psyir.symbols.ScalarType or psyir.symbols.ArrayType "
            f"but found '{type(datatype).__name__}'."
        )

    if isinstance(datatype, ScalarType):
        new_datatype = datatype
    else:
        new_datatype = datatype.datatype

    if new_datatype.intrinsic == ScalarType.Intrinsic.INTEGER:
        return Literal("0", datatype)
    if new_datatype.intrinsic == ScalarType.Intrinsic.REAL:
        # TODO: Literal doesn't accept "0d0", seems like a bug
        # if datatype.precision == ScalarType.Precision.DOUBLE:
        #    return Literal("0d0", datatype)
        return Literal("0.0", new_datatype)

    raise NotImplementedError(
        "Creating null Literals for types other than integer "
        "or real is not implemented yet."
    )


def one(datatype=INTEGER_TYPE):
    """ This function creates a Literal Node with value 1 of a given datatype.

    :param datatype: datatype of the returned Literal Node, \
                     defaults to INTEGER_TYPE.
    :type datatype: Optional[\
                        Union[:py:class:`psyclone.psyir.symbols.ScalarType`, \
                              :py:class:`psyclone.psyir.symbols.ArrayType`]\
                    ]

    :raises NotImplementedError: if datatype is an instance of `ArrayType`.
    :raises TypeError: if datatype is not an instance of `ArrayType` \
                       or `ScalarType`.
    :raises NotImplementedError: if the intrinsic of the 'ScalarType` is \
                                 neither `INTEGER` nor `REAL`.

    :return: a Literal Node with value 1 of correct datatype.
    :rtype: :py:class:`psyclone.psyir.nodes.Literal`
    """
    if not isinstance(datatype, (ScalarType, ArrayType)):
        raise TypeError(
            f"The datatype argument of one should be of type "
            f"psyir.symbols.ScalarType or psyir.symbols.ArrayType "
            f"but found '{type(datatype).__name__}'."
        )

    if isinstance(datatype, ScalarType):
        new_datatype = datatype
    else:
        new_datatype = datatype.datatype

    if new_datatype.intrinsic == ScalarType.Intrinsic.INTEGER:
        return Literal("1", datatype)
    if new_datatype.intrinsic == ScalarType.Intrinsic.REAL:
        # TODO: Literal doesn't accept "1d0", seems like a bug
        # if datatype.precision == ScalarType.Precision.DOUBLE:
        #    return Literal("1d0", datatype)
        return Literal("1.0", new_datatype)

    raise NotImplementedError(
        "Creating unitary Literals for types other than "
        "integer or real is not implemented yet."
    )


def minus(operand):
    """This function creates a UnaryOperation Node with operator MINUS and \
        the operand argument.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: a UnaryOperation node `-operand`.
    :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of minus must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return UnaryOperation.create(UnaryOperation.Operator.MINUS, val)


def div(lhs, rhs):
    """This function creates a BinaryOperation Node with operator DIV \
        corresponding to `lhs / rhs`.

    :param lhs: numerator.
    :type lhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :param rhs: denominator.
    :type rhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if lhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.
    :raises TypeError: if rhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.

    :return: multiply BinaryOperation `lhs / rhs`.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    if not isinstance(lhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"lhs argument of div must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(lhs).__name__}'."
        )
    if not isinstance(rhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"rhs argument of div must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(rhs).__name__}'."
        )

    left = datanode(lhs)
    right = datanode(rhs)

    return BinaryOperation.create(BinaryOperation.Operator.DIV, left, right)


def inverse(operand):
    """This function creates a UnaryOperation Node with operator DIV \
        corresponding to the inverse of the operand argument.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: a UnaryOperation node `1.0/operand` where 1.0 is of datatype \
             REAL_TYPE.
    :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of inverse must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return div(one(REAL_TYPE), val)


def power(lhs, rhs):
    """This function creates a BinaryOperation Node with operator POW \
        corresponding to `lhs**rhs`.

    :param lhs: variable to be raised to a power.
    :type lhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :param rhs: exponent of the power.
    :type rhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if lhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.
    :raises TypeError: if rhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.

    :return: power BinaryOperation `lhs**rhs`.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    if not isinstance(lhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"lhs argument of power must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(lhs).__name__}'."
        )
    if not isinstance(rhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"rhs argument of power must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(rhs).__name__}'."
        )

    left = datanode(lhs)
    right = datanode(rhs)

    return BinaryOperation.create(BinaryOperation.Operator.POW, left, right)


def square(operand):
    """This function creates a BinaryOperation Node with operator POW \
        corresponding to the square of the operand argument.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: a BinaryOperation Node `operand**2` where 2 is of datatype \
             INTEGER_TYPE.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of square must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return power(val, Literal("2", INTEGER_TYPE))


def mul(lhs, rhs):
    """This function creates a BinaryOperation Node with operator MUL \
        corresponding to `lhs * rhs`.

    :param lhs: first variable to be multiplied.
    :type lhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :param rhs: second variable to be multiplied.
    :type rhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if lhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.
    :raises TypeError: if rhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.

    :return: multiply BinaryOperation `lhs * rhs`.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    if not isinstance(lhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"lhs argument of mul must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(lhs).__name__}'."
        )
    if not isinstance(rhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"rhs argument of mul must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(rhs).__name__}'."
        )

    left = datanode(lhs)
    right = datanode(rhs)

    return BinaryOperation.create(BinaryOperation.Operator.MUL, left, right)


def sub(lhs, rhs):
    """This function creates a BinaryOperation Node with operator SUB \
        corresponding to `lhs - rhs`.

    :param lhs: lhs of substraction.
    :type lhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :param rhs: rhs of substraction.
    :type rhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if lhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.
    :raises TypeError: if rhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.

    :return: substraction BinaryOperation `lhs - rhs`.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    if not isinstance(lhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"lhs argument of sub must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(lhs).__name__}'."
        )
    if not isinstance(rhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"rhs argument of sub must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(rhs).__name__}'."
        )

    left = datanode(lhs)
    right = datanode(rhs)

    return BinaryOperation.create(BinaryOperation.Operator.SUB, left, right)


def add(lhs, rhs):
    """This function creates a BinaryOperation Node with operator ADD \
        corresponding to `lhs + rhs`.

    :param lhs: first variable to be added.
    :type lhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :param rhs: second variable to be added.
    :type rhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if lhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.
    :raises TypeError: if rhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.

    :return: addition BinaryOperation `lhs + rhs`.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    if not isinstance(lhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"lhs argument of add must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(lhs).__name__}'."
        )
    if not isinstance(rhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"rhs argument of add must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(rhs).__name__}'."
        )

    left = datanode(lhs)
    right = datanode(rhs)

    return BinaryOperation.create(BinaryOperation.Operator.ADD, left, right)


def log(operand):
    """This function creates a UnaryOperation Node with operator LOG \
        corresponding to the natural logarithm of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: a UnaryOperation Node `log(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of log must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return UnaryOperation.create(UnaryOperation.Operator.LOG, val)


def exp(operand):
    """This function creates a UnaryOperation Node with operator EXP \
        corresponding to the exponential of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: a UnaryOperation Node `exp(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of exp must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return UnaryOperation.create(UnaryOperation.Operator.EXP, val)


def cos(operand):
    """This function creates a UnaryOperation Node with operator COS \
        corresponding to the cosine of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: a UnaryOperation Node `cos(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of cos must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return UnaryOperation.create(UnaryOperation.Operator.COS, val)


def sin(operand):
    """This function creates a UnaryOperation Node with operator SIN \
        corresponding to the sine of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: a UnaryOperation Node `sin(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of sin must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return UnaryOperation.create(UnaryOperation.Operator.SIN, val)


def sqrt(operand):
    """This function creates a UnaryOperation Node with operator SQRT \
        corresponding to the square root of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: a UnaryOperation Node `sqrt(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.UnaryOperation`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of sqrt must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return UnaryOperation.create(UnaryOperation.Operator.SQRT, val)


def sign(lhs, rhs):
    """This function creates a BinaryOperation Node with operator SIGN \
        corresponding to `|lhs| * sign(rhs)`.

    :param lhs: variable giving the absolute value of the result.
    :type lhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]
    :param rhs: variable giving the sign of the result.
    :type rhs: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                     :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if lhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.
    :raises TypeError: if rhs argument is not an instance of 'DataNode' \
                       nor 'DataSymbol'.

    :return: addition BinaryOperation `SIGN(lhs, rhs)`.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    if not isinstance(lhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"lhs argument of sign must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(lhs).__name__}'."
        )
    if not isinstance(rhs, (DataNode, DataSymbol)):
        raise TypeError(
            f"rhs argument of sign must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(rhs).__name__}'."
        )

    left = datanode(lhs)
    right = datanode(rhs)

    return BinaryOperation.create(BinaryOperation.Operator.SIGN, left, right)
