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
Operation, IntrinsicCall and Assignment nodes.
"""


from psyclone.psyir.nodes import (
    Reference,
    DataNode,
    Assignment,
    Literal,
    UnaryOperation,
    BinaryOperation,
    Routine,
    IntrinsicCall,
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
    """This function creates an IntrinsicCall Node with operator LOG \
        corresponding to the natural logarithm of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: an IntrinsicCall Node `log(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of log must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return IntrinsicCall.create(IntrinsicCall.Intrinsic.LOG, [val])


def exp(operand):
    """This function creates an IntrinsicCall Node with operator EXP \
        corresponding to the exponential of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: an IntrinsicCall Node `exp(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of exp must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return IntrinsicCall.create(IntrinsicCall.Intrinsic.EXP, [val])


def cos(operand):
    """This function creates an IntrinsicCall Node with operator COS \
        corresponding to the cosine of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: an IntrinsicCall Node `cos(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of cos must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return IntrinsicCall.create(IntrinsicCall.Intrinsic.COS, [val])


def sin(operand):
    """This function creates an IntrinsicCall Node with operator SIN \
        corresponding to the sine of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: an IntrinsicCall Node `sin(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of sin must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return IntrinsicCall.create(IntrinsicCall.Intrinsic.SIN, [val])


def sqrt(operand):
    """This function creates an IntrinsicCall Node with operator SQRT \
        corresponding to the square root of operand.

    :param operand: operand to be used in the operation.
    :type variable: Union[:py:class:`psyclone.psyir.nodes.DataNode`, \
                          :py:class:`psyclone.psyir.symbols.DataSymbol`]

    :raises TypeError: if the the variable argument is not a DataNode or \
                       DataSymbol instance.

    :return: an IntrinsicCall Node `sqrt(operand)`.
    :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
    """
    if not isinstance(operand, (DataNode, DataSymbol)):
        raise TypeError(
            f"Argument of sqrt must be of type 'DataNode' or "
            f"'DataSymbol' but found '{type(operand).__name__}'."
        )

    val = datanode(operand)

    return IntrinsicCall.create(IntrinsicCall.Intrinsic.SQRT, [val])


def sign(lhs, rhs):
    """This function creates an IntrinsicCall Node with operator SIGN \
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

    :return: an IntrinsicCall `SIGN(lhs, rhs)`.
    :rtype: :py:class:`psyclone.psyir.nodes.IntrinsicCall`
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

    return IntrinsicCall.create(IntrinsicCall.Intrinsic.SIGN, [left, right])


#####################################################
#####################################################
#####################################################
# TODO: functions below need to be tested
# and should be combined?
# NOTE: they could/should be in simplify instead?




def _typecheck_list_of_int_literals(int_literals):
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
        raise TypeError(
            f"'int_literals' argument should be of type 'list' "
            f"but found '{type(int_literals).__name__}'."
        )
    for literal in int_literals:
        if not isinstance(literal, Literal):
            raise TypeError(
                f"'int_literals' argument should be a 'list' "
                f"of elements of type 'Literal' "
                f"but found '{type(literal).__name__}'."
            )
        if not isinstance(literal.datatype, ScalarType):
            raise ValueError(
                f"'int_literals' argument should be a 'list' "
                f"of elements of datatype 'ScalarType' but "
                f"found '{type(literal.datatype).__name__}'."
            )
        if literal.datatype.intrinsic is not ScalarType.Intrinsic.INTEGER:
            raise ValueError(
                f"'int_literals' argument should be a 'list' "
                f"of elements of intrinsic "
                f"'ScalarType.Intrinsic.INTEGER' but found "
                f"'{type(literal.datatype.intrinsic).__name__}'."
            )


def _typecheck_list_of_datanodes(datanodes):
    """Check that the argument is a list of datanodes.

    :param datanodes: list of datanodes.
    :type datanodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

    :raises TypeError: if datanodes is of the wrong type.
    :raises TypeError: if an element of datanodes is of the wrong type.
    """
    if not isinstance(datanodes, list):
        raise TypeError(
            f"'datanodes' argument should be of type 'list' "
            f"but found '{type(datanodes).__name__}'."
        )
    for datanode in datanodes:
        if not isinstance(datanode, DataNode):
            raise TypeError(
                f"'datanodes' argument should be a 'list' "
                f"of elements of type 'DataNode' "
                f"but found '{type(datanode).__name__}'."
            )


def _separate_int_literals(datanodes):
    """| Separates the datanodes from a list into:
    | - a list of scalar integer Literals,
    | - a list of other datanodes. 

    :param datanodes: list of datanodes to separate.
    :type datanodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

    :return: list of integer literals, list of other datanodes
    :rtype: List[:py:class:`psyclone.psyir.nodes.Literal`], \
            List[:py:class:`psyclone.psyir.nodes.DataNode`]
    """
    _typecheck_list_of_datanodes(datanodes)

    int_literals = []
    other_datanodes = []
    for datanode in datanodes:
        if (
            isinstance(datanode, Literal)
            and isinstance(datanode.datatype, ScalarType)
            and (datanode.datatype.intrinsic is ScalarType.Intrinsic.INTEGER)
        ):
            int_literals.append(datanode)
        else:
            other_datanodes.append(datanode)

    return int_literals, other_datanodes


def _add_int_literals(int_literals):
    """Add the int Literals from a list, summing in Python and returning \
    a new Literal.

    :param datanodes: list of literals.
    :type datanodes: List[:py:class:`psyclone.psyir.nodes.Literal`]

    :return: sum, as a Literal.
    :rtype: :py:class:`psyclone.psyir.nodes.Literal`
    """
    _typecheck_list_of_int_literals(int_literals)

    result = 0
    for literal in int_literals:
        result += int(literal.value)

    return Literal(str(result), INTEGER_TYPE)


def _apply_binary_operation(binary_operation):
    if not isinstance(binary_operation, BinaryOperation):
        raise TypeError(
            f"'binary_operation' argument should be of "
            f"type 'BinaryOperation "
            f"but found '{type(datanode).__name__}'."
        )
    if binary_operation.operator is BinaryOperation.Operator.ADD:
        return add_datanodes(binary_operation.children)

    if binary_operation.operator is BinaryOperation.Operator.SUB:
        return substract_datanodes(
            [binary_operation.children[0]], [binary_operation.children[1]]
        )

    if binary_operation.operator is BinaryOperation.Operator.MUL:
        return multiply_datanodes(binary_operation.children)

    return binary_operation


def _apply_all_binary_operations(datanodes):
    _typecheck_list_of_datanodes(datanodes)
    result = []
    for datanode in datanodes:
        if isinstance(datanode, BinaryOperation):
            result.append(_apply_binary_operation(datanode))
        else:
            result.append(datanode)
    return result


def add_datanodes(datanodes):
    """Add the datanodes from a list, dealing with Literals in Python \
    and others in BinaryOperations.

    :param datanodes: list of datanodes.
    :type datanodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

    :return: sum, as a Literal or BinaryOperation.
    :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`, \
                    :py:class:`psyclone.psyir.nodes.BinaryOperation`]
    """
    _typecheck_list_of_datanodes(datanodes)

    int_literals, other_datanodes = _separate_int_literals(datanodes)
    int_sum = _add_int_literals(int_literals)

    if int_sum.value != "0":
        other_datanodes.append(int_sum)

    # other_datanodes = _apply_all_binary_operations(other_datanodes)

    result = zero()
    if len(other_datanodes) != 0:
        result = other_datanodes[0]
        if len(other_datanodes) > 1:
            for datanode in other_datanodes[1:]:
                result = BinaryOperation.create(
                    BinaryOperation.Operator.ADD,
                    result.copy(),
                    datanode.copy(),
                )

    return result


def substract_datanodes(lhs, rhs):
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
    _typecheck_list_of_datanodes(lhs)
    _typecheck_list_of_datanodes(rhs)

    lhs_int_literals, lhs_others = _separate_int_literals(lhs)
    rhs_int_literals, rhs_others = _separate_int_literals(rhs)

    lhs_int_sum = _add_int_literals(lhs_int_literals)
    rhs_int_sum = _add_int_literals(rhs_int_literals)

    int_literal = Literal(
        str(int(lhs_int_sum.value) - int(rhs_int_sum.value)), INTEGER_TYPE
    )

    if int_literal.value != "0":
        lhs_others.append(int_literal)

    # lhs_others = _apply_all_binary_operations(lhs_others)
    # rhs_others = _apply_all_binary_operations(rhs_others)

    result = zero()
    if len(lhs_others) != 0:
        result = lhs_others[0]
        if len(lhs_others) > 1:
            for datanode in lhs_others[1:]:
                result = BinaryOperation.create(
                    BinaryOperation.Operator.ADD,
                    result.copy(),
                    datanode.copy(),
                )

    if len(rhs_others) != 0:
        substract = rhs_others[0]
        if len(rhs_others) > 1:
            for datanode in rhs_others[1:]:
                substract = BinaryOperation.create(
                    BinaryOperation.Operator.ADD,
                    substract.copy(),
                    datanode.copy(),
                )
        result = BinaryOperation.create(
            BinaryOperation.Operator.SUB, result.copy(), substract.copy()
        )

    return result


def _multiply_int_literals(int_literals):
    """Multiply the int literals from a list.
    Performs the multiplication in Python and returns a new Literal.

    :param datanodes: list of literals.
    :type datanodes: List[:py:class:`psyclone.psyir.nodes.Literal`]

    :return: multiplication, as a Literal.
    :rtype: :py:class:`psyclone.psyir.nodes.Literal`
    """
    _typecheck_list_of_int_literals(int_literals)

    result = 1
    for literal in int_literals:
        result *= int(literal.value)

    return Literal(str(result), INTEGER_TYPE)


def multiply_datanodes(datanodes):
    """Multiply the datanodes from a list, dealing with Literals in Python \
    and others in BinaryOperations.

    :param datanodes: list of datanodes.
    :type datanodes: List[:py:class:`psyclone.psyir.nodes.DataNode`]

    :return: multiplication, as a Literal or BinaryOperation.
    :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`, \
                    :py:class:`psyclone.psyir.nodes.BinaryOperation`]
    """
    _typecheck_list_of_datanodes(datanodes)

    int_literals, other_datanodes = _separate_int_literals(datanodes)
    int_mul = _multiply_int_literals(int_literals)

    if int_mul.value != "1":
        other_datanodes.append(int_mul)

    # other_datanodes = _apply_all_binary_operations(other_datanodes)

    result = one()
    if len(other_datanodes) != 0:
        result = other_datanodes[0]
        if len(other_datanodes) > 1:
            for datanode in other_datanodes[1:]:
                result = BinaryOperation.create(
                    BinaryOperation.Operator.MUL,
                    result.copy(),
                    datanode.copy(),
                )

    return result
