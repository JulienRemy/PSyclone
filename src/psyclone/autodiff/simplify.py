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
# Author J. Remy, Inria

"""This file contains functions used to simplify PSyclone Nodes created 
by applying 'psyclone.autodiff' transformations.
"""

from psyclone.psyir.nodes import (
    UnaryOperation,
    BinaryOperation,
    Literal,
    Assignment,
    Node,
    Reference,
)
from psyclone.psyir.symbols import INTEGER_TYPE


def _typecheck_binary_operation(binary_operation):
    if not isinstance(binary_operation, BinaryOperation):
        raise TypeError(
            f"'binary_operation' argument should be of "
            f"type 'BinaryOperation' but found "
            f"'{type(binary_operation).__name__}'."
        )


def simplify_binary_operation(binary_operation, times=1):
    """Simplifies a binary operation 'times' times.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    :param times: number of times to apply simplification, defaults to 1
    :type times: Optional[int]

    :raises TypeError: if binary_operation is of the wrong type.
    :raises TypeError: if times is of the wrong type.
    :raises ValueError: if times is less than 1.

    :return: the simplified operation.
    :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`,
                  :py:class:`psyclone.psyir.nodes.Reference`,
                  :py:class:`psyclone.psyir.nodes.Operation`]
    """
    _typecheck_binary_operation(binary_operation)
    if not isinstance(times, int):
        raise TypeError(
            f"'times' argument should be of type "
            f"'int' but found '{type(times).__name__}'."
        )
    if times < 1:
        raise ValueError(
            f"'times' argument should be at least '1' " f"but found '{times}'."
        )

    result = binary_operation

    if binary_operation.operator in (
        BinaryOperation.Operator.MUL,
        BinaryOperation.Operator.DIV,
    ):
        result = simplify_mul_div(binary_operation)

    # x +- a * x => (1 +- a) * x
    # a * x +- x => (a +- 1) * x
    # x +- x * a => x * (1 +- a)
    # x * a +- x => x * (a +- 1)
    elif binary_operation.operator in (
        BinaryOperation.Operator.ADD,
        BinaryOperation.Operator.SUB,
    ):
        for i, operand in enumerate(binary_operation.children):
            other_operand = binary_operation.children[i - 1]
            if (
                isinstance(operand, Reference)
                and isinstance(other_operand, BinaryOperation)
                and (other_operand.operator is BinaryOperation.Operator.MUL)
            ):
                for j, suboperand in enumerate(other_operand.children):
                    if suboperand == operand:
                        result = simplify_add_sub_factorize(binary_operation)

        if binary_operation.operator is BinaryOperation.Operator.ADD:
            result = simplify_add(binary_operation)

        elif binary_operation.operator is BinaryOperation.Operator.SUB:
            result = simplify_sub(binary_operation)

    if (times > 1) and (result is not binary_operation):
        result = simplify_binary_operation(result, times - 1)

    return result
    ###############################


def _typecheck_add(binary_operation):
    _typecheck_binary_operation(binary_operation)
    if binary_operation.operator is not BinaryOperation.Operator.ADD:
        raise ValueError(
            f"'binary_operation' argument should have "
            f"operator 'BinaryOperation.Operator.ADD' "
            f"but found '{binary_operation.operator}'."
        )


def simplify_add(binary_operation):
    """Simplifies a binary operation with operator ADD.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not ADD.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_add(binary_operation)

    # x + x => 2 * x
    if isinstance(binary_operation.children[0], Reference) and (
        binary_operation.children[0] == binary_operation.children[1]
    ):
        return simplify_add_twice_to_mul_by_2(binary_operation)

    # (-x) + (-y) => -(x+y)
    if (
        isinstance(binary_operation.children[0], UnaryOperation)
        and (binary_operation.children[0].operator is UnaryOperation.Operator.MINUS)
        and isinstance(binary_operation.children[1], UnaryOperation)
        and (binary_operation.children[1].operator is UnaryOperation.Operator.MINUS)
    ):
        return simplify_add_minus_minus(binary_operation)

    # x + (-y) => x - y
    # (-x) + y => y - x
    for i, operand in enumerate(binary_operation.children):
        other_operand = binary_operation.children[i - 1]
        if isinstance(operand, UnaryOperation) and (
            operand.operator is UnaryOperation.Operator.MINUS
        ):
            return simplify_add_plus_minus_or_minus_plus(binary_operation)

    return binary_operation


# x + x => 2 * x
def simplify_add_twice_to_mul_by_2(binary_operation):
    """Simplifies a binary operation with operator ADD and both operands
    being of type 'Reference' and equal.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not ADD.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_add(binary_operation)

    if not (
        isinstance(binary_operation.children[0], Reference)
        and (binary_operation.children[0] == binary_operation.children[1])
    ):
        raise ValueError(
            f"'binary_operation' argument should have children "
            f"of type 'Reference' and both equal "
            f"but found '{binary_operation.children[0].view()}' "
            f"of type '{type(binary_operation.children[0]).__name__}' and "
            f"'{binary_operation.children[1].view()}' "
            f"of type '{type(binary_operation.children[1]).__name__}'."
        )

    return BinaryOperation.create(
        BinaryOperation.Operator.MUL,
        Literal("2", INTEGER_TYPE),
        binary_operation.children[0].copy(),
    )


# (-x) + (-y) => -(x+y)
def simplify_add_minus_minus(binary_operation):
    """Simplifies a binary operation with operator ADD, both operands being
    unary operations with operator MINUS.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not ADD.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_add(binary_operation)

    if not (
        (
            isinstance(binary_operation.children[0], UnaryOperation)
            and (binary_operation.children[0].operator is UnaryOperation.Operator.MINUS)
            and isinstance(binary_operation.children[1], UnaryOperation)
            and (binary_operation.children[1].operator is UnaryOperation.Operator.MINUS)
        )
    ):
        raise ValueError(
            f"'binary_operation' argument children should both be of type "
            f"'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS'"
            f"but found '{binary_operation.children[0].view()}' "
            f"of type '{type(binary_operation.children[0]).__name__}' and "
            f"'{binary_operation.children[1].view()}' "
            f"of type '{type(binary_operation.children[1]).__name__}'."
        )

    return UnaryOperation.create(
        UnaryOperation.Operator.MINUS,
        BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            binary_operation.children[0].children[0].copy(),
            binary_operation.children[1].children[0].copy(),
        ),
    )


# x + (-y) => x - y
# (-x) + y => y - x
def simplify_add_plus_minus_or_minus_plus(binary_operation):
    """Simplifies a binary operation with operator ADD, one operand being
    a Reference and the other being a unary operation with operator MINUS.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not ADD.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_add(binary_operation)

    if not (
        (
            isinstance(binary_operation.children[0], UnaryOperation)
            and (binary_operation.children[0].operator is UnaryOperation.Operator.MINUS)
        )
        ^ (
            isinstance(binary_operation.children[1], UnaryOperation)
            and (binary_operation.children[1].operator is UnaryOperation.Operator.MINUS)
        )
    ):
        raise ValueError(
            f"Exactly one of 'binary_operation' argument children should have "
            f"type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
            f"but found '{binary_operation.children[0].view()}' of type "
            f"'{type(binary_operation.children[0]).__name__}' and "
            f"'{binary_operation.children[1].view()}' of type "
            f"'{type(binary_operation.children[1]).__name__}'."
        )

    for i, operand in enumerate(binary_operation.children):
        other_operand = binary_operation.children[i - 1]
        if isinstance(operand, UnaryOperation) and (
            operand.operator is UnaryOperation.Operator.MINUS
        ):
            return BinaryOperation.create(
                BinaryOperation.Operator.SUB,
                other_operand.copy(),
                operand.children[0].copy(),
            )
            # binary_operation.replace_with(new_sub)


def _typecheck_sub(binary_operation):
    _typecheck_binary_operation(binary_operation)
    if binary_operation.operator is not BinaryOperation.Operator.SUB:
        raise ValueError(
            f"'binary_operation' argument should have "
            f"operator 'BinaryOperation.Operator.SUB' "
            f"but found '{binary_operation.operator}'."
        )


def simplify_sub(binary_operation):
    """Simplifies a binary operation with operator SUB.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not SUB.

    :return: the simplified operation.
    :rtype: Union[:py:class:`psyclone.psyir.nodes.BinaryOperation`,
                  :py:class:`psyclone.psyir.nodes.Literal`]
    """
    _typecheck_sub(binary_operation)

    # x - x => 0
    if (
        isinstance(binary_operation.children[0], Reference)
        and binary_operation.children[0] == binary_operation.children[1]
    ):
        return simplify_sub_itself_to_zero(binary_operation)

    # (-x) - y => -(x+y)
    if (
        isinstance(binary_operation.children[0], UnaryOperation)
        and binary_operation.children[0].operator is UnaryOperation.Operator.MINUS
    ):
        return simplify_sub_minus_plus(binary_operation)

    # x - (-y) => x + y
    if (
        isinstance(binary_operation.children[1], UnaryOperation)
        and binary_operation.children[1].operator is UnaryOperation.Operator.MINUS
    ):
        return simplify_sub_plus_minus(binary_operation)

    return binary_operation


# x - x => 0
def simplify_sub_itself_to_zero(binary_operation):
    """Simplifies a binary operation with operator SUB, both operands being
    of type 'Reference' and equal.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not SUB.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.Literal`
    """
    _typecheck_sub(binary_operation)

    if not (
        isinstance(binary_operation.children[0], Reference)
        and binary_operation.children[0] == binary_operation.children[1]
    ):
        raise ValueError(
            f"'binary_operation' argument should have children "
            f"of type 'Reference' and both equal "
            f"but found '{binary_operation.children[0].view()}' "
            f"of type '{type(binary_operation.children[0]).__name__}' and "
            f"'{binary_operation.children[1].view()}' "
            f"of type '{type(binary_operation.children[1]).__name__}'."
        )

    return Literal("0", INTEGER_TYPE)


# (-x) - y => -(x+y)
def simplify_sub_minus_plus(binary_operation):
    """Simplifies a binary operation with operator SUB, the first operand
    being a unary operation with operator MINUS.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not SUB.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_sub(binary_operation)

    if not (
        isinstance(binary_operation.children[0], UnaryOperation)
        and binary_operation.children[0].operator is UnaryOperation.Operator.MINUS
    ):
        raise ValueError(
            f"'binary_operation' argument should have first child "
            f"of type 'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS' "
            f"but found '{binary_operation.children[0].view()}' "
            f"of type '{type(binary_operation.children[0]).__name__}'."
        )

    return UnaryOperation.create(
        UnaryOperation.Operator.MINUS,
        BinaryOperation.create(
            BinaryOperation.Operator.ADD,
            binary_operation.children[0].children[0].copy(),
            binary_operation.children[1].copy(),
        ),
    )


# x - (-y) => x + y
def simplify_sub_plus_minus(binary_operation):
    """Simplifies a binary operation with operator SUB, the second operand
    being a unary operation with operator MINUS.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not SUB.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_sub(binary_operation)

    if not (
        isinstance(binary_operation.children[1], UnaryOperation)
        and binary_operation.children[1].operator is UnaryOperation.Operator.MINUS
    ):
        raise ValueError(
            f"'binary_operation' argument should have second child "
            f"of type 'UnaryOperation' with operator 'UnaryOperation.Operator.MINUS' "
            f"but found '{binary_operation.children[1].view()}' "
            f"of type '{type(binary_operation.children[1]).__name__}'."
        )

    return BinaryOperation.create(
        BinaryOperation.Operator.ADD,
        binary_operation.children[0].copy(),
        binary_operation.children[1].children[0].copy(),
    )


def _typecheck_add_sub(binary_operation):
    _typecheck_binary_operation(binary_operation)
    if binary_operation.operator not in (
        BinaryOperation.Operator.ADD,
        BinaryOperation.Operator.SUB,
    ):
        raise ValueError(
            f"'binary_operation' argument should have "
            f"operator either 'BinaryOperation.Operator.ADD' "
            f"or 'BinaryOperation.Operator.SUB' "
            f"but found '{binary_operation.operator}'."
        )


# x +- a * x => (1 +- a) * x
# a * x +- x => (a +- 1) * x
# x +- x * a => x * (1 +- a)
# x * a +- x => x * (a +- 1)
def simplify_add_sub_factorize(binary_operation):
    """Simplifies a binary operation with operator ADD or SUB, one operand (1)
    being of type Reference and the other operand being a binary operation
    with one suboperand equal to (1).

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not SUB.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_add_sub(binary_operation)

    if not (
        isinstance(binary_operation.children[0], Reference)
        ^ isinstance(binary_operation.children[1], Reference)
    ):
        raise ValueError(
            f"'binary_operation' argument should have exactly one child "
            f"of type 'Reference' "
            f"but found '{binary_operation.children[0].view()}' "
            f"of type '{type(binary_operation.children[0]).__name__}' and "
            f"'{binary_operation.children[1].view()}' "
            f"of type '{type(binary_operation.children[1]).__name__}'."
        )

    for i, operand in enumerate(binary_operation.children):
        other_operand = binary_operation.children[i - 1]
        if isinstance(operand, Reference):
            if not isinstance(other_operand, BinaryOperation):
                raise TypeError(
                    f"'binary_operation' argument should have one child "
                    f"of type 'Reference' and the other of type 'BinaryOperation' "
                    f"but found '{operand.view()}' of type 'Reference' and "
                    f"'{other_operand.view()}'"
                    f"of type '{type(other_operand).__name__}'."
                )
            if other_operand.operator is not BinaryOperation.Operator.MUL:
                raise ValueError(
                    f"'binary_operation' argument should have one child "
                    f"of type 'Reference' and the other of type 'BinaryOperation' "
                    f"with operator 'BinaryOperation.Operator.MUL' but found '{operand.view()}' of type 'Reference' and "
                    f"'{other_operand.view()}' with operator '{other_operand.operator}'."
                )
            if not any(
                [suboperand == operand for suboperand in other_operand.children]
            ):
                raise ValueError(
                    f"'binary_operation' argument should have one child 'operand' "
                    f"of type 'Reference' and the other of type 'BinaryOperation' "
                    f"with an operand equal to 'operand' "
                    f"but found '{operand.view()}' of type 'Reference' and "
                    f"{[suboperand.view() for suboperand in other_operand.children]}."
                )

    for i, operand in enumerate(binary_operation.children):
        other_operand = binary_operation.children[i - 1]
        if (
            isinstance(operand, Reference)
            and isinstance(other_operand, BinaryOperation)
            and (other_operand.operator is BinaryOperation.Operator.MUL)
        ):
            for j, suboperand in enumerate(other_operand.children):
                other_suboperand = other_operand.children[j - 1]
                if suboperand == operand:
                    if isinstance(other_suboperand, Literal):
                        (
                            whole,
                            dot,
                            decimal,
                        ) = other_suboperand.value.partition(".")
                        if binary_operation.operator is BinaryOperation.Operator.ADD:
                            new_whole = str(int(whole) + 1)
                            factor = Literal(
                                new_whole + dot + decimal,
                                other_suboperand.datatype,
                            )
                        else:
                            if i == 0:
                                new_whole = str(1 - int(whole))
                            else:
                                new_whole = str(int(whole) - 1)

                            factor = Literal(
                                new_whole + dot + decimal,
                                other_suboperand.datatype,
                            )
                    else:
                        new_suboperands = [
                            Literal("1", INTEGER_TYPE),
                            other_suboperand.copy(),
                        ]
                        if i == 1:
                            new_suboperands.reverse()
                        factor = BinaryOperation.create(
                            binary_operation.operator, *new_suboperands
                        )
                    new_operands = [operand.copy(), factor]
                    if j == 1:
                        new_operands.reverse()
                    return BinaryOperation.create(
                        BinaryOperation.Operator.MUL, *new_operands
                    )

    raise ValueError(
        f"'simplify_add_sub_factorize' called on argument "
        f"{binary_operation.view()} encountered a problem "
        f"and should not have reached this point."
    )


def _typecheck_mul_div(binary_operation):
    _typecheck_binary_operation(binary_operation)
    if binary_operation.operator not in (
        BinaryOperation.Operator.MUL,
        BinaryOperation.Operator.DIV,
    ):
        raise ValueError(
            f"'binary_operation' argument should have "
            f"an operator among 'BinaryOperation.Operator.MUL' and "
            f"'BinaryOperation.Operator.DIV' but found "
            f"'{binary_operation.operator}'."
        )


def simplify_mul_div(binary_operation):
    """Simplifies a binary operation with operator MUL or DIV.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not MUL or DIV.

    :return: the simplified operation.
    :rtype: Union[:py:class:`psyclone.psyir.nodes.BinaryOperation`,
                  :py:class:`psyclone.psyir.nodes.Literal`]
    """
    _typecheck_mul_div(binary_operation)

    # x * 1 or 1 * x => x, x * 0 or 0 * x => 0
    if (binary_operation.operator is BinaryOperation.Operator.MUL) and (
        isinstance(binary_operation.children[0], Literal)
        or isinstance(binary_operation.children[1], Literal)
    ):
        for operand in binary_operation.children:
            if isinstance(operand, Literal):
                if operand.value == "1":
                    return simplify_mul_by_one(binary_operation)
                if operand.value in ("0", "0.0", "0."):
                    return simplify_mul_by_zero(binary_operation)

    # (-x) */ (-y) => x */ y
    if (
        isinstance(binary_operation.children[0], UnaryOperation)
        and isinstance(binary_operation.children[1], UnaryOperation)
        and (
            binary_operation.children[0].operator
            == binary_operation.children[1].operator
            == UnaryOperation.Operator.MINUS
        )
    ):
        return simplify_mul_div_minus_minus(binary_operation)

    # (-x) */ y or x */ (-y) => -(x */ y)
    if (
        isinstance(binary_operation.children[0], UnaryOperation)
        and (binary_operation.children[0].operator is UnaryOperation.Operator.MINUS)
    ) ^ (
        isinstance(binary_operation.children[1], UnaryOperation)
        and (binary_operation.children[1].operator is UnaryOperation.Operator.MINUS)
    ):
        return simplify_mul_div_plus_minus_or_minus_plus(binary_operation)

    return binary_operation


def _typecheck_unary_operation(unary_operation):
    if not isinstance(unary_operation, UnaryOperation):
        raise TypeError(
            f"'unary_operation' argument should be "
            f"of type 'UnaryOperation' but found "
            f"'{type(unary_operation).__name__}'."
        )


def _typecheck_minus(unary_operation):
    _typecheck_unary_operation(unary_operation)
    if unary_operation.operator is not UnaryOperation.Operator.MINUS:
        raise ValueError(
            f"'unary_operation' argument should have "
            f"operator 'UnaryOperation.Operator.MINUS' but found "
            f"'{unary_operation.operator}'."
        )


# (-x) */ (-y) => x */ y
def simplify_mul_div_minus_minus(binary_operation):
    """Simplifies a binary operation with operator MUL or DIV, both operands
    being unary operations with operator MINUS.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not MUL or DIV.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_mul_div(binary_operation)

    for operand in binary_operation.children:
        _typecheck_minus(operand)

    new_operands = [operand.children[0].copy() for operand in binary_operation.children]
    return BinaryOperation.create(binary_operation.operator, *new_operands)
    # binary_operation.replace_with(new_binary_operation)


# (-x) */ y or x */ (-y) => -(x */ y)
def simplify_mul_div_plus_minus_or_minus_plus(binary_operation):
    """Simplifies a binary operation with operator MUL or DIV, exactly one operand
    being a unary operation with operator MINUS.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not MUL or DIV.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.BinaryOperation`
    """
    _typecheck_mul_div(binary_operation)

    if not (
        (
            isinstance(binary_operation.children[0], UnaryOperation)
            and (binary_operation.children[0].operator is UnaryOperation.Operator.MINUS)
        )
        ^ (
            isinstance(binary_operation.children[1], UnaryOperation)
            and (binary_operation.children[1].operator is UnaryOperation.Operator.MINUS)
        )
    ):
        raise ValueError(
            f"Exactly one of 'binary_operation' argument children should have "
            f"type 'UnaryOperation' and operator 'UnaryOperation.Operator.MINUS' "
            f"but found '{binary_operation.children[0].view()}' of type "
            f"'{type(binary_operation.children[0]).__name__}' and "
            f"'{binary_operation.children[1].view()}' of type "
            f"'{type(binary_operation.children[1]).__name__}'."
        )

    if not (
        isinstance(binary_operation.children[0], UnaryOperation)
        or isinstance(binary_operation.children[1], UnaryOperation)
    ):
        raise TypeError(
            f"At least one of 'binary_operation' argument children should be "
            f"of type 'UnaryOperation' but found "
            f"'{type(binary_operation.children[0]).__name__}' and "
            f"'{type(binary_operation.children[1]).__name__}'."
        )
    if isinstance(binary_operation.children[0], UnaryOperation) and isinstance(
        binary_operation.children[1], UnaryOperation
    ):
        if not (
            (binary_operation.children[0].operator is UnaryOperation.Operator.MINUS)
            ^ (binary_operation.children[1].operator is UnaryOperation.Operator.MINUS)
        ):
            raise ValueError(
                f"Exactly one of 'binary_operation' argument children should have "
                f"operator 'UnaryOperation.Operator.MINUS' but found "
                f"'{binary_operation.children[0].operator}' and '{binary_operation.children[1].operator}'."
            )
    # for operand in binary_operation.children:
    #    if isinstance(operand, UnaryOperation):
    #        _typecheck_minus(operand)

    for i, operand in enumerate(binary_operation.children):
        if isinstance(operand, UnaryOperation) and (
            operand.operator is UnaryOperation.Operator.MINUS
        ):
            plus_operand = operand.children[0].copy()
            other_operand = binary_operation.children[1 - i].copy()
            new_operands = [plus_operand, other_operand]
            if i == 1:
                new_operands.reverse()
            new_binary_operation = BinaryOperation.create(
                binary_operation.operator, *new_operands
            )
            # minus_operation = UnaryOperation.create(
            #    UnaryOperation.Operator.MINUS, new_binary_operation
            # )
            return UnaryOperation.create(
                UnaryOperation.Operator.MINUS, new_binary_operation
            )
            # binary_operation.replace_with(minus_operation)

    raise ValueError(
        f"'simplify_mul_div_plus_minus_or_minus_plus' called on argument "
        f"{binary_operation.view()} encountered a problem "
        f"and should not have reached this point."
    )


def _typecheck_at_least_one_literal_operand(binary_operation):
    _typecheck_binary_operation(binary_operation)
    if not (
        isinstance(binary_operation.children[0], Literal)
        or isinstance(binary_operation.children[1], Literal)
    ):
        raise TypeError(
            f"At least one of 'binary_operation' argument children should be "
            f"of type 'Literal' but found "
            f"'{type(binary_operation.children[0]).__name__}' and "
            f"'{type(binary_operation.children[1]).__name__}'."
        )


def _valuecheck_at_least_one_literal_operand(binary_operation, values):
    _typecheck_binary_operation(binary_operation)
    if not isinstance(values, list):
        raise TypeError(
            f"'values' argument should be of type 'list[str]' but "
            f"found '{type(values).__name__}'."
        )
    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                f"'values' argument should be of type 'list[str]' "
                f"but found an element of type "
                f"'{type(value).__name__}'."
            )

    if isinstance(binary_operation.children[0], Literal) and isinstance(
        binary_operation.children[1], Literal
    ):
        if not (
            (binary_operation.children[0].value in values)
            or (binary_operation.children[1].value in values)
        ):
            raise ValueError(
                f"At least one of 'binary_operation' argument children should be "
                f"a Literal with value in {values} but found "
                f"'{binary_operation.children[0].value}' and '{binary_operation.children[1].value}'."
            )
    else:
        for i, operand in enumerate(binary_operation.children):
            if isinstance(operand, Literal):
                if operand.value not in values:
                    raise ValueError(
                        f"At least one of 'binary_operation' argument children should be "
                        f"a Literal with value in {values} but found "
                        f"'{binary_operation.children[0].view()}' and '{binary_operation.children[1].view()}'."
                    )


# x * 1 or 1 * x => x
def simplify_mul_by_one(binary_operation):
    """Simplifies a binary operation with operator MUL or DIV, one operand
    being of type Literal with value '1'.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not MUL or DIV.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
    """
    _typecheck_mul_div(binary_operation)
    _typecheck_at_least_one_literal_operand(binary_operation)

    _valuecheck_at_least_one_literal_operand(binary_operation, ["1"])

    for i, operand in enumerate(binary_operation.children):
        if isinstance(operand, Literal):
            # NOTE: not 1.0
            if operand.value == "1":
                return binary_operation.children[1 - i].copy()
                # binary_operation.replace_with(
                #    binary_operation.children[1 - i].copy()
                # )

    raise ValueError(
        f"'simplify_mul_by_one' called on argument "
        f"{binary_operation.view()} encountered a problem "
        f"and should not have reached this point."
    )


# x * 0 or 0 * x => 0
def simplify_mul_by_zero(binary_operation):
    """Simplifies a binary operation with operator MUL or DIV, one operand
    being of type Literal with value '0', '0.' or '0.0'.

    :param binary_operation: operation to simplify.
    :type binary_operation: :py:class:`psyclone.psyir.nodes.BinaryOperation`

    :raises TypeError: if binary_operation is of the wrong type.
    :raises ValueError: if the operator of binary_operation is not MUL or DIV.
    :raises ValueError: if the operands are not as described.

    :return: the simplified operation.
    :rtype: :py:class:`psyclone.psyir.nodes.DataNode`
    """
    _typecheck_mul_div(binary_operation)
    _typecheck_at_least_one_literal_operand(binary_operation)

    _valuecheck_at_least_one_literal_operand(binary_operation, ["0", "0.", "0.0"])

    for i, operand in enumerate(binary_operation.children):
        if isinstance(operand, Literal):
            if operand.value in ("0", "0.0", "0."):
                return Literal("0", INTEGER_TYPE)

    raise ValueError(
        f"'simplify_mul_by_zero' called on argument "
        f"{binary_operation.view()} encountered a problem "
        f"and should not have reached this point."
    )


def _typecheck_assignment(assignment):
    if not isinstance(assignment, Assignment):
        raise TypeError(
            f"'assignment' argument should be "
            f"of type 'Assignment' but found "
            f"'{type(assignment).__name__}'."
        )


def simplify_assignment(assignment):
    """Simplifies an assignment.
    Returns None for a simplified self-assignment whose original
    should be detached from the AST.

    :param assignment: assignment to simplify.
    :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`

    :raises TypeError: if assignment is of the wrong type.

    :return: the simplified assignment.
    :rtype: Union[:py:class:`psyclone.psyir.nodes.Assignment,
                  NoneType]`
    """
    _typecheck_assignment(assignment)

    if assignment.lhs == assignment.rhs:
        return simplify_self_assignment(assignment)

    return assignment


# x = x => None
def simplify_self_assignment(assignment):
    """Simplifies a self-assignment. Returns None.

    :param assignment: assignment to simplify.
    :type assignment: :py:class:`psyclone.psyir.nodes.Assignment`

    :raises TypeError: if assignment is of the wrong type.
    :raises TypeError: if the assignment lhs and rhs are not equal.

    :return: None
    :rtype: `NoneType`
    """
    _typecheck_assignment(assignment)

    if assignment.lhs != assignment.rhs:
        raise ValueError(
            f"'assignment' argument should satisfy "
            f"'assignment.lhs == assignment.rhs' but found "
            f"lhs '{assignment.lhs.view()}' and "
            f"rhs '{assignment.rhs.view()}'."
        )

    return None


def simplify_node(node, times=1):
    """Simplifies a node. If the node is not simplified, the returned node is
    still attached.

    :param node: node to simplify.
    :type node: :py:class:`psyclone.psyir.nodes.Node`
    :param times: number of times operation simplification should be
        applied, optional, default to 1.
    :type times: Optional[int]

    :raises TypeError: if node is of the wrong type.
    :raises TypeError: if times is of the wrong type.
    :raises ValueError: if times is less than 1.

    :return: the simplified node.
    :rtype: Union[:py:class:`psyclone.psyir.nodes.Node,
                  NoneType]`
    """
    if not isinstance(node, Node):
        raise TypeError(
            f"'node' argument should be "
            f"of type 'Node' but found "
            f"'{type(node).__name__}'."
        )
    if not isinstance(times, int):
        raise TypeError(
            f"'times' argument should be "
            f"of type 'int' but found "
            f"'{type(times).__name__}'."
        )
    if times < 1:
        raise ValueError(
            f"'times' argument should be at least '1' " f"but found '{times}'."
        )

    if isinstance(node, Assignment):
        return simplify_assignment(node)

    if isinstance(node, BinaryOperation):
        return simplify_binary_operation(node, times)

    return node
