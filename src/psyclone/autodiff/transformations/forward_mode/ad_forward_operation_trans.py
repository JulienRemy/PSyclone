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

"""This module provides a Transformation for forward-mode automatic 
differentiation of PSyIR Operation nodes."""

from psyclone.psyir.nodes import (
    Reference,
    Literal,
    UnaryOperation,
    BinaryOperation,
    Operation,
    IntrinsicCall
)
from psyclone.psyir.symbols import INTEGER_TYPE, REAL_TYPE

from psyclone.autodiff.transformations import ADOperationTrans
from psyclone.autodiff import (
    one,
    minus,
    power,
    sqrt,
    log,
    mul,
    sub,
    add,
    sin,
    cos,
    square,
    div,
    zero,
)


class ADForwardOperationTrans(ADOperationTrans):
    """A class for automatic differentation transformations of Operation nodes \
    using forward-mode.
    Requires an ADForwardRoutineTrans instance as context, where the \
    derivative symbols can be found.
    This applies the chain rule to all operands and returns the derivative.
    """

    def apply(self, operation, options=None):
        """Applies the transformation. This generates the derivative operation.
        If some children of the Operation node being transformed are themselves \
        Operation nodes, `apply` is used recursively.

        :param operation: operation Node to be transformed.
        :type operation: :py:class:`psyclone.psyir.nodes.Operation`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: derivative.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`,
                      :py:class:`psyclone.psyir.nodes.Reference`,
                      :py:class:`psyclone.psyir.nodes.Operation`,
                      :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        """
        self.validate(operation, options)

        return self.differentiate(operation)

    def differentiate(self, operation):
        """Compute the derivative of the operation argument.

        :param operation: operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.Operation`

        :raises TypeError: if operation is of the wrong type.
        :raises NotImplementedError: if operation is an NaryOperation instance.

        :return: derivative.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`,
                      :py:class:`psyclone.psyir.nodes.Reference`,
                      :py:class:`psyclone.psyir.nodes.Operation`,
                      :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        """
        super().differentiate(operation)

        if isinstance(operation, UnaryOperation):
            return self.differentiate_unary(operation)

        if isinstance(operation, BinaryOperation):
            return self.differentiate_binary(operation)

    def differentiate_unary(self, operation):
        """Compute the derivative of unary operation.

        :param operation: unary operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.UnaryOperation`

        :raises TypeError: if operation is of the wrong type.
        :raises NotImplementedError: if the operator derivative hasn't been \
                                     implemented yet.

        :return: derivative.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`,
                      :py:class:`psyclone.psyir.nodes.Reference`,
                      :py:class:`psyclone.psyir.nodes.Operation`]
        """
        super().differentiate_unary(operation)

        operator = operation.operator
        operand = operation.children[0].copy()

        if isinstance(operand, Literal):
            return zero()

        if isinstance(operand, Reference):
            # operand_d_sym = self.routine_trans.data_symbol_differential_map[
            #     operand.symbol
            # ]
            # operand_d = Reference(operand_d_sym)
            operand_d = self.routine_trans.reference_to_differential_of(operand)

        if isinstance(operand, Operation):
            operand_d = self.apply(operand)

        if operator == UnaryOperation.Operator.PLUS:
            return operand_d
        if operator == UnaryOperation.Operator.MINUS:
            return minus(operand_d)
        if operator == UnaryOperation.Operator.SQRT:
            # TODO: x=0 should print something, raise an exception or something?
            return div(operand_d, mul(Literal("2", INTEGER_TYPE), operation))
        if operator == UnaryOperation.Operator.EXP:
            return mul(operation, operand_d)
        if operator == UnaryOperation.Operator.LOG:
            return div(operand_d, operand)
        if operator == UnaryOperation.Operator.LOG10:
            return div(
                operand_d, mul(operand, log(Literal("10.0", REAL_TYPE)))
            )
        if operator == UnaryOperation.Operator.COS:
            return mul(minus(sin(operand)), operand_d)
        if operator == UnaryOperation.Operator.SIN:
            return mul(cos(operand), operand_d)
        if operator == UnaryOperation.Operator.TAN:
            return mul(add(one(REAL_TYPE), square(operation)), operand_d)
            # return div(operand_d, square(cos(operand)))
        if operator == UnaryOperation.Operator.ACOS:
            return minus(
                div(operand_d, sqrt(sub(one(REAL_TYPE), square(operand))))
            )
        if operator == UnaryOperation.Operator.ASIN:
            return div(operand_d, sqrt(sub(one(REAL_TYPE), square(operand))))
        if operator == UnaryOperation.Operator.ATAN:
            return div(operand_d, add(one(REAL_TYPE), square(operand)))
        if operator == UnaryOperation.Operator.ABS:
            # This could also be implemented using an if block
            return mul(div(operand, operation.copy()), operand_d)
            # NOTE: version belowed caused large errors compared to Tapenade
            # as operand * ... / abs(operand) != operand / abs(operand) * ...
            # return div(mul(operand, operand_d), operation.copy())

        # if operator == UnaryOperation.Operator.CEIL:
        #    # 0             if sin(pi * operand) == 0
        #    # undefined     otherwise...
        #    could return 0 but that's error prone

        _not_implemented = ["NOT", "CEIL", "REAL", "INT", "NINT"]
        raise NotImplementedError(
            f"Differentiating UnaryOperation with "
            f"operator '{operator}' is not implemented yet. "
            f"Not implemented Unary operators are "
            f"{_not_implemented}."
        )

    def differentiate_binary(self, operation):
        """Compute the local partial derivatives of both operands of the \
        operation argument.

        :param operation: binary operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.BinarOperation`

        :raises TypeError: if operation is of the wrong type.
        :raises NotImplementedError: if the operator derivative hasn't been \
                                     implemented yet.

        :return: derivative.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.BinaryOperation`,
                      :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        """
        # pylint: disable=too-many-locals, unbalanced-tuple-unpacking

        super().differentiate_binary(operation)

        operator = operation.operator
        operands = [child.copy() for child in operation.children]

        operands_d = []
        for operand in operands:
            if isinstance(operand, Literal):
                operands_d.append(zero())
            elif isinstance(operand, Reference):
                operand_d \
                    = self.routine_trans.reference_to_differential_of(operand)
                operands_d.append(operand_d)
            else:  # Operation
                operands_d.append(self.apply(operand))

        lhs, rhs = operands
        lhs_d, rhs_d = operands_d

        if operator == BinaryOperation.Operator.ADD:
            return add(lhs_d, rhs_d)
        if operator == BinaryOperation.Operator.SUB:
            return sub(lhs_d, rhs_d)
        if operator == BinaryOperation.Operator.MUL:
            return add(mul(lhs_d, rhs), mul(rhs_d, lhs))
        if operator == BinaryOperation.Operator.DIV:
            return div(sub(lhs_d, div(mul(rhs_d, lhs), rhs)), rhs)
        if operator == BinaryOperation.Operator.POW:
            if isinstance(rhs, Literal):
                whole, dot, decimal = rhs.value.partition(".")
                whole_minus_1 = str(int(whole) - 1)
                exponent = Literal(whole_minus_1 + dot + decimal, rhs.datatype)
            else:
                exponent = sub(rhs, one())
            return add(
                mul(lhs_d, mul(rhs, power(lhs, exponent))),
                mul(rhs_d, mul(operation, log(lhs))),
            )
        if operator == BinaryOperation.Operator.DOT_PRODUCT:
            return IntrinsicCall.create(IntrinsicCall.Intrinsic.SUM,
                                        [add(mul(rhs, lhs_d), mul(lhs, rhs_d))])
        if operator == BinaryOperation.Operator.MATMUL:
            return add(BinaryOperation.create(BinaryOperation.Operator.MATMUL,
                                              lhs_d, rhs),
                       BinaryOperation.create(BinaryOperation.Operator.MATMUL,
                                              lhs, rhs_d))

            # TODO: should POW, SQRT, etc. take into account non-derivability ?
            # like Tapenade does?

            # IF (lhs .LE. 0.0)
            #    IF (rhs .EQ. 0.0 .OR. rhs .NE. INT(rhs))) THEN
            #        assigned_var_d = 0.D0
            #    ELSE
            #        assigned_var_d = rhs*lhs**(rhs-1)*lhs_d
            #    ENDIF
            # ELSE
            #    assigned_var_d = rhs*lhs**(rhs-1)*lhs_d + lhs**rhs*LOG(lhs)*rhs_d
            # END IF
            #
            # lhs_le_0 = BinaryOperation.create(BinaryOperation.Operator.LE,
            #                                  lhs.copy(),
            #                                  zero(REAL_TYPE))
            # rhs_eq_0 = BinaryOperation.create(BinaryOperation.Operator.EQ,
            #                                  rhs.copy(),
            #                                  zero(REAL_TYPE))
            # int_rhs = UnaryOperation.create(UnaryOperation.Operator.INT,
            #                                rhs.copy())
            # rhs_ne_int_rhs = BinaryOperation.create(BinaryOperation.Operator.NE,
            #                                        rhs.copy(),
            #                                        int_rhs)
            # condition = BinaryOperation.create(BinaryOperation.Operator.OR,
            #                                   rhs_eq_0,
            #                                   rhs_ne_int_rhs)
            #
            # dummy_sym = DataSymbol("DUMMY___", REAL_TYPE)
            # dummy_d_zero = assign_zero(dummy_sym)
            # dummy_d_1_rhs = mul(mul(rhs, power(lhs, exponent)), lhs_d)
            # dummy_d_1 = assign(dummy_sym, dummy_d_1_rhs)
            # if_block_1 = IfBlock.create(condition,
            #                            [dummy_d_zero],
            #                            [dummy_d_1])
            #
            # log_lhs = UnaryOperation.create(UnaryOperation.Operator.LOG,
            #                                lhs.copy())
            # dummy_d_2 = assign(dummy_sym,
            #                   add(dummy_d_1_rhs,
            #                       mul(mul(operation,
            #                               log_lhs),
            #                           rhs_d)))
            # if_block_2 = IfBlock.create(lhs_le_0,
            #                            [if_block_1],
            #                            [dummy_d_2])
            # return if_block_2

        # TODO:
        # REM? undefined for some values of lhs/rhs
        # MIN if block
        # MAX if block

        _not_implemented = [
            "REM",
            "MIN",
            "MAX",
            "EQ",
            "NE",
            "GT",
            "LT",
            "GE",
            "LE",
            "AND",
            "OR",
            "SIGN",
            "REAL",
            "INT",
            "CAST",
            "SIZE",
            "LBOUND",
            "UBOUND",
        ]
        raise NotImplementedError(
            f"Differentiating BinaryOperation with "
            f"operator '{operator}' is not implemented yet. "
            f"Not implemented Binary operators are "
            f"{_not_implemented}."
        )

    # TODO: implement these
    # @abstractmethod
    # def differentiate_nary_operation(self, operation):
    # MAX
    # MIN
