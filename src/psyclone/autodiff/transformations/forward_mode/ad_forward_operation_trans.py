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
    """A class for automatic differentation transformations of Operation or \
    IntrinsicCall nodes using forward-mode.
    Requires an ADForwardRoutineTrans instance as context, where the \
    derivative symbols can be found.
    This applies the chain rule to all operands and returns the derivative.
    """

    def apply(self, operation, options=None):
        """Applies the transformation. This generates the derivative operation.
        If some children of the Operation node being transformed are themselves \
        Operation nodes, `apply` is used recursively.

        :param operation: operation Node to be transformed.
        :type operation: Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
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
        :type operation: :Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]

        :raises TypeError: if operation is of the wrong type.

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

        if isinstance(operation, IntrinsicCall):
            return self.differentiate_intrinsic(operation)

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
                      :py:class:`psyclone.psyir.nodes.Operation`,
                      :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
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

        if isinstance(operand, (Operation, IntrinsicCall)):
            operand_d = self.apply(operand)

        if operator == UnaryOperation.Operator.PLUS:
            return operand_d
        if operator == UnaryOperation.Operator.MINUS:
            return minus(operand_d)

        _not_implemented = ["NOT"]
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
            else:  # Operation or IntrinsicCall
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

        # TODO:
        # REM? undefined for some values of lhs/rhs

        _not_implemented = [
            "REM",
            "EQ",
            "NE",
            "GT",
            "LT",
            "GE",
            "LE",
            "AND",
            "OR",
            "EQV", 
            "NEQV"
        ]
        raise NotImplementedError(
            f"Differentiating BinaryOperation with "
            f"operator '{operator}' is not implemented yet. "
            f"Not implemented Binary operators are "
            f"{_not_implemented}."
        )


    def differentiate_intrinsic(self, intrinsic_call):
        """Compute the derivative of an intrinsic call.

        :param intrinsic_call: intrinsic call Node to be differentiated.
        :type intrinsic_call: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :raises TypeError: if intrinsic is of the wrong type.
        :raises NotImplementedError: if the intrinsic derivative hasn't been \
                                     implemented yet.

        :return: derivative.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.Literal`,
                      :py:class:`psyclone.psyir.nodes.Reference`,
                      :py:class:`psyclone.psyir.nodes.Operation`,
                      :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        """
        # pylint: disable=unbalanced-tuple-unpacking
        # pylint: disable=too-many-return-statements, too-many-branches
        super().differentiate_intrinsic(intrinsic_call)

        intrinsic = intrinsic_call.intrinsic
        arguments = [child.copy() for child in intrinsic_call.children]

        # Unary intrinsics
        if len(arguments) == 1:
            argument = arguments[0]

            if isinstance(argument, Literal):
                return [zero()]

            if isinstance(argument, Reference):
                # argument_d_sym = self.routine_trans.\
                #                        data_symbol_differential_map[
                #     argument.symbol
                # ]
                # argument_d = Reference(argument_d_sym)
                argument_d = self.routine_trans.\
                                reference_to_differential_of(argument)

            if isinstance(argument, (Operation, IntrinsicCall)):
                argument_d = self.apply(argument)

            if intrinsic == IntrinsicCall.Intrinsic.SQRT:
                # TODO: x=0 should print something,
                # raise an exception or something?
                return div(argument_d,
                           mul(Literal("2", INTEGER_TYPE), intrinsic_call))
            if intrinsic == IntrinsicCall.Intrinsic.EXP:
                return mul(intrinsic_call, argument_d)
            if intrinsic == IntrinsicCall.Intrinsic.LOG:
                return div(argument_d, argument)
            if intrinsic == IntrinsicCall.Intrinsic.LOG10:
                return div(
                    argument_d, mul(argument, log(Literal("10.0", REAL_TYPE)))
                )
            if intrinsic == IntrinsicCall.Intrinsic.COS:
                return mul(minus(sin(argument)), argument_d)
            if intrinsic == IntrinsicCall.Intrinsic.SIN:
                return mul(cos(argument), argument_d)
            if intrinsic == IntrinsicCall.Intrinsic.TAN:
                return mul(add(one(REAL_TYPE),
                               square(intrinsic_call)),
                               argument_d)
                # return div(argument_d, square(cos(argument)))
            if intrinsic == IntrinsicCall.Intrinsic.ACOS:
                return minus(
                    div(argument_d, sqrt(sub(one(REAL_TYPE),
                                             square(argument))))
                )
            if intrinsic == IntrinsicCall.Intrinsic.ASIN:
                return div(argument_d, sqrt(sub(one(REAL_TYPE),
                                                square(argument))))
            if intrinsic == IntrinsicCall.Intrinsic.ATAN:
                return div(argument_d, add(one(REAL_TYPE), square(argument)))
            if intrinsic == IntrinsicCall.Intrinsic.ABS:
                # This could also be implemented using an if block
                return mul(div(argument, intrinsic_call.copy()), argument_d)
                # NOTE: version belowed caused large errors compared to Tapenade
                # as argument * ... / abs(argument)
                # != argument / abs(argument) * ...
                # return div(mul(argument, argument_d), intrinsic_call.copy())

            # if intrinsic == IntrinsicCall.Intrinsic.CEILING:
            #    # 0             if sin(pi * argument) == 0
            #    # undefined     otherwise...
            #    could return 0 but that's error prone

            raise NotImplementedError(
            f"Differentiating unary IntrinsicCall with "
            f"intrinsic '{intrinsic.name}' is not implemented yet."
        )

        # Binary intrinsics
        if len(arguments) == 2:

            super().differentiate_intrinsic(intrinsic_call)

            arguments_d = []
            for argument in arguments:
                if isinstance(argument, Literal):
                    arguments_d.append(zero())
                elif isinstance(argument, Reference):
                    argument_d = self.routine_trans.\
                                    reference_to_differential_of(argument)
                    arguments_d.append(argument_d)
                else:  # Operation or IntrinsicCall
                    arguments_d.append(self.apply(argument))

            lhs, rhs = arguments
            lhs_d, rhs_d = arguments_d

            if intrinsic == IntrinsicCall.Intrinsic.DOT_PRODUCT:
                return IntrinsicCall.create(IntrinsicCall.Intrinsic.SUM,
                                            [add(mul(rhs, lhs_d),
                                                 mul(lhs, rhs_d))])
            if intrinsic == IntrinsicCall.Intrinsic.MATMUL:
                return add(IntrinsicCall.create(IntrinsicCall.Intrinsic.MATMUL,
                                                [lhs_d, rhs]),
                        IntrinsicCall.create(IntrinsicCall.Intrinsic.MATMUL,
                                                [lhs, rhs_d]))

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
                # lhs_le_0 = IntrinsicCall.create(IntrinsicCall.Intrinsic.LE,
                #                                  [lhs.copy(),
                #                                  zero(REAL_TYPE)])
                # rhs_eq_0 = IntrinsicCall.create(IntrinsicCall.Intrinsic.EQ,
                #                                  [rhs.copy(),
                #                                  zero(REAL_TYPE)])
                # int_rhs = IntrinsicCall.create(IntrinsicCall.Intrinsic.INT,
                #                                [rhs.copy()])
                # rhs_ne_int_rhs = IntrinsicCall.create(IntrinsicCall.Intrinsic.NE,
                #                                        [rhs.copy(),
                #                                        int_rhs])
                # condition = IntrinsicCall.create(IntrinsicCall.Intrinsic.OR,
                #                                   [rhs_eq_0,
                #                                   rhs_ne_int_rhs])
                #
                # dummy_sym = DataSymbol("DUMMY___", REAL_TYPE)
                # dummy_d_zero = assign_zero(dummy_sym)
                # dummy_d_1_rhs = mul(mul(rhs, power(lhs, exponent)), lhs_d)
                # dummy_d_1 = assign(dummy_sym, dummy_d_1_rhs)
                # if_block_1 = IfBlock.create(condition,
                #                            [dummy_d_zero],
                #                            [dummy_d_1])
                #
                # log_lhs = IntrinsicCall.create(IntrinsicCall.Intrinsic.LOG,
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
            # MIN if block
            # MAX if block

            raise NotImplementedError(
            f"Differentiating binary IntrinsicCall with "
            f"intrinsic '{intrinsic.name}' is not implemented yet."
            )

        raise NotImplementedError(
        f"Differentiating IntrinsicCall with "
        f"intrinsic '{intrinsic.name}' is not implemented yet. "
        f"No intrinsics or arity larger than 2 have been implemented."
        )
