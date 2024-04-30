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

"""This module provides a Transformation for reverse-mode automatic 
differentiation of PSyIR Operation and IntrinsicCall nodes."""

from psyclone.psyir.nodes import (
    Assignment,
    Reference,
    Literal,
    UnaryOperation,
    BinaryOperation,
    Operation,
    IntrinsicCall,
    DataNode,
)
from psyclone.psyir.symbols import (
    INTEGER_TYPE,
    REAL_TYPE,
    ScalarType,
    ArrayType,
)
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADOperationTrans
from psyclone.autodiff import (
    one,
    minus,
    inverse,
    power,
    sqrt,
    log,
    mul,
    sub,
    add,
    increment,
    assign,
    sin,
    cos,
    square,
    div,
)


class ADReverseOperationTrans(ADOperationTrans):
    """A class for automatic differentation transformations of Operation and 
    IntrinsicCall nodes using reverse-mode.
    Requires an ADReverseRoutineTrans instance as context, where the adjoint \
    symbols can be found.
    This applies the chain rule to all operands and returns the recording and \
    returning motions that correspond.
    """

    def validate(self, operation, parent_adj, options=None):
        """Validates the arguments of the `apply` method.

        :param operation: operation Node to be transformed.
        :type operation: Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        :param parent_adj: datanode of the adjoint of the parent Node, where \
                           the parent node can be the LHS of an Assignment, 
                           an enclosing Operation, etc.
        :type parent_adj: :py:class:`psyclone.psyir.nodes.DataNode`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if operation is of the wrong type.
        :raises TransformationError: if parent_adj is of the wrong type.
        :raises TransformationError: if parent_adj is a Reference and \
                                     its symbol is not found among \
                                     the adjoint symbols of the contextual \
                                     `ADReverseRoutineTrans`.
        """
        # pylint: disable=arguments-renamed

        super().validate(operation, options)

        if not isinstance(parent_adj, DataNode):
            raise TransformationError(
                f"'parent_adj' argument should be a "
                f"PSyIR 'DataNode' but found '{type(parent_adj).__name__}'."
            )

        if (
            isinstance(parent_adj, Reference)
            and parent_adj.symbol not in self.routine_trans.adjoint_symbols
        ):
            raise TransformationError(
                f"'parent_adj.symbol' DataSymbol "
                f"'{parent_adj.name}' cannot be found "
                f"among the existing adjoint symbols."
            )

    def apply(self, operation, parent_adj, options=None):
        """Applies the transformation. This generates the returning motion \
        statements that increment adjoints as required for reverse-mode \
        automatic differentiation.

        The `parent_adj` argument is the `DataNode` of the adjoint of the \
        parent node of the `operation` node, to use in incrementing the \
        adjoints of the operands of `operation`: 
        - if the parent node is an `Assignment`, this is its LHS adjoint,
        - if the parent node is an `Operation` or `IntrinsicCall`, \
            this is the adjoint of `operation` itself.

        For all children of the Operation or IntrinsicCall node:
        - if they are Literal nodes, nothing is done, \
            there is no adjoint to increment,
        - if they are Reference nodes, the parent node 
        If some children of the Operation or IntrinsicCall node being \
        transformed are themselves Operation or IntrinsicCall nodes, \
        `apply` is used recursively.

        If this Operation/IntrinsicCall node has an Assignment node as \
        ancestor, this distinguishes between children Reference nodes that are \
        the LHS of the Assignment (in which case it is an **iterative** \
        assignment *eg* `a = a + ... * (a + ...)`) or not.

        The adjoints incrementations are returned as two lists:
        - this first contains the adjoint incrementations of \
            *non-iterative* type,
        - the second the adjoint incrementations of *iterative* type.

        Note that is only returns **returning motion** statements. Indeed \
        operation results are not recorded to the tape for now.

        | Options:
        | - bool 'verbose' : toggles preceding and inline comments around the \
                           adjoining of the operation in the returning motion.

        :param operation: operation Node to be transformed.
        :type operation: Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]
        :param parent_adj: datanode of the adjoint of the parent Node, where \
                           the parent node can be the LHS of an Assignment, \
                           an enclosing Operation or IntrinsicCall, etc.
        :type parent_adj: :py:class:`psyclone.psyir.nodes.DataNode`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: couple composed of the **returning** motion adjoints \
                 incrementations for the operation being transformed and \
                 the **iterative** incrementations to an ancestor assignment \
                 LHS adjoint if it exists.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Assignment`], \
                List[:py:class:`psyclone.psyir.nodes.Assignment`]
        """
        # pylint: disable=arguments-renamed, too-many-locals

        self.validate(operation, parent_adj, options)

        # verbose option adds comments to the first and last returning 
        # statements
        verbose = self.unpack_option("verbose", options)

        # List of Node corresponding to the returning motion
        returning = []

        # This is None if the operation is not on the RHS of an Assignment node
        ancestor_assignment = operation.ancestor(Assignment)
        # List storing incrementations to the LHS adjoint of the
        # ancestor assignment.
        # Used if the assignment is iterative. eg. var = operations(var, ...)
        assignment_lhs_adj_incr = []

        # Generate all local partial derivatives
        partials = self.differentiate(operation)

        # Special case: binary operation with only one reference
        # eg. x*x, etc.
        # NOTE: this is especially needed for assignments such as x = x*x
        # otherwise the incrementations to x_adj are done sequentially,
        # which is wrong

        # TODO: activity analysis

        first_operand = operation.children[0]
        if (first_operand.datatype.intrinsic is ScalarType.Intrinsic.REAL
            and
            (
                isinstance(operation, BinaryOperation)
                or (
                    isinstance(operation, IntrinsicCall)
                    and len(operation.children) == 2
                )
            )
            and isinstance(first_operand, Reference)
            and operation.children[1] == first_operand
        ):
            if first_operand in self.routine_trans.active_datanodes:
                adj = self.routine_trans.reference_to_differential_of(first_operand)

                parent_adj_mul = mul(parent_adj, add(partials[0], partials[1]))
                adj_incr = increment(adj, parent_adj_mul)

                # If this operand is the LHS of an Assignment of which this
                # operation node is a descendant, the incrementation to the adjoint
                # will be done last
                if (ancestor_assignment is not None) and (
                    ancestor_assignment.lhs == first_operand
                ):
                    assignment_lhs_adj_incr.append(adj_incr)

                # Otherwise incrementation can be done now
                else:
                    returning.append(adj_incr)

        # General case, go through the operands
        else:
            # Increment the adjoints of the operands where needed
            for operand, partial in zip(operation.children, partials):
                if operand not in self.routine_trans.active_datanodes:
                    print(f"Found passive operand {operand.debug_string()} in {operation.debug_string()}, skipping it.")
                    continue

                if isinstance(operand, Literal):
                    # If the operand is a Literal, it has no adjoint
                    # to increment
                    pass
                elif isinstance(operand, Reference):
                    # Non real values have no adjoints
                    if operand.datatype.intrinsic is not ScalarType.Intrinsic.REAL:
                        continue
                    # If the operand is a Reference, increment its adjoint
                    adj = self.routine_trans.reference_to_differential_of(
                        operand
                    )
                    parent_adj_mul = mul(parent_adj, partial)
                    if isinstance(adj.datatype, ScalarType) and isinstance(
                        parent_adj.datatype, ArrayType
                    ):
                        parent_adj_mul = IntrinsicCall.create(
                            IntrinsicCall.Intrinsic.SUM, [parent_adj_mul]
                        )
                    adj_incr = increment(adj, parent_adj_mul)

                    # If this operand is the LHS of an Assignment of which this
                    # operation node is a descendant, the incrementation to the
                    # adjoint will be done last
                    if (ancestor_assignment is not None) and (
                        ancestor_assignment.lhs == operand
                    ):
                        assignment_lhs_adj_incr.append(adj_incr)

                    # Otherwise incrementation can be done now
                    else:
                        returning.append(adj_incr)

                elif isinstance(operand, (Operation, IntrinsicCall)):
                    # If the operand is an Operation, its adjoints get passed
                    # as parent_adj of the recursive call

                    parent_adj_mul = mul(parent_adj, partial)

                    # then recursively apply the transformation and collect the
                    # statements
                    op_returning, op_lhs_adj_incr = self.apply(
                        operand, parent_adj_mul.copy(), options
                    )
                    returning.extend(op_returning)
                    assignment_lhs_adj_incr.extend(op_lhs_adj_incr)

                # TODO: function calls go here
                # elif isinstance(operand, Call):

                else:
                    raise NotImplementedError(
                        f"Transforming an Operation or IntrinsicCall with "
                        f"operand or argument of type "
                        f"'{type(operand).__name__}' is not yet supported."
                    )

        # Verbose mode adds comments to the first and last returning statements
        # TODO: writer should be initialization argument of the (container?)
        # transformation
        if verbose and len(returning) != 0:
            from psyclone.psyir.backend.fortran import FortranWriter

            fwriter = FortranWriter()
            src = fwriter(operation.copy())
            returning[0].preceding_comment = f"Adjoining {src}"
            returning[-1].inline_comment = f"Finished adjoining {src}"

        return returning, assignment_lhs_adj_incr

    def differentiate(self, operation):
        """Compute the local partial derivatives of every operand of the \
            operation argument.

        :param operation: operation or intrinsic call Node to be differentiated.
        :type operation: Union[:py:class:`psyclone.psyir.nodes.Operation`, \
                               :py:class:`psyclone.psyir.nodes.IntrinsicCall`]

        :raises TypeError: if operation is of the wrong type.

        :return: list of local partial derivatives of operation.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        super().differentiate(operation)

        if isinstance(operation, UnaryOperation):
            # Single element list
            return [self.differentiate_unary(operation)]

        if isinstance(operation, BinaryOperation):
            return self.differentiate_binary(operation)

        if isinstance(operation, IntrinsicCall):
            return self.differentiate_intrinsic(operation)

    def differentiate_unary(self, operation):
        """Compute the local derivative of the single operand of the \
            operation argument.

        :param operation: unary operation Node to be differentiated.
        :type operation: :py:class:`psyclone.psyir.nodes.UnaryOperation`

        :raises TypeError: if operation is of the wrong type.
        :raises NotImplementedError: if the operator derivative hasn't been \
                                     implemented yet.

        :return: local derivative of operation.
        :rtype: Union[:py:class:`psyclone.psyir.nodes.UnaryOperation`, \
                      :py:class:`psyclone.psyir.nodes.Literal`]
        """
        # pylint: disable=too-many-return-statements, too-many-branches

        super().differentiate_unary(operation)

        operator = operation.operator

        if operator == UnaryOperation.Operator.PLUS:
            return one(REAL_TYPE)
        if operator == UnaryOperation.Operator.MINUS:
            return minus(one(REAL_TYPE))

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

        :return: list of local partial derivatives of operation.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        super().differentiate_binary(operation)

        operator = operation.operator
        lhs, rhs = [child.copy() for child in operation.children]

        if operator == BinaryOperation.Operator.ADD:
            return [one(REAL_TYPE), one(REAL_TYPE)]
        if operator == BinaryOperation.Operator.SUB:
            return [one(REAL_TYPE), minus(one(REAL_TYPE))]
        if operator == BinaryOperation.Operator.MUL:
            return [rhs, lhs]
        if operator == BinaryOperation.Operator.DIV:
            return [inverse(rhs), minus(div(lhs, square(rhs)))]
        if operator == BinaryOperation.Operator.POW:
            if isinstance(rhs, Literal):
                whole, dot, decimal = rhs.value.partition(".")
                whole_minus_1 = str(int(whole) - 1)
                exponent = Literal(whole_minus_1 + dot + decimal, rhs.datatype)
            else:
                exponent = sub(rhs, one())
            return [mul(rhs, power(lhs, exponent)), mul(operation, log(lhs))]
            # TODO: should this take cases where derivatives are undefined
            # into account, like Tapenade does?

            # IF (x .LE. 0.0) THEN
            #    IF (y .EQ. 0.0 .OR. y .NE. INT(y))) THEN
            #        xb = 0.D0
            #    END IF
            #    yb = 0.D0
            # ELSE
            #    xb = y*x**(y-1)*zb
            #    yb = x**y*LOG(x)*zb
            # END IF

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
            "NEQV",
        ]
        raise NotImplementedError(
            f"Differentiating BinaryOperation with "
            f"operator '{operator}' is not implemented yet. "
            f"Not implemented Binary operators are "
            f"{_not_implemented}."
        )

    def differentiate_intrinsic(self, intrinsic_call):
        """Compute the local derivatives of the arguments of the intrinsic call.

        :param intrinsic_call: unary intrinsic_call Node to be differentiated.
        :type intrinsic_call: :py:class:`psyclone.psyir.nodes.IntrinsicCall`

        :raises TypeError: if intrinsic_call is of the wrong type.
        :raises NotImplementedError: if the intrinsic derivative hasn't been \
                                     implemented yet.

        :return: list of local partial derivatives of intrinsic_call.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        # pylint: disable=too-many-return-statements, too-many-branches

        super().differentiate_intrinsic(intrinsic_call)

        intrinsic = intrinsic_call.intrinsic
        arguments = [child.copy() for child in intrinsic_call.children]

        # Unary intrinsics
        if len(arguments) == 1:
            argument = arguments[0]

            if intrinsic == IntrinsicCall.Intrinsic.SQRT:
                # TODO: x=0 should print something,
                # raise an exception or something?
                return [
                    inverse(mul(Literal("2", INTEGER_TYPE), intrinsic_call))
                ]
            if intrinsic == IntrinsicCall.Intrinsic.EXP:
                return [intrinsic_call.copy()]
            if intrinsic == IntrinsicCall.Intrinsic.LOG:
                return [inverse(argument)]
            if intrinsic == IntrinsicCall.Intrinsic.LOG10:
                return [inverse(mul(argument, log(Literal("10.0", REAL_TYPE))))]
            if intrinsic == IntrinsicCall.Intrinsic.COS:
                return [minus(sin(argument))]
            if intrinsic == IntrinsicCall.Intrinsic.SIN:
                return [cos(argument)]
            if intrinsic == IntrinsicCall.Intrinsic.TAN:
                return [add(one(REAL_TYPE), square(intrinsic_call))]
                # return inverse(square(cos(argument)))
            if intrinsic == IntrinsicCall.Intrinsic.ACOS:
                return [
                    minus(inverse(sqrt(sub(one(REAL_TYPE), square(argument)))))
                ]
            if intrinsic == IntrinsicCall.Intrinsic.ASIN:
                return [inverse(sqrt(sub(one(REAL_TYPE), square(argument))))]
            if intrinsic == IntrinsicCall.Intrinsic.ATAN:
                return [inverse(add(one(REAL_TYPE), square(argument)))]
            if intrinsic == IntrinsicCall.Intrinsic.ABS:
                # This could also be implemented using an if block
                return [div(argument, intrinsic_call.copy())]
                # return sign(one(argument.datatype), argument)
            # if intrinsic == IntrinsicCall.Intrinsic.CEILING:
            #    # 0             if sin(pi * argument) == 0
            #    # undefined     otherwise...
            #    could return 0 but that's error prone

            raise NotImplementedError(
                f"Differentiating unary IntrinsicCall with "
                f"intrinsic '{intrinsic.name}' is not implemented yet. "
            )

        # Binary intrinsic call
        if len(arguments) == 2:
            lhs, rhs = arguments

            if intrinsic == IntrinsicCall.Intrinsic.DOT_PRODUCT:
                return [rhs, lhs]

            # - MATMUL(matrix, vector): to be implemented in reverse-mode
            #   using the new IntrinsicCall SPREAD operation from PR #1987
            #   to compute an outer product.
            #   NOTE: also requires getting 'parent_adj' argument from apply
            #         see c_adj below.
            #         => move 'parent_adj_mul' op. to differentiate_... functions.

            #   c = matmul(a, b)
            #   ! with b a matrix :
            #   a_adj = a_adj + MATMUL(c_adj, TRANSPOSE(b))
            #   ! with b a vector :
            #   a_adj = a_adj + SPREAD(b, dim=2, ncopies=SIZE(b))
            #                          * SPREAD(c_adj, dim=1, ncopies=SIZE(b))
            #   b_adj = b_adj + MATMUL(TRANSPOSE(a), c_adj)

            # TODO:
            # MIN if block
            # MAX if block
            # MATMUL pending

            raise NotImplementedError(
                f"Differentiating binary IntrinsicCall with "
                f"intrinsic '{intrinsic.name}' is not implemented yet. "
            )

        raise NotImplementedError(
            f"Differentiating IntrinsicCall with "
            f"intrinsic '{intrinsic.name}' is not implemented yet. "
            f"No intrinsics with arity larger than 2 have been implemented."
        )
