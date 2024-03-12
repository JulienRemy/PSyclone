from psyclone.psyir.nodes import Operation, UnaryOperation, BinaryOperation
from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADDataNode
from psyclone.autodiff.psyir.symbols import (
    ADDataSymbol,
    ADOperationAdjointSymbol,
)


class ADOperation(Operation, ADDataNode):
    # def __init__(self, operator, parent=None, motion=ADMotion.ADVANCING,
    #    advancing_node=None):
    #    super().__init__(operator, parent)
    #    ADDataNode.__init_ad__(motion, advancing_node)

    # TODO: activity, recursively defined on children

    def __init__(self, operator, parent=None):
        super().__init__(operator, parent)
        self.__init_ad__(parent=parent)
        if parent is not None:
            self.forward_data_flow.append(parent)
            parent.backward_data_flow.append(self)

        # NOTE: this is updated in the create classmethod of derived classes
        # as it requires knowing about the operation datatype
        self._operation_adjoint_symbol = None

    @property
    def operation_adjoint_symbol(self):
        return self._operation_adjoint_symbol

    def addchild(self, child, index=None):
        if not isinstance(child, ADDataNode):
            raise TypeError("")
        super().addchild(child, index)
        self.backward_data_flow.append(child)
        child.forward_data_flow.append(self)

    @classmethod
    def from_psyir(cls, operation):
        if not isinstance(operation, Operation):
            raise TypeError("")
        if isinstance(operation, ADOperation):
            raise TypeError("")
        # NOTE: not parent, this is applied recursively
        ad_operation = cls(operator=operation.operator)
        for child in operation.children:
            if isinstance(child, ADDataNode):
                ad_operation.addchild(child)
            else:
                ad_operation.addchild(ADPSyIR.from_psyir(child))

        return ad_operation


class ADUnaryOperation(UnaryOperation, ADOperation):
    _children_valid_format = "ADDataNode"

    # def __init__(self, operator, parent=None, motion=ADMotion.ADVANCING,
    #     advancing_node=None):
    #     super().__init__(operator, parent)
    #     ADOperation.__init_ad__(motion, advancing_node)

    def __init__(self, operator, parent=None):
        super().__init__(operator, parent)
        self.__init_ad__(parent=parent)
        if parent is not None:
            self.forward_data_flow.append(parent)
            parent.backward_data_flow.append(self)

    # def addchild(self, child, index=None):
    #     super().addchild(child, index)
    #     self.backward_data_flow.append(child)
    #     child.forward_data_flow.append(self)

    @staticmethod
    def _validate_child(position, child):
        """
        :param int position: the position to be validated.
        :param child: a child to be validated.
        :type child: :py:class:`psyclone.psyir.nodes.Node`

        :return: whether the given child and position are valid for this node.
        :rtype: bool

        """
        return position == 0 and isinstance(child, ADDataNode)

    @staticmethod
    def create(operator, operand):
        unary_operation = UnaryOperation.create(operator, operand)
        ad_op = ADUnaryOperation.from_psyir(unary_operation)
        # TODO: this could be a later ADTrans
        ad_op._operation_adjoint_symbol = ADOperationAdjointSymbol(ad_op)
        return ad_op

    # TODO: is_linear


class ADBinaryOperation(BinaryOperation, ADOperation):
    _children_valid_format = "ADDataNode, ADDataNode"

    # def __init__(self, operator, parent=None, motion=ADMotion.ADVANCING,
    #     advancing_node=None):
    #     super().__init__(operator, parent)
    #     ADOperation.__init_ad__(motion, advancing_node)

    def __init__(self, operator, parent=None):
        super().__init__(operator, parent)
        self.__init_ad__(parent=parent)
        if parent is not None:
            self.forward_data_flow.append(parent)
            parent.backward_data_flow.append(self)

    # def addchild(self, child, index=None):
    #     super().addchild(child, index)
    #     self.backward_data_flow.append(child)
    #     child.forward_data_flow.append(self)

    @staticmethod
    def _validate_child(position, child):
        """
        :param int position: the position to be validated.
        :param child: a child to be validated.
        :type child: :py:class:`psyclone.psyir.nodes.Node`

        :return: whether the given child and position are valid for this node.
        :rtype: bool

        """
        return position in (0, 1) and isinstance(child, ADDataNode)

    @staticmethod
    def create(operator, lhs, rhs):
        binary_operation = BinaryOperation.create(operator, lhs, rhs)
        ad_op = ADBinaryOperation.from_psyir(binary_operation)
        # TODO: this could be a later ADTrans
        ad_op._operation_adjoint_symbol = ADOperationAdjointSymbol(ad_op)
        return ad_op

    # TODO: is_linear
