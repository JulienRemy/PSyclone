from psyclone.autodiff.psyir.nodes import ADAssignment, ADReference, ADBinaryOperation

class ADIncrement(ADAssignment):
    _children_valid_format = "ADReference, ADBinaryOperation"
    _text_name = "ADIncrement"

    @staticmethod
    def _validate_child(position, child):
        """
        :param int position: the position to be validated.
        :param child: a child to be validated.
        :type child: :py:class:`psyclone.psyir.nodes.Node`

        :return: whether the given child and position are valid for this node.
        :rtype: bool

        """
        return ((position == 0 and isinstance(child, ADReference))
                or (position == 0 and isinstance(child, ADBinaryOperation)))

    @classmethod
    def create(cls, lhs, rhs):
        if not isinstance(lhs, ADReference):
            raise TypeError("")
        if lhs.access is not ADReference.Access.INCREMENT:
            raise ValueError("")
        addition = ADBinaryOperation.create(ADBinaryOperation.Operator.ADD,
                                            lhs.copy(),
                                            rhs)
        ad_increment = cls()
        ad_increment.addchild(lhs, addition)
        return ad_increment

    def lower_to_language_level(self):
         super().lower_to_language_level()