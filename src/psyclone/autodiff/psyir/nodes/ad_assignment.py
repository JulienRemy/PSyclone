from psyclone.psyir.nodes import Assignment
from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADNode, ADDataNode

class ADAssignment(Assignment, ADNode):
    _children_valid_format = "ADDataNode, ADDataNode"
    _text_name = "ADAssignment"

    # TODO: activity, recursively defined on children

    def __init__(self, ast=None, children=None, parent=None, annotations=None):
        super().__init__(ast, children, parent, annotations)
        #if parent or children:
        #    raise GenerationError("Please create ADAssignment nodes through "
        #                          "the create class method.")
        self.__init_ad__(children, parent)
        if children:
            self._set_children_data_flow()

    def _set_children_data_flow(self):
            if len(self.children) != 2:
                raise ValueError("")
            for child in self.children:
                if not isinstance(child, ADDataNode):
                    raise TypeError("")
            self.children[0].backward_data_flow.append(self.children[1])
            self.children[1].forward_data_flow.append(self.children[0])

    def addchild(self, child, index=None):
        if not isinstance(child, ADDataNode):
            raise TypeError("")
        super().addchild(child, index)
        if len(self.children) == 2:
            self._set_children_data_flow()

    @staticmethod
    def _validate_child(position, child):
        '''
        :param int position: the position to be validated.
        :param child: a child to be validated.
        :type child: :py:class:`psyclone.psyir.nodes.Node`

        :return: whether the given child and position are valid for this node.
        :rtype: bool

        '''
        return position < 2 and isinstance(child, ADDataNode)
    
    @classmethod
    def from_psyir(cls, assignment):
        if not isinstance(assignment, Assignment):
            raise TypeError("")
        if isinstance(assignment, ADAssignment):
            raise TypeError("")
        # NOTE: not parent, nor children, nor ast to recursively raise to AD
        ad_assignment = cls(annotations = assignment.annotations)
        ad_children = []
        # Go through children right to left to give correct versions to references
        for child in assignment.children[::-1]:
            if isinstance(child, ADDataNode):
                ad_children.insert(0, child)
                #ad_assignment.addchild(child)
            else:
                ad_children.insert(0, ADPSyIR.from_psyir(child))
                #ad_assignment.addchild(ADPSyIR.from_psyir(child))
        for ad_child in ad_children:
            ad_assignment.addchild(ad_child)
        return ad_assignment
    
    @staticmethod
    def create(lhs, rhs):
        assignment = Assignment.create(lhs, rhs)
        return ADAssignment.from_psyir(assignment)