from psyclone.psyir.nodes import Loop
from psyclone.errors import GenerationError
from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADDataNode, ADNode
from psyclone.autodiff.psyir.symbols import ADDataSymbol, ADVariableSymbol

class ADLoop(Loop, ADNode):
    _children_valid_format = "ADDataNode, ADDataNode, ADDataNode, ADSchedule"
    _text_name = "ADLoop"

    def __init__(self, variable=None, annotations=None, **kwargs):
        if not isinstance(variable, ADDataSymbol):
            raise TypeError("")
        super().__init__(variable, annotations, **kwargs)
        children = kwargs.get("children", None)
        parent = kwargs.get("children", None)
        self.__init_ad__(children, parent)

        # TODO
        # self._number_of_iterations = self.compute_number_of_iterations()
        # self._iteration_counter = self.compute_iteration_counter()

    def compute_number_of_iterations(self):
        # TODO
        raise NotImplementedError("")
    
    def compute_iteration_counter(self):
        # TODO
        raise NotImplementedError("")

    # @classmethod
    # def _check_variable(cls, variable):
    #     if not isinstance(variable, ADDataSymbol):
    #         try:
    #             variable_name = f"'{variable.name}'"
    #         except AttributeError:
    #             variable_name = "property"
    #         raise GenerationError(
    #             f"Variable {variable_name} in ADLoop class should be an "
    #             f"ADDataSymbol but found '{type(variable).__name__}'.")
    #     super()._check_variable(variable)

    @staticmethod
    def _validate_child(position, child):
        '''
        :param int position: the position to be validated.
        :param child: a child to be validated.
        :type child: :py:class:`psyclone.psyir.nodes.Node`

        :return: whether the given child and position are valid for this node.
        :rtype: bool

        '''
        from psyclone.autodiff.psyir.nodes import ADSchedule
        return (position in (0, 1, 2) and isinstance(child, ADDataNode)) or (
            position == 3 and isinstance(child, ADSchedule))
    
    @classmethod
    def from_psyir(cls, loop):
        if not isinstance(loop, Loop):
            raise TypeError("")
        if isinstance(loop, ADLoop):
            raise TypeError("")
        ad_loop_variable = ADVariableSymbol.from_psyir(loop.variable, is_loop_variable=True)
        ad_loop = cls(variable = ad_loop_variable, annotations = loop.annotations)
        for child in loop.children:
            if isinstance(child, ADDataNode):
                ad_loop.addchild(child)
            else:
                ad_loop.addchild(ADPSyIR.from_psyir(child))
        return ad_loop
    
    @classmethod
    def create(cls, variable, start, stop, step, children):
        loop = super().create(variable, start, stop, step, children)
        return cls.from_psyir(loop)