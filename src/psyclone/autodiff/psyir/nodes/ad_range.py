from psyclone.psyir.nodes import Range
from psyclone.psyir.symbols import INTEGER_TYPE
from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADDataNode, ADLiteral, ADNode
from psyclone.autodiff.psyir.symbols import (
    ADDataSymbol,
)


class ADRange(Range, ADDataNode):
    _children_valid_format = "ADDataNode, ADDataNode, ADDataNode"
    _text_name = "ADRange"

    def __init__(self, ast=None, children=None, parent=None, annotations=None):
        super().__init__(ast, children, parent, annotations)
        self.__init_ad__(children, parent)

    @staticmethod
    def _validate_child(position, child):
        return position < 3 and isinstance(child, ADDataNode)

    def addchild(self, child, index=None):
        if not isinstance(child, ADDataNode):
            raise TypeError("")
        super().addchild(child, index)
        self.backward_data_flow.append(child)
        child.forward_data_flow.append(self)

    @classmethod
    def from_psyir(cls, psyir_range):
        if not isinstance(psyir_range, Range):
            raise TypeError("")
        if isinstance(psyir_range, ADRange):
            raise TypeError("")
        # NOTE: not parent, this is applied recursively
        bounds = (psyir_range.start, psyir_range.stop, psyir_range.step)
        ad_bounds = []
        for bound in bounds:
            if isinstance(bound, ADNode):
                ad_bounds.append(bound)
            else:
                ad_bounds.append(ADPSyIR.from_psyir(bound))
        return cls.create(ad_bounds[0], ad_bounds[1], ad_bounds[2])

    @classmethod
    def create(cls, start, stop, step=None):
        ad_range = cls()
        if step is None:
            step = ADLiteral("1", INTEGER_TYPE)
        for bound in (start, stop, step):
            ad_range.addchild(bound)
        ad_range._check_completeness()
        return ad_range

    @classmethod
    def _check_valid_input(cls, value, name):
        if not isinstance(value, (ADNode, ADLiteral)):
            raise TypeError("")
        super()._check_valid_input(value, name)
