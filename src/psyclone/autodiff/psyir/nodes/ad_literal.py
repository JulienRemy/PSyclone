from psyclone.psyir.nodes import Literal
from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADDataNode

class ADLiteral(Literal, ADDataNode):
    _text_name = "ADLiteral"

    def __init__(self, value, datatype, parent=None):
        super().__init__(value, datatype, parent)
        self.__init_ad__(parent = parent)

    @classmethod
    def from_psyir(cls, literal):
        if not isinstance(literal, Literal):
            raise TypeError("")
        if isinstance(literal, ADLiteral):
            raise TypeError("")
        # NOTE: not parent to recursively apply this
        ad_literal = cls(literal.value, literal.datatype)
        for child in literal.children:
            if isinstance(child, ADDataNode):
                ad_literal.addchild(child)
            else:
                ad_literal.addchild(ADPSyIR.from_psyir(child))
        return ad_literal