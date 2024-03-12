from psyclone.psyir.nodes import IfBlock

from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADNode, ADDataNode


class ADIfBlock(IfBlock, ADNode):
    def __init__(self, ast=None, children=None, parent=None, annotations=None):
        super().__init__(ast, children, parent, annotations)
        self.__init_ad__(children, parent)

    @classmethod
    def create(cls, if_condition, if_body, else_body=None):
        if_block = super().create(if_condition, if_body, else_body)
        return cls.from_psyir(if_block)

    @staticmethod
    def _validate_child(position, child):
        from psyclone.autodiff.psyir.nodes import ADSchedule

        return (position == 0 and isinstance(child, ADDataNode)) or (
            position in (1, 2) and isinstance(child, ADSchedule)
        )

    @classmethod
    def from_psyir(cls, if_block):
        if not isinstance(if_block, IfBlock):
            raise TypeError("")
        if isinstance(if_block, ADIfBlock):
            raise TypeError("")
        # NOTE: not parent, nor children, nor ast to recursively raise to AD
        ad_if_block = cls(annotations=if_block.annotations)
        for child in if_block.children:
            if isinstance(child, ADNode):
                ad_if_block.addchild(child)
            else:
                ad_if_block.addchild(ADPSyIR.from_psyir(child))
        return ad_if_block
