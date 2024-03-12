from psyclone.psyir.nodes import Statement

from psyclone.autodiff.psyir.nodes import ADNode  # , ADCall, ADIntrinsicCall


class ADStatement(Statement, ADNode):
    def __init__(self, ast=None, children=None, parent=None, annotations=None):
        super().__init__(ast, children, parent, annotations)
        self.__init_ad__(children, parent)
