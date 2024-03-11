from types import NoneType

from psyclone.psyir.nodes import DataNode
from psyclone.autodiff.psyir.nodes import ADNode


class ADDataNode(DataNode, ADNode):
    # def __init__(
    #     self,
    #     ast=None,
    #     children=None,
    #     parent=None,
    #     annotations=None,
    #     #motion=ADMotion.ADVANCING,
    #     #advancing_node=None,
    # ):
    #     super().__init__(ast, children, parent, annotations)
    #     #self.__init_ad__(motion, advancing_node)
    #     self.__init_ad__()

    # def __init_ad__(self, motion=ADMotion.ADVANCING, advancing_node=None):
    #     ADNode.__init_ad__(motion, advancing_node)

    def __init__(self, ast=None, children=None, parent=None, annotations=None):
        super().__init__(ast, children, parent, annotations)
        self.__init_ad__(children, parent)

    def __init_ad__(self, children=None, parent=None):
        super().__init_ad__(children, parent)

        self._forward_data_flow = []
        self._backward_data_flow = []

    @property
    def forward_data_flow(self):
        return self._forward_data_flow

    @property
    def backward_data_flow(self):
        return self._backward_data_flow

    def depends_on(self, my_type, linearity=None): # stop_type=None,):
        if not issubclass(my_type, ADNode):
            raise TypeError(f"{my_type.__name__}")
        # if stop_type and not issubclass(stop_type, ADNode):
        #     raise TypeError("")
        if linearity and not isinstance(linearity, bool):
            raise TypeError("")
        
        from psyclone.autodiff.psyir.nodes import ADOperation

        depends_on = []
        for bwd in self.backward_data_flow:
            if isinstance(bwd, my_type):# and bwd not in depends_on:
                depends_on.append(bwd)

            elif isinstance(bwd, (ADOperation)):#, ADCall)):
                if (linearity is None) or (bwd.linearity is linearity):
                    recurse = bwd.depends_on(my_type, linearity)
                    depends_on.extend(recurse)
                    # for node in recurse:
                    #     if node not in depends_on:
                    #         depends_on.append(node)

            #if stop_type and not isinstance(bwd, stop_type):
            #    if isinstance(bwd, (ADOperation)):#, ADCall)):
            #        if (linearity is not None) and (bwd.linearity is linearity):
            #            depends_on.extend(bwd.depends_on(my_type, stop_type, linearity))
            #    else:
            #        depends_on.extend(bwd.depends_on(my_type, stop_type, linearity))

        return depends_on

    def enters_in(self, my_type, linearity=None):#, stop_type=None):
        if not issubclass(my_type, ADNode):
            raise TypeError(f"{my_type.__name__}")
        # if stop_type and not issubclass(stop_type, ADNode):
        #     raise TypeError("")
        if linearity and not isinstance(linearity, bool):
            raise TypeError("")
        
        from psyclone.autodiff.psyir.nodes import ADOperation

        enters_in = []
        for fwd in self.forward_data_flow:
            if isinstance(fwd, my_type):# and fwd not in enters_in:
                enters_in.append(fwd)
            
            elif isinstance(fwd, (ADOperation)):#, ADCall)):
                if (linearity is None) or (fwd.linearity is linearity):
                    recurse = fwd.enters_in(my_type, linearity)
                    enters_in.extend(recurse)
                    # for node in recurse:
                    #     if node not in enters_in:
                    #         enters_in.append(node)
                    #enters_in.extend(fwd.enters_in(my_type, linearity))

            # if stop_type and not isinstance(fwd, stop_type):
            #     if isinstance(fwd, (ADOperation)):#, ADCall)):
            #         if (linearity is not None) and (fwd.linearity is linearity):
            #             enters_in.extend(fwd.enters_in(my_type, stop_type, linearity))
            #     else:
            #         enters_in.extend(fwd.enters_in(my_type, stop_type, linearity))

        return enters_in

    @property
    def size(self):
        # TODO: compute size as in ADTape, depending on datatype
        raise NotImplementedError("")

    # TODO: ADConfig this
    def TBR_status_for_variables(self, dependent_vars, independent_vars):
        # TODO
        raise NotImplementedError("")

    # TODO: ADConfig this
    def is_to_be_recorded(self, dependent_vars, independent_vars):
        # TODO
        raise NotImplementedError("")

    @property
    def python_tape(self):
        # TODO
        raise NotImplementedError("")

    @property
    def fortran_tape_array(self):
        # TODO
        raise NotImplementedError("")

    @property
    def fortran_tape_indices(self):
        # TODO
        raise NotImplementedError("")

    @property
    def fortran_tape_slice(self):
        # TODO
        raise NotImplementedError("")

    # TODO: ADConfig this
    def is_to_be_recomputed(self, dependent_vars, independent_vars):
        # TODO
        raise NotImplementedError("")

    @property
    def recomputing_STH(self):
        # TODO
        raise NotImplementedError("")
