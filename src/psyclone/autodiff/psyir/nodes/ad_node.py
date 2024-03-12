from types import NoneType
from enum import Enum

from psyclone.psyir.nodes import Node
from psyclone.autodiff.psyir import ADMotion, ADPSyIR


class ADNode(Node, ADPSyIR):
    def __init__(
        self,
        ast=None,
        children=None,
        parent=None,
        annotations=None,
        # motion=ADMotion.ADVANCING,
        # advancing_node=None,
    ):
        super().__init__(ast, children, parent, annotations)
        # self.__init_ad__(motion, advancing_node)
        self.__init_ad__(parent, children)

    # def __init_ad__(self, motion=ADMotion.ADVANCING, advancing_node=None):
    #    if (not isinstance(motion, Enum)) or (motion not in ADMotion):
    #        raise TypeError("")
    #    if not isinstance(advancing_node, (ADNode, NoneType)):
    #        raise TypeError("")
    #    if (motion is not ADMotion.ADVANCING) and (advancing_node is None):
    #        raise ValueError("")
    #
    #    self._motion = motion
    #
    #    if motion is ADMotion.ADVANCING:
    #        self._advancing_node = self
    #        self._tangent_nodes = []
    #        self._recording_nodes = []
    #        self._returning_nodes = []
    #    else:
    #        self._advancing_node = advancing_node
    #        if motion is ADMotion.TANGENT:
    #            self.advancing_node.tangent_nodes.append(self)
    #        elif motion is ADMotion.RECORDING:
    #            self.advancing_node.recording_nodes.append(self)
    #        elif motion is ADMotion.RETURNING:
    #            self.advancing_node.returning_nodes.append(self)
    #
    #    # TODO
    #    self._activity_analysis = None
    #
    #    self._forward_data_flow = []
    #    self._backward_data_flow = []

    def __init_ad__(self, children=None, parent=None):
        if parent is not None and not isinstance(parent, ADNode):
            raise TypeError(
                f"The parent of an ADNode must also be an ADNode "
                f"but got '{type(parent).__name__}'"
            )
        if children is not None:
            for child in children:
                if not isinstance(child, ADNode):
                    raise TypeError(
                        f"Every child of an ADNode must also be an "
                        f"ADNode but got '{type(child).__name__}'"
                    )
        # TODO
        self._activity_analysis = None

    @property
    def activity_analysis(self):
        return self._activity_analysis

    def is_active_for_variables(self, dependent_vars, independent_vars):
        # TODO
        raise NotImplementedError("")


#    @property
#    def motion(self):
#        return self._motion
#
#    @property
#    def advancing_node(self):
#        if self.motion is ADMotion.ADVANCING:
#            return self._advancing_node
#
#        return self.advancing_node.advancing_node
#
#    @property
#    def tangent_nodes(self):
#        if self.motion is ADMotion.ADVANCING:
#            return self._tangent_nodes
#
#        return self.advancing_node.tangent_nodes
#
#    @property
#    def recording_nodes(self):
#        if self.motion is ADMotion.ADVANCING:
#            return self._recording_nodes
#
#        return self.advancing_node.recording_nodes
#
#    @property
#    def returning_nodes(self):
#        if self.motion is ADMotion.ADVANCING:
#            return self._returning_nodes
#
#        return self.advancing_node.returning_nodes
