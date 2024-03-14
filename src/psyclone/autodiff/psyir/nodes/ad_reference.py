from enum import Enum
from psyclone.psyir.nodes import Reference, Assignment, BinaryOperation
from psyclone.autodiff.psyir.nodes import ADDataNode
from psyclone.autodiff.psyir.symbols import ADDataSymbol, ADVariableSymbol


class ADReference(Reference, ADDataNode):
    _text_name = "ADReference"

    class Access(Enum):
        READ = 1
        WRITE = 2
        INCREMENT = 3

    # def __init__(self, symbol, version, access, motion=ADMotion.ADVANCING,
    #     advancing_node=None, **kwargs):
    #     if not isinstance(symbol, ADDataSymbol):
    #         raise TypeError("")

    #     super().__init__(symbol, **kwargs)
    #     self.__init_ad__(version, access, motion, advancing_node)

    # def __init_ad__(self, version, access, motion=ADMotion.ADVANCING, advancing_node=None):
    #     if not isinstance(version, int):
    #         raise TypeError("")
    #     if not isinstance(access, Enum) or access not in ADReference.Access:
    #         raise TypeError("")

    #     self._version = version
    #     self._access = access

    #     self.symbol.log_reference(self)

    #     super().__init_ad__(motion, advancing_node)

    def __init__(self, symbol, version, access, **kwargs):
        if not isinstance(symbol, ADDataSymbol):
            raise TypeError("")

        super().__init__(symbol, **kwargs)
        self.__init_ad_reference__(version, access, **kwargs)

    def __init_ad_reference__(self, version, access, **kwargs):
        if not isinstance(version, int):
            raise TypeError(f"{type(version).__name__}")
        if not isinstance(access, Enum) or access not in ADReference.Access:
            raise TypeError("")

        children = kwargs.get("children", None)
        parent = kwargs.get("parent", None)
        super().__init_ad__(children, parent)

        self._version = version
        self._access = access

        self.symbol.log_reference(self)

    @property
    def version(self):
        return self._version

    @property
    def access(self):
        return self._access

    def node_str(self, colour=True):
        return (
            f"{self.coloured_name(colour)}[name:'{self.name}', "
            f"version:{self.version}, access:{self.access.name}]"
        )

    @classmethod
    def from_psyir(cls, reference):
        if not isinstance(reference, Reference):
            raise TypeError("")
        if isinstance(reference, ADReference):
            raise TypeError("")
        if isinstance(reference.symbol, ADDataSymbol):
            ad_symbol = reference.symbol
        else:
            ad_symbol = ADVariableSymbol.from_psyir(reference.symbol)

        # TODO: reference appearing in a Call node with an intent (in)out
        # just before should also be a next version, even if it's a read
        if (
            isinstance(reference.parent, Assignment)
            and reference == reference.parent.lhs
        ):
            # TODO: ADIncrement goes here
            return ad_symbol.create_new_version_reference()
            # version = ad_symbol.versions
            # access = ADReference.Access.WRITE
        else:
            return ad_symbol.create_last_version_reference()
            # version = ad_symbol.versions - 1
            # access = ADReference.Access.READ

        # return cls(ad_symbol,
        #            version,
        #            access)

    # @property
    # def symbol(self):
    #     ''' Return the referenced ADDataSymbol.

    #     :returns: the referenced ADDataSymbol.
    #     :rtype: :py:class:`psyclone.autodiff.psyir.symbols.ADDataSymbol`

    #     '''
    #     return self._symbol

    # @symbol.setter
    # def symbol(self, symbol):
    #     '''
    #     :param symbol: the new symbol being referenced.
    #     :type symbol: :py:class:`psyclone.autodiff.psyir.symbols.ADDataSymbol`

    #     :raises TypeError: if the symbol argument is not of type ADDataSymbol.

    #     '''
    #     if not isinstance(symbol, ADDataSymbol):
    #         raise TypeError(
    #             f"The {type(self).__name__} symbol setter expects an AD PSyIR "
    #             f"ADDataSymbol object but found '{type(symbol).__name__}'.")
    #     self._symbol = symbol
