from psyclone.psyir.nodes import ArrayReference, Assignment

from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADNode, ADDataNode, ADReference
from psyclone.autodiff.psyir.symbols import ADDataSymbol, ADVariableSymbol


class ADArrayReference(ArrayReference, ADReference):
    _text_name = "ADArrayReference"

    def __init__(self, symbol, version, access, indices, **kwargs):
        if not isinstance(symbol, ADDataSymbol):
            raise TypeError("")

        array_ref = ArrayReference.create(
            symbol, [index.copy() for index in indices]
        )

        super().__init__(symbol, version, access, **kwargs)
        for index in array_ref.children:
            if isinstance(index, ADNode):
                self.addchild(index)
            else:
                self.addchild(ADPSyIR.from_psyir(index))

    @classmethod
    def from_psyir(cls, array_reference):
        if isinstance(array_reference.symbol, ADDataSymbol):
            ad_symbol = array_reference.symbol
        else:
            ad_symbol = ADVariableSymbol.from_psyir(array_reference.symbol)
        # TODO deal with indices wrt new version or not...
        # For now this treats the whole array as a new version even if writing
        # only one element of it.
        if (
            isinstance(array_reference.parent, Assignment)
            and array_reference == array_reference.parent.lhs
        ):
            ad_ref = ad_symbol.create_new_version_array_reference(
                array_reference.indices
            )
        else:
            ad_ref = ad_symbol.create_last_version_array_reference(
                array_reference.indices
            )
        return ad_ref

    def addchild(self, child, index=None):
        if not isinstance(child, ADNode):
            raise TypeError("")
        ADNode.addchild(self, child, index)
        self.backward_data_flow.append(child)
        child.forward_data_flow.append(self)

    @classmethod
    def create(cls, symbol, indices):
        if not isinstance(symbol, ADDataSymbol):
            raise TypeError("")
        if not isinstance(indices, list):
            raise TypeError("")
        for index in indices:
            if not (index == ":" or isinstance(index, ADDataNode)):
                raise TypeError("")
        array_ref = super().create(symbol, indices)
        ad_array_ref = ADPSyIR.from_psyir(array_ref)
        return ad_array_ref
