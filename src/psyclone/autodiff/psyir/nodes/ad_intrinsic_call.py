from psyclone.psyir.nodes import IntrinsicCall

from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADCall, ADNode


class ADIntrinsicCall(IntrinsicCall, ADCall):
    _text_name = "ADIntrinsicCall"

    def __init__(self, routine, **kwargs):
        super().__init__(routine, **kwargs)
        children = kwargs.get("children", None)
        parent = kwargs.get("parent", None)
        self.__init_ad__(children, parent)

    @classmethod
    def from_psyir(cls, call):
        if not isinstance(call, IntrinsicCall):
            raise TypeError("")
        if isinstance(call, ADIntrinsicCall):
            raise TypeError("")
        ad_call = cls(routine=call.intrinsic, annotations=call.annotations)
        for argument, name in zip(call.children, call.argument_names):
            if isinstance(argument, ADNode):
                ad_call.append_named_arg(name, argument)
            else:
                ad_call.append_named_arg(name, ADPSyIR.from_psyir(argument))
        return ad_call

    @classmethod
    def create(cls, routine, arguments):
        call = super().create(routine, arguments)
        return cls.from_psyir(call)
