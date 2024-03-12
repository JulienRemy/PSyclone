from psyclone.psyir.nodes import Call

from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADDataNode, ADStatement
from psyclone.autodiff.psyir.symbols import ADRoutineSymbol


class ADCall(Call, ADStatement, ADDataNode):
    _text_name = "ADCall"
    _symbol_type = ADRoutineSymbol

    def __init__(self, routine, **kwargs):
        super().__init__(routine, **kwargs)
        children = kwargs.get("children", None)
        parent = kwargs.get("parent", None)
        self.__init_ad__(children, parent)

    @staticmethod
    def _validate_child(position, child):
        return isinstance(child, ADDataNode)

    def node_str(self, colour=True):
        return (
            f"{self.coloured_name(colour)}[name='{self.routine.name}', "
            f"motion='{self.routine.motion.name}'"
        )

    @classmethod
    def from_psyir(cls, call):
        if not isinstance(call, Call):
            raise TypeError("")
        if isinstance(call, ADCall):
            raise TypeError("")
        
        # if call.routine.name in ADDSL.routine_names:
        #     return ADDSL.from_psyir(call)

        ad_routine_symbol = ADRoutineSymbol.from_psyir(call.routine)
        ad_call = cls(routine=ad_routine_symbol, annotations=call.annotations)
        for argument, name in zip(call.children, call.argument_names):
            if isinstance(argument, ADDataNode):
                ad_call.append_named_arg(name, argument)
            else:
                ad_call.append_named_arg(name, ADPSyIR.from_psyir(argument))
        return ad_call

    @classmethod
    def create(cls, routine, arguments):
        call = super().create(routine, arguments)
        return cls.from_psyir(call)
