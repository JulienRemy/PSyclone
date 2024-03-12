from psyclone.psyir.nodes import Routine

from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADSchedule, ADNode
from psyclone.autodiff.psyir.symbols import ADRoutineSymbol


class ADRoutine(Routine, ADSchedule):
    _text_name = "ADRoutine"

    def __init__(self, name, is_program=False, **kwargs):
        super().__init__(name, is_program, **kwargs)
        children = kwargs.get("children", None)
        parent = kwargs.get("parent", None)
        self.__init_ad__(children, parent)

        routine_symbol = self.routine_symbol
        if isinstance(routine_symbol, ADRoutineSymbol):
            ad_routine_symbol = routine_symbol
        else:
            ad_routine_symbol = ADPSyIR.from_psyir(routine_symbol)
        self.symbol_table.remove(routine_symbol)
        self.symbol_table.add(ad_routine_symbol, "own_routine_symbol")

    # TODO
    def generate_data_flow_graph(self):
        # TODO: deal with all in/out args
        raise NotImplementedError("")

    @classmethod
    def create(
        cls,
        name,
        symbol_table,
        children,
        is_program=False,
        return_symbol_name=None,
    ):
        routine = super().create(
            name, symbol_table, children, is_program, return_symbol_name
        )
        for child in children:
            if not isinstance(child, ADNode):
                raise TypeError("")

        return cls.from_psyir(routine)

    def node_str(self, colour=True):
        return (
            self.coloured_name(colour) + f"[name:'{self.name}', "
            f"motion:{self.routine_symbol.motion.name}]"
        )

    # TODO
    # This could benefit from
    # - argument management and adder (_d, _adj, tape)
    # -

    @property
    def routine_symbol(self):
        return self.symbol_table.lookup_with_tag("own_routine_symbol")

    @property
    def motion(self):
        return self.routine_symbol.motion

    @property
    def advancing_routine(self):
        return self.routine_symbol.advancing_routine_symbol.routine

    @property
    def tangent_routine(self):
        return self.routine_symbol.tangent_routine_symbol.routine

    @property
    def recording_routine(self):
        return self.routine_symbol.recording_routine_symbol.routine

    @property
    def returning_routine(self):
        return self.routine_symbol.returning_routine_symbol.routine

    @property
    def reversing_routine(self):
        return self.routine_symbol.reversing_routine_symbol.routine
