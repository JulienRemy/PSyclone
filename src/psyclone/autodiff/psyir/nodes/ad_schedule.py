from psyclone.psyir.nodes import Schedule
from psyclone.psyir.symbols import SymbolTable, DataSymbol, RoutineSymbol

from psyclone.autodiff.psyir import ADPSyIR
from psyclone.autodiff.psyir.nodes import ADNode  # , ADCall, ADIntrinsicCall


# NOTE: this could use an ADScopingNode parent
# iff the symbol table needs more info for AD?
class ADSchedule(Schedule, ADNode):

    # TODO:
    # Could benefit from a tape slice/recomputing schedule attr?
    #   Definitely is where recomputing "anticipated returns" should go
    # ?? Has an ADMotion? or the ADNode itself?
    # Is in effect a scope...
    # Might be a kernel as a for loop body, a conditional branch, a routine body
    # Is active if any statement in it is
    #   and these are active if ...

    def __init__(self, children=None, parent=None, symbol_table=None):
        super().__init__(children, parent, symbol_table)
        self.__init_ad__(children, parent)

        # TODO

    @staticmethod
    def _validate_child(position, child):
        from psyclone.autodiff.psyir.nodes import (
            ADAssignment,
            ADLoop,
            ADIfBlock,
        )

        # TODO: this might benefit from a real parent class?
        ADStatement = (
            ADAssignment,
            ADLoop,
            ADIfBlock,
        )  # ,ADCall, ADIntrinsicCall)
        return isinstance(child, ADStatement)

    def addchild(self, child, index=None):
        from psyclone.autodiff.psyir.nodes import (
            ADAssignment,
            ADLoop,
            ADIfBlock,
        )

        # TODO: this might benefit from a real parent class?
        ADStatement = (
            ADAssignment,
            ADLoop,
            ADIfBlock,
        )  # ,ADCall, ADIntrinsicCall)
        if not isinstance(child, ADStatement):
            raise TypeError("")
        return super().addchild(child, index)

    @classmethod
    def from_psyir(cls, schedule):
        if not isinstance(schedule, Schedule):
            raise TypeError("")
        if isinstance(schedule, ADSchedule):
            raise TypeError("")
        # First transform the symbols in the table
        symbol_table = schedule.symbol_table
        new_symbol_table = SymbolTable(None, symbol_table.default_visibility)
        syms_to_tags = symbol_table.get_reverse_tags_dict()
        arg_map = dict()
        for symbol in symbol_table.symbols:
            if isinstance(symbol, (DataSymbol, RoutineSymbol)):
                new_symbol = ADPSyIR.from_psyir(symbol)
            else:
                new_symbol = symbol.copy()
            if symbol in syms_to_tags:
                tag = syms_to_tags[symbol]
            else:
                tag = None
            new_symbol_table.add(new_symbol, tag)

            # To ensure proper ordering of the argument list
            if symbol in symbol_table.argument_list:
                arg_map[symbol] = new_symbol

        new_argument_list = []
        for symbol in symbol_table.argument_list:
            new_argument_list.append(arg_map[symbol])

        new_symbol_table.specify_argument_list(new_argument_list)

        # NOTE: not parent, nor children, to recursively raise to AD
        from psyclone.autodiff.psyir.nodes import ADRoutine

        if cls is ADRoutine:
            ad_schedule = ADRoutine(
                name=schedule.name, symbol_table=new_symbol_table
            )
        else:
            ad_schedule = cls(symbol_table=new_symbol_table)
        for child in schedule.children:
            if isinstance(child, ADNode):
                ad_schedule.addchild(child)
            else:
                ad_schedule.addchild(ADPSyIR.from_psyir(child))
        return ad_schedule

    def recompute_up_to(self, references):
        # TODO
        raise NotImplementedError("")

    # NOTE: which tape (type?).
    def tape_records(self):
        # TODO
        raise NotImplementedError("")

    # NOTE: which tape (type?).
    def tape_restores(self):
        # TODO
        raise NotImplementedError("")

    # NOTE: which tape (type?).
    def tape_slice(self):
        # TODO
        raise NotImplementedError("")
