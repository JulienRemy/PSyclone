import subprocess
#from collections import OrderedDict
from collections.abc import Iterable
from itertools import product
import numpy as np

from psyclone.autodiff.transformations import ADContainerTrans, ADRoutineTrans
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.backend.fortran import FortranWriter
from psyclone.psyir.nodes import (
    Container,
    ArrayReference,
    Call,
    Literal,
    UnaryOperation,
    BinaryOperation,
    Routine,
    Reference,
)
from psyclone.psyir.symbols import (
    ArrayType,
    DataSymbol,
    REAL_DOUBLE_TYPE,
    INTEGER_TYPE,
    RoutineSymbol,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface

# TODO: correct import
from comparator_generator import ComparatorGenerator
from subroutine_generator import FortranSubroutineGenerator


class NumericalComparator(object):
    # _fortran_version = "fortran90"

    @classmethod
    def compare(
        cls,
        tapenade_path,
        file_path,
        routine_name,
        dependent_vars,
        independent_vars,
        argument_values,
        reversal_schedule,
        output_type,
        options=None,
    ):
        if not isinstance(argument_values, dict):
            raise TypeError("")
        for var, values in argument_values.items():
            if not isinstance(var, str):
                raise TypeError("")
            if not isinstance(values, Iterable):
                raise TypeError("")

        #independent_vars = list(independent_vars_and_values.keys())

        comparator, argument_names = ComparatorGenerator.compare(
            tapenade_path,
            file_path,
            routine_name,
            dependent_vars,
            independent_vars,
            reversal_schedule,
            output_type,
            options,
        )

        if set(argument_names) != set(argument_values.keys()):
            raise KeyError("Arguments names returned by "
                           "ComparatorGenerator.compare do not match the keys "
                           "of the 'argument_values' argument.")

        sorted_argument_values = []
        for arg_name in argument_names:
            sorted_argument_values.append(argument_values[arg_name])

        runs_values = []
        runs_result = []
        for values in product(*sorted_argument_values):
            runs_values.append(values)
            runs_result.append(comparator(*values))

            print(", ".join([f"{arg} = {val}" for arg, val in zip(argument_names, values)]))
            print(f"{output_type}:")
            if output_type == "Jacobians_values":
                jacobians = comparator(*values)
                print(f"Autodiff:\n{jacobians[0]}")
                print(f"Tapenade:\n{jacobians[1]}")
            else:
                print(comparator(*values))
            print("===================")

        #print(max(runs_result))

if __name__ == "__main__":
    from psyclone.autodiff.ad_reversal_schedule import ADSplitReversalSchedule

    schedule = ADSplitReversalSchedule()
    # ComparatorGenerator('./tapenade_3.16/bin/tapenade', 'foo_bar.f90', 'foo', ['f', 'g'], ['x', 'w'], schedule, {'verbose': True})

    argument_values = {"x": [1, 2, 3], "w": [4, 5, 6]}
    NumericalComparator.compare(
        "./tapenade_3.16",
        "foo_bar.f90",
        "foo",
        ["f", "g"],
        ["x", "w"],
        argument_values,
        schedule,
        "Jacobians_values",
        {"verbose": True},
    )
