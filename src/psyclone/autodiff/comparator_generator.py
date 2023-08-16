# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2021-2022, Science and Technology Facilities Council.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
# Authors: J. Remy, Inria

"""This file contains a Fortran subroutine generator class for the purpose of \
comparing numerical results of programs transformed using `psyclone.autodiff` \
with Tapenade (3.16) in reverse-mode (without `vector`/`multi` mode).
Both tools are used to transform a subroutine, which possible nested calls and \
fill Jacobians for all dependent and independent variables. 
These are then compared in a Fortran subroutine, which is called from Python \
using `numpy.f2py`.
NOTE: this is a work in progress, it hasn't been cleaned up, refactored nor \
tested yet.
"""

import subprocess
from importlib import import_module

from psyclone.autodiff.transformations import ADContainerTrans, ADRoutineTrans
from psyclone.psyir.frontend.fortran import FortranReader
from psyclone.psyir.backend.fortran import FortranWriter
from psyclone.psyir.nodes import (
    ArrayReference,
    Call,
    Literal,
    UnaryOperation,
    BinaryOperation,
    IntrinsicCall,
    Routine,
    Reference,
)
from psyclone.psyir.symbols import (
    ArrayType,
    REAL_DOUBLE_TYPE,
    REAL_SINGLE_TYPE,
    INTEGER_TYPE,
    RoutineSymbol,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface

from psyclone.autodiff import SubroutineGenerator

# NOTE: this is unused but called using subprocess
from numpy import f2py


class ComparatorGenerator(object):
    """This class is a Fortran subroutine generator for the purpose of \
    comparing numerical results of programs transformed using `psyclone.autodiff` \
    with Tapenade (3.16) in reverse-mode (without `vector`/`multi` mode).
    Both tools are used to transform a subroutine, with possible nested calls and \
    fill Jacobians for all dependent and independent variables. 
    These are then compared in a Fortran subroutine, which is called from Python \
    using `numpy.f2py`.
    NOTE: this is a work in progress, it hasn't been cleaned up, refactored nor \
    tested yet.
    """

    # _fortran_version = "fortran90"

    _default_scalar_datatype = REAL_DOUBLE_TYPE

    @staticmethod
    def _apply_autodiff(
        file_container,
        routine_name,
        dependent_vars,
        independent_vars,
        reversal_schedule,
        options=None
    ):
        """Applies an ADContainerTrans from psyclone.autodiff.

        :param file_container: PSyIR file container containing the routine.
        :type file_container: :py:class:`psyclone.psyir.nodes.FileContainer`
        :param routine_name: name of the routine.
        :type routine_name: str
        :param dependent_vars: list of names of dependent variables.
        :type dependent_vars: list[str]
        :param independent_vars: list of names of independent variables.
        :type independent_vars: list[str]
        :param reversal_schedule: AD reversal schedule.
        :type reversal_schedule: :py:class:`psyclone.autodiff.ADReversalSchedule`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :return: a copied and modified container with all necessary \
            Routine definitions.
        :rtype: :py:class:`psyclone.psyir.nodes.Container`
        """

        container_trans = ADContainerTrans()
        return container_trans.apply(
            file_container,
            routine_name,
            dependent_vars,
            independent_vars,
            reversal_schedule,
            options,
        )

    # TODO: doc this, clean it, refactor it, test it
    @classmethod
    def compare(
        cls,
        tapenade_path,
        file_path,
        routine_name,
        dependent_vars,
        independent_vars,
        reversal_schedule,
        output_type="Linf_error",
        options=None,
    ):
        
        if not isinstance(tapenade_path, str):
            raise TypeError(
                f"'tapenade_path' argument of ComparatorGenerator "
                f"should be of type 'str' but found "
                f"'{type(tapenade_path).__name__}'."
            )
        if not isinstance(file_path, str):
            raise TypeError(
                f"'file_path' argument of ComparatorGenerator "
                f"should be of type 'str' but found "
                f"'{type(file_path).__name__}'."
            )
        if not isinstance(output_type, str):
            raise TypeError(
                f"'output_type' argument of ComparatorGenerator "
                f"should be of type 'str' but found "
                f"'{type(output_type).__name__}'."
            )
        if output_type not in (
            "Linf_error",
            "L1_error",
            "Jacobians_values",
            "Jacobians_error",
        ):
            raise ValueError(
                f"'output_type' argument of ComparatorGenerator "
                f"should be one of 'Linf_error', 'L1_error', "
                f"'Jacobians_values' and 'Jacobians_error' but found "
                f"'{output_type}'."
            )

        # Other arguments will fail typechecks when transforming

        # Reader and writer
        freader = FortranReader()
        fwriter = FortranWriter()

        # Apply autodiff
        file_container = freader.psyir_from_file(file_path)
        autodiff_result = cls._apply_autodiff(
            file_container,
            routine_name,
            dependent_vars,
            independent_vars,
            reversal_schedule,
            options,
        )

        # Apply Tapenade
        head = (
            f"{routine_name}({' '.join(dependent_vars)})/({' '.join(independent_vars)})"
        )
        subprocess.run(
            [tapenade_path + "/bin/tapenade", file_path, "-head", head, "-reverse"],
            text=True,
            check=True,
            capture_output=True,
        )

        # Position of the dot in the file name
        # TODO: detect the dot, this is messy...
        dot_index = -4  # file_path.find('.', -1, 0)
        tapenade_output_file_path = file_path[:dot_index] + "_b" + file_path[dot_index:]

        # create the comparator subroutine
        comparator = SubroutineGenerator(routine_name + "_comp")

        # Deal with the input subroutine arguments
        # These are the arguments of the comparator subroutine, same as those of the input routine
        comparator_arguments = []
        # These are variables storing the original values of the comparator arguments
        saved_arguments = []
        # These are the arguments used in calling the autodiff or Tapenade reversing routine
        rev_call_args = []
        # These are the symbols of the adjoints of the dependent vars
        dependent_adjoints = []
        # These are the symbols of the adjoints of the independent vars
        independent_adjoints = []

        # The input subroutine, as a PSyIR Routine
        input_subroutine = None
        routines = file_container.walk(Routine)
        for routine in routines:
            if routine.name == routine_name:
                input_subroutine = routine
                break
        if input_subroutine is None:
            raise ValueError("")

        # Now go through the input routine arguments
        for original_arg in input_subroutine.symbol_table.argument_list:
            # Add arguments of correct intent to the comparator subroutine
            # these are also used in calling the reversing routine

            comparator_var = None
            # Intent(out) arguments of the input routine are variables of the comparator
            if original_arg.interface.access is ArgumentInterface.Access.WRITE:
                comparator_var = comparator.new_variable(
                    original_arg.name, original_arg.datatype
                )
                rev_call_args.append(comparator_var)
            # Others have input values, so are arguments of the comparator
            else:
                comparator_var = comparator.new_arg(
                    original_arg.name,
                    original_arg.datatype,
                    original_arg.interface.access,
                )
                comparator_arguments.append(comparator_var)
                rev_call_args.append(comparator_var)

            # Add variables to save the values of arguments to the comparator subroutine
            # and save them, except for intent(in) which cannot be modified
            # We will restore the values to 'comparator_arg' before each call
            # If intent(in), None so that the comparator_arguments and saved_arguments list
            # still correspond
            if original_arg.interface.access in (
                ArgumentInterface.Access.READ,
                ArgumentInterface.Access.WRITE,
            ):
                saved_arguments.append(None)
            else:
                saved_arg = comparator.new_variable(
                    "saved_" + original_arg.name, original_arg.datatype
                )
                comparator.new_assignment(saved_arg, comparator_var)
                saved_arguments.append(saved_arg)

            # If this is a (in)dependent variable, it has an adjoint
            if original_arg.name in dependent_vars + independent_vars:
                # Create the name of the adjoint and a new variable for it
                # TODO: correct datatype
                arg_adj_name = (
                    ADRoutineTrans._adjoint_prefix
                    + original_arg.name
                    + ADRoutineTrans._adjoint_suffix
                )
                arg_adj = comparator.new_variable(arg_adj_name, cls._default_scalar_datatype)

                # Add it to the correct list of adjoint symbols
                if original_arg.name in dependent_vars:
                    dependent_adjoints.append(arg_adj)
                if original_arg.name in independent_vars:
                    independent_adjoints.append(arg_adj)

                # Add it as a call argument to the reversing routine
                rev_call_args.append(arg_adj)

        # Now deal with the Jacobians we are going to generate by calling
        # the reversing routines repeatedly
        # Create an ArrayType of correct dimensions for the autodiff/Tapenade Jacobians
        jacobian_datatype = ArrayType(
            cls._default_scalar_datatype, [len(independent_vars), len(dependent_vars)]
        )

        # Create variables for them, as intent(out) argument or local variable
        if output_type == "Jacobians_values":
            J_autodiff = comparator.new_out_arg(
                "J_" + routine_name + "_autodiff", jacobian_datatype
            )
            J_tapenade = comparator.new_out_arg(
                "J_" + routine_name + "_tapenade", jacobian_datatype
            )
        else:
            J_autodiff = comparator.new_variable(
                "J_" + routine_name + "_autodiff", jacobian_datatype
            )
            J_tapenade = comparator.new_variable(
                "J_" + routine_name + "_tapenade", jacobian_datatype
            )

        jacobians = (J_autodiff, J_tapenade)
        reversing_names = (
            ADRoutineTrans._reversing_prefix
            + routine_name
            + ADRoutineTrans._reversing_suffix,
            routine_name + "_b",
        )

        # Now for both the autodiff and Tapenade (jacobian, reversing routine) pair
        for J, rev_name in zip(jacobians, reversing_names):
            # For every row of the jacobian
            for row, dep_adj in enumerate(dependent_adjoints):
                # Restore saved versions
                for comparator_arg, saved_arg in zip(
                    comparator_arguments, saved_arguments
                ):
                    if saved_arg is not None:
                        comparator.new_assignment(comparator_arg, saved_arg)

                # Assign 1 to dependent adjoint for this row
                comparator.new_assignment(dep_adj, Literal("1.0", dep_adj.datatype))
                # Assign 0 to all others
                for other_dep_adj in dependent_adjoints:
                    if other_dep_adj != dep_adj:
                        comparator.new_assignment(
                            other_dep_adj, Literal("0.0", other_dep_adj.datatype)
                        )

                # Assign 0 to all independent adjoints but the dependent one
                for indep_adj in independent_adjoints:
                    if dep_adj != indep_adj:
                        comparator.new_assignment(
                            indep_adj, Literal("0.0", dep_adj.datatype)
                        )

                # Create a RoutineSymbol for the autodiff/Tapenade reversing routine and a call to it
                rev_symbol = RoutineSymbol(rev_name)
                rev_call_arg_refs = [Reference(sym) for sym in rev_call_args]
                rev_call = Call.create(rev_symbol, rev_call_arg_refs)
                comparator.subroutine.addchild(rev_call)

                # Assign the independent adjoints returns to elements of the Jacobian matrix
                for col, indep_adj in enumerate(independent_adjoints):
                    J_element_ref = ArrayReference.create(
                        J,
                        [
                            Literal(str(col + 1), INTEGER_TYPE),
                            Literal(str(row + 1), INTEGER_TYPE),
                        ],
                    )
                    comparator.new_assignment(J_element_ref, indep_adj)

        # Generate the required output if different from 'Jacobians_values'
        if output_type in ("L1_error", "Linf_error", "Jacobians_error"):
            # In any case, compute the absolute error between the Jacobians
            # This is J_..._autodiff - J_..._tapenade
            J_diff = BinaryOperation.create(
                BinaryOperation.Operator.SUB,
                Reference(J_autodiff),
                Reference(J_tapenade),
            )
            # This is abs(J_..._autodiff - J_..._tapenade)
            J_abs_diff = UnaryOperation.create(UnaryOperation.Operator.ABS, J_diff)

            # Assign it to an intent(out) argument or not
            if output_type == "Jacobians_error":
                J_error = comparator.new_out_arg(
                    "J_" + routine_name + "_error", jacobian_datatype
                )
                comparator.new_assignment(J_error, J_abs_diff)
            else:
                J_error = comparator.new_variable(
                    "J_" + routine_name + "_error", jacobian_datatype
                )
                comparator.new_assignment(J_error, J_abs_diff)

                # Compute the L1 norm or the Linf norm
                if output_type == "L1_error":
                    L1_error = comparator.new_out_arg("L1_error", cls._default_scalar_datatype)
                    L1_sum = IntrinsicCall.create(
                        IntrinsicCall.Intrinsic.SUM, [Reference(J_error)]
                    )
                    comparator.new_assignment(L1_error, L1_sum)
                elif output_type == "Linf_error":
                    Linf_error = comparator.new_out_arg("Linf_error", cls._default_scalar_datatype)
                    Linf_max = IntrinsicCall.create(
                        IntrinsicCall.Intrinsic.MAXVAL, [Reference(J_error)]
                    )
                    comparator.new_assignment(Linf_error, Linf_max)

        ##############################
        # Now write all this to a file with:
        # - the autodiff transformation result
        # - the Tapenade transformation result
        # - the comparator subroutine

        comparator_file_path = file_path[:dot_index] + "_comp" + file_path[dot_index:]
        # Write to the file
        with open(comparator_file_path, "w") as comparator_file:
            autodiff_string = fwriter(autodiff_result)
            comparator_file.write("! psyclone.autodiff produced:\n")
            comparator_file.write(autodiff_string)

            comparator_file.write("\n\n\n! ===================================== \n")
            comparator_file.write("! Tapenade produced:\n")
            with open(tapenade_output_file_path, "r") as tapenade_file:
                comparator_file.write(tapenade_file.read())

            comparator_file.write("\n\n\n! ===================================== \n")
            comparator_file.write("! Comparator subroutine:\n")
            comparator_file.write(comparator.write())

        ###############################
        # Finally, compile using numpy.f2py
        module_and_function_name = routine_name + "_comp"

        # TODO: fix the extension
        subprocess.run(
            [
                "f2py3",
                "-c",
                "-m",
                module_and_function_name,
                comparator_file_path,
                tapenade_path + "/ADFirstAidKit/adStack.c",
            ],
            check=True,
            capture_output=True,
        )

        module = import_module(module_and_function_name)

        return getattr(module, module_and_function_name), [
            arg.name for arg in comparator_arguments
        ]


"""
if __name__ == "__main__":
    from psyclone.autodiff.ad_reversal_schedule import ADSplitReversalSchedule

    schedule = ADSplitReversalSchedule()
    output_types = ("Jacobians_values", "Jacobians_error", "L1_error", "Linf_error")
    output_type = output_types[1]
    foo_comp, arg_names = ComparatorGenerator.compare(
        "./tapenade_3.16",
        "foo.f90",        #f2py_result = f2py.compile(
        #    sourcecode, modulename=module_and_function_name, extension=".f90"
        #)
        #if f2py_result != 0:
        #    raise ValueError("f2py compilation failed.")
        "foo",
        ["f", "g"],
        ["x", "w"],
        schedule,
        output_type,
        {"verbose": True},
    )

    # from numpy import f2py
    # with open('foo_bar_comp.f90', 'r') as sourcefile:
    #    sourcecode = sourcefile.read()
    # f2py_result = f2py.compile(sourcecode, modulename='TEST', extension='.f90')
    # if f2py_result != 0:
    #    raise ValueError("")
    ##f2py.run_main(['-c', '-m TEST', 'foo_bar_comp.f90'])
    print(foo_comp.__doc__)

    if output_type == "Jacobians_values":
        J_autodiff, J_tapenade = foo_comp(2.23, 3.51, 1.2)
        print(J_autodiff)
        print(J_tapenade)
    else:
        error = foo_comp(2.23, 3.51, 1.2)
        print(error)
"""
