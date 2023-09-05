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
# Author: J. Remy, Université Grenoble Alpes, Inria

"""This file contains a Fortran subroutine generator class for the purpose of \
comparing numerical results of programs transformed using `psyclone.autodiff` \
with Tapenade (3.16) in forward- and reverse-mode (without `vector`/`multi` \
mode).
Both tools are used to transform a subroutine, which possible nested calls and \
fill Jacobians for all dependent and independent variables. 
These are then compared in a Fortran subroutine, which is called from Python \
using `numpy.f2py`.
"""
# pylint: disable=unused-import

import subprocess
from importlib import import_module

# NOTE: this is unused but called using subprocess
from numpy import f2py

from psyclone.autodiff.transformations import (
    ADReverseContainerTrans,
    ADReverseRoutineTrans,
    ADForwardContainerTrans,
    ADForwardRoutineTrans,
)
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
    REAL_TYPE,
    INTEGER_TYPE,
    RoutineSymbol,
)
from psyclone.psyir.symbols.interfaces import ArgumentInterface
from psyclone.line_length import FortLineLength

from psyclone.autodiff import SubroutineGenerator


class ComparatorGenerator(object):
    """This class is a Fortran subroutine generator for the purpose of \
    comparing numerical results of programs transformed using \
    `psyclone.autodiff` with Tapenade (3.16) in forward- and reverse-mode \
    (without `vector`/`multi` mode).
    Both tools are used to transform a subroutine, with possible nested calls \
    and fill Jacobians for all dependent and independent variables. 
    These are then compared in a Fortran subroutine, which is called from \
    Python using `numpy.f2py`.
    """

    # pylint: disable=useless-object-inheritance, too-few-public-methods

    # TODO: specify Fortran version
    # _fortran_version = "fortran90"

    _default_scalar_datatype = REAL_TYPE

    @staticmethod
    def _apply_reverse_autodiff(
        file_container,
        routine_name,
        dependent_vars,
        independent_vars,
        reversal_schedule,
        options=None,
    ):
        """Applies an ADReverseContainerTrans from psyclone.autodiff.

        :param file_container: PSyIR file container containing the routine.
        :type file_container: :py:class:`psyclone.psyir.nodes.FileContainer`
        :param routine_name: name of the routine.
        :type routine_name: Str
        :param dependent_vars: list of names of dependent variables.
        :type dependent_vars: List[Str]
        :param independent_vars: list of names of independent variables.
        :type independent_vars: List[Str]
        :param reversal_schedule: AD reversal schedule.
        :type reversal_schedule: :py:class:`psyclone.autodiff.ADReversalSchedule`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: a copied and modified container with all necessary \
                 Routine definitions.
        :rtype: :py:class:`psyclone.psyir.nodes.Container`
        """
        # pylint: disable=too-many-arguments

        container_trans = ADReverseContainerTrans()
        return container_trans.apply(
            file_container,
            routine_name,
            dependent_vars,
            independent_vars,
            reversal_schedule,
            options,
        )

    @staticmethod
    def _apply_forward_autodiff(
        file_container,
        routine_name,
        dependent_vars,
        independent_vars,
        options=None,
    ):
        """Applies an ADForwardContainerTrans from psyclone.autodiff.

        :param file_container: PSyIR file container containing the routine.
        :type file_container: :py:class:`psyclone.psyir.nodes.FileContainer`
        :param routine_name: name of the routine.
        :type routine_name: Str
        :param dependent_vars: list of names of dependent variables.
        :type dependent_vars: List[Str]
        :param independent_vars: list of names of independent variables.
        :type independent_vars: List[Str]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: a copied and modified container with all necessary \
                 Routine definitions.
        :rtype: :py:class:`psyclone.psyir.nodes.Container`
        """

        container_trans = ADForwardContainerTrans()
        return container_trans.apply(
            file_container,
            routine_name,
            dependent_vars,
            independent_vars,
            options,
        )

    @staticmethod
    def _check_args(tapenade_path, file_path, output_type, mode):
        # Other arguments will be checked when transforming

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
        if not isinstance(mode, str):
            raise TypeError(
                f"'mode' argument of ComparatorGenerator "
                f"should be of type 'str' but found "
                f"'{type(mode).__name__}'."
            )
        if mode not in ("forward", "reverse"):
            raise ValueError(
                f"'mode' argument of ComparatorGenerator "
                f"should be either 'forward' or 'reverse' "
                f"but found '{mode}'."
            )

    # TODO: doc this
    # TODO: this is a mess.
    @classmethod
    def compare(
        cls,
        tapenade_path,
        file_path,
        routine_name,
        dependent_vars,
        independent_vars,
        output_type,
        options,
        mode,
        reversal_schedule=None,
    ):
        # Other arguments will be checked when transforming
        cls._check_args(tapenade_path, file_path, output_type, mode)

        # Reader and writer
        freader = FortranReader()
        fwriter = FortranWriter()

        # Apply autodiff
        file_container = freader.psyir_from_file(file_path)
        if mode == "forward":
            autodiff_result = cls._apply_forward_autodiff(
                file_container,
                routine_name,
                dependent_vars,
                independent_vars,
                options,
            )
        else:
            autodiff_result = cls._apply_reverse_autodiff(
                file_container,
                routine_name,
                dependent_vars,
                independent_vars,
                reversal_schedule,
                options,
            )

        # Apply Tapenade
        head = f"{routine_name}({' '.join(dependent_vars)})"
        head += f"/({' '.join(independent_vars)})"
        tapenade_mode = "-tangent" if mode == "forward" else "-reverse"

        # Position of the dot in the file name
        dot_index = file_path.rfind('.')
        tapenade_postfix = "_d" if mode == "forward" else "_b"
        tapenade_output_file_path = (
            file_path[:dot_index] + tapenade_postfix + file_path[dot_index:]
        )

        slash_index = tapenade_output_file_path.rfind('/')
        if slash_index != -1:
            tapenade_output_dir = file_path[:(slash_index + 1)]
        else:
            tapenade_output_dir = "./"

        subprocess.run(
            [
                tapenade_path + "/bin/tapenade",
                file_path,
                "-head",
                head,
                "-O",
                tapenade_output_dir,
                tapenade_mode,
            ],
            text=True,
            check=True,
            capture_output=True,
        )

        # create the comparator subroutine
        comparator = SubroutineGenerator(routine_name + "_comp")

        # Deal with the input subroutine arguments
        # These are the arguments of the comparator subroutine,
        # same as those of the input routine
        comparator_arguments = []
        # These are variables storing the original values of the
        # comparator arguments
        saved_arguments = []
        # These are the arguments used in calling the autodiff or
        # Tapenade routine
        call_args = []
        # These are the symbols of the derivatives/adjoints of the
        # dependent vars
        dependent_diffs = []
        # These are the symbols of the derivatives/adjoints of the
        # independent vars
        independent_diffs = []

        # The input subroutine, as a PSyIR Routine
        input_subroutine = None
        routines = file_container.walk(Routine)
        for routine in routines:
            if routine.name == routine_name:
                input_subroutine = routine
                break
        if input_subroutine is None:
            raise ValueError(
                f"No routine named '{routine_name}' was found "
                f"in the file '{file_path}'."
            )

        # Now go through the input routine arguments
        for original_arg in input_subroutine.symbol_table.argument_list:
            # Add arguments of correct intent to the comparator subroutine
            # these are also used in calling the transformed routines

            comparator_var = None
            # Intent(out) arguments of the input routine are variables of 
            # the comparator
            if original_arg.interface.access is ArgumentInterface.Access.WRITE:
                comparator_var = comparator.new_variable(
                    original_arg.name, original_arg.datatype
                )
                call_args.append(comparator_var)
            # Others have input values, so are arguments of the comparator
            else:
                comparator_var = comparator.new_arg(
                    original_arg.name,
                    original_arg.datatype,
                    original_arg.interface.access,
                )
                comparator_arguments.append(comparator_var)
                call_args.append(comparator_var)

            # Add variables to save the values of arguments to the comparator
            # subroutine and save them, 
            # except for intent(in) which cannot be modified.
            # We will restore the values to 'comparator_arg' before each call
            # If intent(in), None so that the comparator_arguments and
            # saved_arguments list still correspond.
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

            # If this is a (in)dependent variable,
            # it has a derivative / an adjoint
            if original_arg.name in dependent_vars + independent_vars:
                # Create the name of the derivative/adjoint
                # and a new variable for it
                # TODO: correct datatype
                if mode == "forward":
                    arg_diff_name = (
                        ADForwardRoutineTrans._differential_prefix
                        + original_arg.name
                        + ADForwardRoutineTrans._differential_postfix
                    )
                else:
                    arg_diff_name = (
                        ADReverseRoutineTrans._differential_prefix
                        + original_arg.name
                        + ADReverseRoutineTrans._differential_postfix
                    )
                arg_diff = comparator.new_variable(
                    arg_diff_name, cls._default_scalar_datatype
                )

                # Add it to the correct list of diff symbols
                if original_arg.name in dependent_vars:
                    dependent_diffs.append(arg_diff)
                if original_arg.name in independent_vars:
                    independent_diffs.append(arg_diff)

                # Add it as a call argument of the transformed routine
                call_args.append(arg_diff)

        # Now deal with the Jacobians we are going to generate by calling
        # the transformed routines repeatedly
        # Create an ArrayType of correct dimensions for the 
        # autodiff/Tapenade Jacobians
        jacobian_datatype = ArrayType(
            cls._default_scalar_datatype,
            [len(independent_vars), len(dependent_vars)],
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
        if mode == "forward":
            transformed_names = (
                ADForwardRoutineTrans._tangent_prefix
                + routine_name
                + ADForwardRoutineTrans._tangent_postfix,
                routine_name + tapenade_postfix,
            )
        else:
            transformed_names = (
                ADReverseRoutineTrans._reversing_prefix
                + routine_name
                + ADReverseRoutineTrans._reversing_postfix,
                routine_name + tapenade_postfix,
            )

        # Now for both the autodiff and Tapenade 
        # (jacobian, reversing routine) pair
        for J, transformed_name in zip(jacobians, transformed_names):
            # Depending on the mode, we assign 1 to the (in)dependent variables
            # one by one
            if mode == "forward":
                first_diffs = independent_diffs
                second_diffs = dependent_diffs
            else:
                first_diffs = dependent_diffs
                second_diffs = independent_diffs

            # For every column/row (if forward/reverse) of the jacobian
            for first_dim, first_diff in enumerate(first_diffs):
                # Restore saved versions
                for comparator_arg, saved_arg in zip(
                    comparator_arguments, saved_arguments
                ):
                    if saved_arg is not None:
                        comparator.new_assignment(comparator_arg, saved_arg)

                # Assign 1 to independent derivative/dependent adjoint 
                # for this column/row
                comparator.new_assignment(
                    first_diff, Literal("1.0", first_diff.datatype)
                )
                # Assign 0 to all others
                for other_first_diff in first_diffs:
                    if other_first_diff != first_diff:
                        comparator.new_assignment(
                            other_first_diff,
                            Literal("0.0", other_first_diff.datatype),
                        )

                # Assign 0 to all dependent derivatives/independent adjoints 
                # but the indepdent/dependent one
                for second_diff in second_diffs:
                    if first_diff != second_diff:
                        comparator.new_assignment(
                            second_diff, Literal("0.0", first_diff.datatype)
                        )

                # Create a RoutineSymbol for the autodiff/Tapenade routine 
                # and a call to it
                transformed_symbol = RoutineSymbol(transformed_name)
                call_arg_refs = [Reference(sym) for sym in call_args]
                call = Call.create(transformed_symbol, call_arg_refs)
                comparator.subroutine.addchild(call)

                # Assign the dependent derivatives/independent adjoints 
                # returns to elements of the Jacobian matrix
                for second_dim, second_diff in enumerate(second_diffs):
                    if mode == "forward":
                        row = second_dim
                        col = first_dim
                    else:
                        row = first_dim
                        col = second_dim

                    J_element_ref = ArrayReference.create(
                        J,
                        [
                            Literal(str(col + 1), INTEGER_TYPE),
                            Literal(str(row + 1), INTEGER_TYPE),
                        ],
                    )
                    comparator.new_assignment(J_element_ref, second_diff)

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
            J_abs_diff = UnaryOperation.create(
                UnaryOperation.Operator.ABS, J_diff
            )

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
                    L1_error = comparator.new_out_arg(
                        "L1_error", cls._default_scalar_datatype
                    )
                    L1_sum = IntrinsicCall.create(
                        IntrinsicCall.Intrinsic.SUM, [Reference(J_error)]
                    )
                    comparator.new_assignment(L1_error, L1_sum)
                elif output_type == "Linf_error":
                    Linf_error = comparator.new_out_arg(
                        "Linf_error", cls._default_scalar_datatype
                    )
                    Linf_max = IntrinsicCall.create(
                        IntrinsicCall.Intrinsic.MAXVAL, [Reference(J_error)]
                    )
                    comparator.new_assignment(Linf_error, Linf_max)

        ##############################
        # Now write all this to a file with:
        # - the autodiff transformation result
        # - the Tapenade transformation result
        # - the comparator subroutine
        # taking care of using line_length on PSyclone outputs

        line_length = FortLineLength()

        comparator_file_path = (
            file_path[:dot_index] + "_comp" + file_path[dot_index:]
        )
        # Write to the file
        with open(comparator_file_path, "w") as comparator_file:
            autodiff_string = fwriter(autodiff_result)
            autodiff_string = line_length.process(autodiff_string)
            comparator_file.write("! psyclone.autodiff produced:\n")
            comparator_file.write(autodiff_string)

            comparator_file.write(
                "\n\n\n! ===================================== \n"
            )
            comparator_file.write("! Tapenade produced:\n")
            with open(tapenade_output_file_path, "r") as tapenade_file:
                comparator_file.write(tapenade_file.read())

            comparator_file.write(
                "\n\n\n! ===================================== \n"
            )
            comparator_file.write("! Comparator subroutine:\n")
            comparator_string = comparator.write()
            comparator_string = line_length.process(comparator_string)
            comparator_file.write(comparator_string)

        ###############################
        # Finally, compile using numpy.f2py

        # Create a module name that doesn't already exist
        correct_module_name = False
        module_name_postfix = 0
        while not correct_module_name:
            try:
                import_module(
                    routine_name + "_comp_" + str(module_name_postfix)
                )
            except ModuleNotFoundError:
                correct_module_name = True
            else:
                module_name_postfix += 1
        module_name = routine_name + "_comp_" + str(module_name_postfix)

        print(f"This will create a module '{module_name}'.")

        # TODO: fix the extension
        subprocess.run(
            [
                "f2py3",
                "-c",
                "-m",
                module_name,
                comparator_file_path,
                tapenade_path + "/ADFirstAidKit/adStack.c",
            ],
            check=True,
            capture_output=True,
        )

        module = import_module(module_name)

        # Move the .so files
        subprocess.run(f"mv *.so {tapenade_output_dir}",
                        shell=True,
                        check=True,
                        capture_output=True)

        return getattr(module, routine_name + "_comp"), [
            arg.name for arg in comparator_arguments
        ]
