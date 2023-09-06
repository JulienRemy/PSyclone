# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2021-2023, Science and Technology Facilities Council.
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

"""This module provides a Transformation for reverse-mode automatic 
differentiation of PSyIR Container nodes."""


from psyclone.psyir.nodes import Routine
from psyclone.psyir.symbols import RoutineSymbol

from psyclone.autodiff import ADReversalSchedule
from psyclone.autodiff.tapes import ADValueTape
from psyclone.autodiff.transformations import ADContainerTrans


class ADReverseContainerTrans(ADContainerTrans):
    """A class for automatic differentation transformation of Container nodes \
    in reverse-mode.
    This is the transformation to apply on the PSyIR AST generated from a \
    source.
    """

    def __init__(self):
        # pylint: disable=use-dict-literal
        super().__init__()

        # This stores {RoutineSymbol: ADValueTape}
        # TODO: control flow and loop value_tapes
        self._value_tape_map = dict()

    def add_routine_trans(self, routine_trans):
        """Add a new routine transformations to the list.

        :param routine_trans: routine transformation.
        :type routine_trans: \
            :py:class:`psyclone.autodiff.transformations.ADReverseRoutineTrans`

        :raises TypeError: if routine_trans is of the wrong type.
        """
        # Avoid circular import
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseRoutineTrans

        if not isinstance(routine_trans, ADReverseRoutineTrans):
            raise TypeError(
                f"'routine_trans' argument should be of "
                f"type 'ADReverseRoutineTrans' but found"
                f"'{type(routine_trans).__name__}'."
            )
        self._routine_transformations.append(routine_trans)

    def add_transformed_routines(self, original_symbol, transformed_symbols):
        """Add some transformed routines to the map.

        :param original_symbol: routine symbol of the original.
        :type original_symbol: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        :param transformed_symbols: List of three transformed routine sybmols.
        :type transformed_symbols: \
                        List[:py:class:`psyclone.psyir.symbols.RoutineSymbol`]

        :raises TypeError: if original_symbol is of the wrong type.
        :raises TypeError: if transformed_symbol is of the wrong type.
        :raises ValueError: if the length of transformed_symbol is not 3.
        :raises TypeError: if any of the elements of transformed_symbol are \
                           of the wrong type.
        """
        if not isinstance(original_symbol, RoutineSymbol):
            raise TypeError(
                f"'original_symbol' argument should be of "
                f"type 'RoutineSymbol' but found "
                f"'{type(original_symbol).__name__}'."
            )
        if not isinstance(transformed_symbols, list):
            raise TypeError(
                f"'transformed_symbols' argument should be of "
                f"type 'list[RoutineSymbol]' of length 3 but found "
                f"'{type(transformed_symbols).__name__}'."
            )
        for sym in transformed_symbols:
            if not isinstance(sym, RoutineSymbol):
                raise TypeError(
                    f"'transformed_symbols' argument should be of "
                    f"type 'list[RoutineSymbol]' but found "
                    f"an element of type "
                    f"'{type(sym).__name__}'."
                )

        if len(transformed_symbols) != 3:
            raise ValueError(
                f"'transformed_symbols' argument should be of "
                f"a list of length 3 but found length "
                f"{len(transformed_symbols)}."
            )

        self._routine_map[original_symbol] = transformed_symbols

    @property
    def value_tape_map(self):
        """Returns the map between original routine symbols and value_tapes.

        :return: dictionnary with the original routine symbols as keys \
                 and the value_tape as value.
        :rtype: Dict[:py:class:`psyclone.psyir.symbols.RoutineSymbol`, \
                     :py:class:`psyclone.autodiff.ADValueTape`]
        """
        return self._value_tape_map

    def add_value_tape(self, routine_symbol, value_tape):
        """Add a new value_tape to the map.

        :param routine_symbol: routine symbol of the original.
        :type routine_symbol: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        :param value_tape: value_tape used by the transformed routines.
        :type value_tape: :py:class:`psyclone.autodiff.ADValueTape`

        :raises TypeError: if routine_symbol is of the wrong type.
        :raises TypeError: if value_tape is of the wrong type.
        """
        if not isinstance(routine_symbol, RoutineSymbol):
            raise TypeError(
                f"'routine_symbol' argument should be of "
                f"type 'RoutineSymbol' but found"
                f"'{type(routine_symbol).__name__}'."
            )
        if not isinstance(value_tape, ADValueTape):
            raise TypeError(
                f"'value_tape' argument should be of "
                f"type 'ADValueTape' but found"
                f"'{type(value_tape).__name__}'."
            )
        self._value_tape_map[routine_symbol] = value_tape

    @property
    def reversal_schedule(self):
        """Returns the reversal schedule used to transform nested routine \
        calls.

        :return: reversal schedule.
        :rtype: :py:class:`psyclone.autodiff.ADReversalSchedule`
        """
        return self._reversal_schedule

    @reversal_schedule.setter
    def reversal_schedule(self, reversal_schedule):
        if not isinstance(reversal_schedule, ADReversalSchedule):
            raise TypeError(
                f"'reversal_schedule' argument should be of "
                f"type 'RoutineSymbol' but found"
                f"'{type(reversal_schedule).__name__}'."
            )
        self._reversal_schedule = reversal_schedule

    def validate(
        self,
        container,
        routine_name,
        dependent_vars,
        independent_vars,
        reversal_schedule,
        options=None,
    ):
        """Validates the arguments of the `apply` method.

        :param container: Container Node to the transformed.
        :type container: :py:class:`psyclone.psyir.nodes.Container`
        :param routine_name: name of the Routine to be transformed.
        :type routine_name: `Str`
        :param dependent_vars: list of dependent variables names to be \
                               differentiated.
        :type dependent_vars: `List[Str]`
        :param independent_vars: list of independent variables names to \
                                 differentiate with respect to.
        :type independent_vars: `List[Str]`
        :param reversal_schedule: reversal schedule for routined called \
                                  inside the one to transform 
                                  (and inside them, etc.).
        :type reversal_schedule: :py:class:`psyclone.autodiff.ADReversalSchedule`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if the transformation has already been \
                                     applied.
        :raises TransformationError: if container is of the wrong type.
        :raises TypeError: if routine_name is of the wrong type.
        :raises TypeError: if dependent_vars is of the wrong type.
        :raises TypeError: if at least one element of dependent_vars is \
                           of the wrong type.
        :raises TypeError: if independent_vars is of the wrong type.
        :raises TypeError: if at least one element of independent_vars is \
                           of the wrong type.
        :raises TypeError: if reversal_schedule is of the wrong type.
        :raises TransformationError: if no Routine named routine_name can \
                                     be found in the container.
        """
        # pylint: disable=arguments-renamed, too-many-arguments

        super().validate(
            container, routine_name, dependent_vars, independent_vars, options
        )

        if not isinstance(reversal_schedule, ADReversalSchedule):
            raise TypeError(
                f"'reversal_schedule' argument should be of "
                f"type 'ADReversalSchedule' but found"
                f"'{type(reversal_schedule).__name__}'."
            )

    def apply(
        self,
        container,
        routine_name,
        dependent_vars,
        independent_vars,
        reversal_schedule,
        options=None,
    ):
        """Applies the transformation, returning a new container with routine \
        definitions for both motions using the reverse-mode of automatic \
        differentiation.

        | Options:
        | - bool 'jacobian': whether to generate the Jacobian routine. Defaults \
                           to False.
        | - bool 'verbose' : toggles explanatory comments. Defaults to False.
        | - bool 'simplify': True to apply simplifications after applying AD \
                           transformations. Defaults to True.
        | - int 'simplify_n_times': number of time to apply simplification \
                                  rules to BinaryOperation nodes. Defaults to 5.
        | - bool 'inline_operation_adjoints': True to inline all possible \
                                            operation adjoints definitions. \
                                            Defaults to True.

        :param container: Container Node to the transformed.
        :type container: :py:class:`psyclone.psyir.nodes.Container`
        :param routine_name: name of the Routine to be transformed.
        :type routine_name: `Str`
        :param dependent_vars: list of dependent variables names to be \
                               differentiated.
        :type dependent_vars: `List[Str]`
        :param independent_vars: list of independent variables names to \
                                 differentiate with respect to.
        :type independent_vars: `List[str]`
        :param reversal_schedule: reversal schedule for routined called \
                                  inside the one to transform \
                                  (and inside them, etc.).
        :type reversal_schedule: :py:class:`psyclone.autodiff.ADReversalSchedule`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: a copied and modified container with all necessary \
                 Routine definitions.
        :rtype: :py:class:`psyclone.psyir.nodes.Container`
        """
        # pylint: disable=arguments-renamed, too-many-arguments

        self.validate(
            container,
            routine_name,
            dependent_vars,
            independent_vars,
            reversal_schedule,
            options,
        )

        self._was_applied = True

        # Container (being transformed)
        self.container = container.copy()

        # Reversal schedule for the transformation
        self.reversal_schedule = reversal_schedule

        # All Routine nodes and their names
        routines = self.container.walk(Routine)
        routine_names = [routine.name for routine in routines]

        # Routine to be transformed
        index = routine_names.index(routine_name)
        routine = routines[index]

        # Create the ADReverseRoutineTrans for it
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADReverseRoutineTrans

        routine_trans = ADReverseRoutineTrans(self)

        # Transform the Routine
        routine_trans.apply(
            routine,
            dependent_vars,
            independent_vars,
            value_tape=None,
            options=options,
        )
        # This adds all necessary entries to self.container and to the maps

        return self.container
