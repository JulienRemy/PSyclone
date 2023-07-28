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
# Author J. Remy, Inria

"""This module provides a Transformation for reverse-mode automatic 
differentiation of PSyIR Container nodes."""

from psyclone.transformations import TransformationError

from psyclone.psyir.nodes import Container, Routine
from psyclone.psyir.symbols import RoutineSymbol

from psyclone.autodiff import own_routine_symbol, ADReversalSchedule
from psyclone.autodiff.tapes import ADValueTape
from psyclone.autodiff.transformations import ADTrans

class ADContainerTrans(ADTrans):
    """A class for automatic differentation transformations of Container nodes.
    This is the transformation to apply on the PSyIR AST generated from a source.
    """

    def __init__(self):
        # Transformation should only be applied once.
        # Applying it modifies its attributes.
        self._was_applied = False

        # This stores {RoutineSymbol : ADRoutineTrans}
        # self._routine_transformations = dict()

        # This stores [ADRoutineTrans]
        self._routine_transformations = []

        # This stores {RoutineSymbol : (recording RoutineSymbol,
        #                               returning RoutineSymbol,
        #                               reversing RoutineSymbol)}
        self._routine_map = dict()

        # This stores {RoutineSymbol: ADValueTape}
        # TODO: control flow and loop value_tapes
        self._value_tape_map = dict()

    @property
    def container(self):
        """Returns the container being transformed.

        :return: container.
        :rtype: :py:class:`psyclone.psyir.node.Container`
        """
        return self._container

    @container.setter
    def container(self, container):
        if not isinstance(container, Container):
            raise TypeError(
                f"'container' argument should be of "
                f"type 'Container' but found"
                f"'{type(container).__name__}'."
            )

        self._container = container

    @property
    def routine_transformations(self):
        """Returns the routine transformations used in this container.

        :return: list of routine transformations.
        :rtype: list[:py:class:`psyclone.autodiff.transformations.ADRoutineTrans`]
        """
        return self._routine_transformations

    def add_routine_trans(self, routine_trans):
        """Add a new routine transformations to the list.

        :param routine_trans: routine transformation.
        :type routine_trans: :py:class:`psyclone.autodiff.transformations.ADRoutineTrans`

        :raises TypeError: if routine_trans is of the wrong type.
        """
        from psyclone.autodiff.transformations import ADRoutineTrans

        if not isinstance(routine_trans, ADRoutineTrans):
            raise TypeError(
                f"'value_tape' argument should be of "
                f"type 'ADRoutineTrans' but found"
                f"'{type(routine_trans).__name__}'."
            )
        self._routine_transformations.append(routine_trans)

    @property
    def routine_map(self):
        """Returns the map between the original routine symbols \
        and their three transformed routines symbols.

        :return: dictionnary with original routine symbols as keys \
            and lists of all three transformed routine symbols as values.
        :rtype: dict[:py:class:`psyclone.psyir.symbols.RoutineSymbol`, 
                      list[:py:class:`psyclone.psyir.symbols.RoutineSymbol`]]
        """
        return self._routine_map

    def add_transformed_routines(self, original_symbol, transformed_symbols):
        """Add some transformed routines to the map.

        :param original_symbol: routine symbol of the original.
        :type original_symbol: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        :param transformed_symbols: list of three transformed routine sybmols.
        :type transformed_symbols: list[:py:class:`psyclone.psyir.symbols.RoutineSymbol`]

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
        if len(transformed_symbols) != 3:
            raise ValueError(
                f"'transformed_symbols' argument should be of "
                f"a list of length 3 but found length "
                f"{len(transformed_symbols)}."
            )
        for sym in transformed_symbols:
            if not isinstance(sym, RoutineSymbol):
                raise TypeError(
                    f"'transformed_symbols' argument should be of "
                    f"type 'list[RoutineSymbol]' of length 3 but found "
                    f"an element of type "
                    f"'{type(sym).__name__}'."
                )
        self._routine_map[original_symbol] = transformed_symbols

    @property
    def value_tape_map(self):
        """Returns the map between original routine symbols and value_tapes.

        :return: dictionnary with the original routine symbols as keys \
            and the value_tape as value.
        :rtype: dict[:py:class:`psyclone.psyir.symbols.RoutineSymbol`, \
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
        :type routine_name: `str`
        :param dependent_vars: list of dependent variables names to be \
            differentiated.
        :type dependent_vars: `List[str]`
        :param independent_vars: list of independent variables names to \
            differentiate with respect to.
        :type independent_vars: `List[str]`
        :param reversal_schedule: reversal schedule for routined called \
            inside the one to transform (and inside them, etc.).
        :type reversal_schedule: :py:class:`psyclone.autodiff.ADReversalSchedule`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :raises TransformationError: if the transformation has already been applied.
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
        super().validate(container, options)

        if self._was_applied:
            raise TransformationError(
                "ADContainerTrans instance can only "
                "be applied once but was already "
                "applied."
            )

        if not isinstance(routine_name, str):
            raise TypeError(
                f"'routine_name' argument should be of "
                f"type 'str' but found"
                f"'{type(routine_name).__name__}'."
            )

        if not isinstance(dependent_vars, list):
            raise TypeError(
                f"'dependent_vars' argument should be of "
                f"type 'list' but found"
                f"'{type(dependent_vars).__name__}'."
            )
        for var in dependent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'dependent_vars' argument should be of "
                    f"type 'list[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )

        if not isinstance(independent_vars, list):
            raise TypeError(
                f"'independent_vars' argument should be of "
                f"type 'list' but found"
                f"'{type(independent_vars).__name__}'."
            )
        for var in independent_vars:
            if not isinstance(var, str):
                raise TypeError(
                    f"'independent_vars' argument should be of "
                    f"type 'list[str]' but found an element of type"
                    f"'{type(var).__name__}'."
                )
        if not isinstance(reversal_schedule, ADReversalSchedule):
            raise TypeError(
                f"'reversal_schedule' argument should be of "
                f"type 'ADReversalSchedule' but found"
                f"'{type(reversal_schedule).__name__}'."
            )

        routines = container.walk(Routine)
        routine_names = [routine.name for routine in routines]
        if routine_name not in routine_names:
            raise TransformationError(
                f"Found no Routine named '{routine_name}' "
                f"inside the Container to be transformed."
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

        Options:
        - bool 'jacobian': whether to generate the Jacobian routine. Defaults \
            to False.
        - bool 'verbose' : toggles preceding comment before the Jacobian \
            routine definition. Defaults to False.
        - bool 'simplify': True to apply simplifications after applying AD \
            transformations. Defaults to True.
        - int 'simplify_n_times': number of time to apply simplification \
            rules to BinaryOperation nodes. Defaults to 5.
        - bool 'inline_operation_adjoints': True to inline all possible \
            operation adjoints definitions. Defaults to True.

        :param container: Container Node to the transformed.
        :type container: :py:class:`psyclone.psyir.nodes.Container`
        :param routine_name: name of the Routine to be transformed.
        :type routine_name: `str`
        :param dependent_vars: list of dependent variables names to be \
            differentiated.
        :type dependent_vars: `List[str]`
        :param independent_vars: list of independent variables names to \
            differentiate with respect to.
        :type independent_vars: `List[str]`
        :param reversal_schedule: reversal schedule for routined called \
            inside the one to transform (and inside them, etc.).
        :type reversal_schedule: :py:class:`psyclone.autodiff.ADReversalSchedule`
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :return: a copied and modified container with all necessary \
            Routine definitions.
        :rtype: :py:class:`psyclone.psyir.nodes.Container`
        """
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

        # Symbol
        # routine_symbol = own_routine_symbol(routine)

        # Create the ADRoutineTrans for it
        from psyclone.autodiff.transformations import ADRoutineTrans

        routine_trans = ADRoutineTrans(self)

        # Transform the Routine
        routine_trans.apply(
            routine, dependent_vars, independent_vars, value_tape=None, options=options
        )
        # This adds all necessary entries to self.container and to the maps

        return self.container

    def routine_from_symbol(self, routine_symbol):
        """Get the Routine (definition) associated to a RoutineSymbol if it \
        is present in the Container being transformed.
        Note: this compares routine symbols names since the routine symbols in \
        Call nodes are not necessarily the same as those used in the associated \
        Routine node.

        :param symbol: symbol of the routine.
        :type symbol: :py:class:`psyclone.psyir.symbols.RoutineSymbol`

        :raises TypeError: if routine_symbol is of the wrong type.
        :raises ValueError: if there is no Routine named the same as the \
            argument routine_symbol.

        :return: routine definition.
        :rtype: :py:class:`psyclone.psyir.nodes.Routine`
        """
        if not isinstance(routine_symbol, RoutineSymbol):
            raise TypeError(
                f"'routine_symbol' argument should be of "
                f"type 'RoutineSymbol' but found"
                f"'{type(routine_symbol).__name__}'."
            )

        # Get all Routines and their RoutineSymbols
        routines = self.container.walk(Routine)
        routine_symbols_names = [
            own_routine_symbol(routine).name for routine in routines
        ]

        # Compare the names
        if routine_symbol.name not in routine_symbols_names:
            raise ValueError(
                f"No Routine with RoutineSymbol "
                f"'{routine_symbol.name}' could be found in the "
                f"'container' argument."
            )

        index = routine_symbols_names.index(routine_symbol.name)

        return routines[index]
