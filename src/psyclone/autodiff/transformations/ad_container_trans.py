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

"""This module provides an abstract Transformation for automatic 
differentiation of PSyIR Container nodes."""

from abc import ABCMeta, abstractmethod

from psyclone.transformations import TransformationError

from psyclone.psyir.nodes import Container, Routine
from psyclone.psyir.symbols import RoutineSymbol

from psyclone.autodiff import own_routine_symbol
from psyclone.autodiff.transformations import ADTrans


class ADContainerTrans(ADTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformation \
    of Container nodes.
    This is the transformation to apply on the PSyIR AST generated from a source.
    """

    def __init__(self):
        # Transformation should only be applied once.
        # Applying it modifies its attributes.
        self._was_applied = False

        # This stores [ADRoutineTrans]
        self._routine_transformations = []

        # This stores {RoutineSymbol : transformed RoutineSymbol(s)}
        self._routine_map = dict()

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
        :rtype: List[Union[:py:class:`psyclone.autodiff.transformations.ADForwardRoutineTrans`,
                           :py:class:`psyclone.autodiff.transformations.ADReverseRoutineTrans`]]
        """
        return self._routine_transformations

    @abstractmethod
    def add_routine_trans(self, routine_trans):
        """Add a new routine transformations to the list.

        :param routine_trans: routine transformation.
        :type routine_trans: Union[:py:class:`psyclone.autodiff.transformations.ADForwardRoutineTrans`,
                                   :py:class:`psyclone.autodiff.transformations.ADReverseRoutineTrans`]
        """

    @property
    def routine_map(self):
        """Returns the map between the original routine symbols \
        and their transformed routines symbols.

        :return: dictionnary with original routine symbols as keys \
            and lists of all transformed routine symbols as values.
        :rtype: dict[:py:class:`psyclone.psyir.symbols.RoutineSymbol`, 
                      list[:py:class:`psyclone.psyir.symbols.RoutineSymbol`]]
        """
        return self._routine_map

    def validate(
        self,
        container,
        routine_name,
        dependent_vars,
        independent_vars,
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
        :raises TransformationError: if no Routine named routine_name can \
            be found in the container.
        """
        super().validate(container, options)

        if self._was_applied:
            raise TransformationError(
                "ADReverseContainerTrans instance can only "
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

        routines = container.walk(Routine)
        routine_names = [routine.name for routine in routines]
        if routine_name not in routine_names:
            raise TransformationError(
                f"Found no Routine named '{routine_name}' "
                f"inside the Container to be transformed."
            )

    @abstractmethod
    def apply(
        self,
        container,
        routine_name,
        dependent_vars,
        independent_vars,
        options=None,
    ):
        """Applies the transformation.

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
        :param options: a dictionary with options for transformations, \
            defaults to None.
        :type options: Optional[Dict[str, Any]]

        :return: a copied and modified container with all necessary \
            Routine definitions.
        :rtype: :py:class:`psyclone.psyir.nodes.Container`
        """

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
