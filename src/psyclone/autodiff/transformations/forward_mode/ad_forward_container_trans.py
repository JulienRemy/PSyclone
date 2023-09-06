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

"""This module provides a Transformation for forward-mode automatic 
differentiation of PSyIR Container nodes."""


from psyclone.psyir.nodes import Routine
from psyclone.psyir.symbols import RoutineSymbol

from psyclone.autodiff.transformations import ADContainerTrans


class ADForwardContainerTrans(ADContainerTrans):
    """A class for automatic differentation transformation of Container nodes \
    in foward-mode.
    This is the transformation to apply on the PSyIR AST generated from a \
    source.
    """

    def add_routine_trans(self, routine_trans):
        """Add a new routine transformations to the list.

        :param routine_trans: routine transformation.
        :type routine_trans: \
            :py:class:`psyclone.autodiff.transformations.ADForwardRoutineTrans`

        :raises TypeError: if routine_trans is of the wrong type.
        """
        # Avoid circular import
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADForwardRoutineTrans

        if not isinstance(routine_trans, ADForwardRoutineTrans):
            raise TypeError(
                f"'routine_trans' argument should be of "
                f"type 'ADForwardRoutineTrans' but found"
                f"'{type(routine_trans).__name__}'."
            )
        self._routine_transformations.append(routine_trans)

    def add_transformed_routine(self, original_symbol, transformed_symbol):
        """Add a transformed routine to the map.

        :param original_symbol: routine symbol of the original.
        :type original_symbol: :py:class:`psyclone.psyir.symbols.RoutineSymbol`
        :param transformed_symbols: transformed routine symbol.
        :type transformed_symbols: \
                                :py:class:`psyclone.psyir.symbols.RoutineSymbol`

        :raises TypeError: if original_symbol is of the wrong type.
        :raises TypeError: if transformed_symbol is of the wrong type.
        """
        if not isinstance(original_symbol, RoutineSymbol):
            raise TypeError(
                f"'original_symbol' argument should be of "
                f"type 'RoutineSymbol' but found "
                f"'{type(original_symbol).__name__}'."
            )
        if not isinstance(transformed_symbol, RoutineSymbol):
            raise TypeError(
                f"'transformed_symbol' argument should be of "
                f"type 'RoutineSymbol' but found "
                f"'{type(transformed_symbol).__name__}'."
            )

        self._routine_map[original_symbol] = transformed_symbol

    def apply(
        self,
        container,
        routine_name,
        dependent_vars,
        independent_vars,
        options=None,
    ):
        """Applies the transformation, returning a new container with routine \
        definitions using the forward-mode of automatic differentiation.

        | Options:
        | - bool 'jacobian': whether to generate the Jacobian routine. \
                           Defaults to False.
        | - bool 'verbose' : toggles explanatory comments. Defaults to False.
        | - bool 'simplify': True to apply simplifications after applying AD \
                           transformations. Defaults to True.
        | - int 'simplify_n_times': number of time to apply simplification \
                                  rules to BinaryOperation nodes. Defaults to 5.

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
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: a copied and modified container with all necessary \
                 Routine definitions.
        :rtype: :py:class:`psyclone.psyir.nodes.Container`
        """
        # pylint: disable=too-many-arguments

        self.validate(
            container,
            routine_name,
            dependent_vars,
            independent_vars,
            options,
        )

        self._was_applied = True

        # Container (being transformed)
        self.container = container.copy()

        # All Routine nodes and their names
        routines = self.container.walk(Routine)
        routine_names = [routine.name for routine in routines]

        # Routine to be transformed
        index = routine_names.index(routine_name)
        routine = routines[index]

        # Create the ADForwardRoutineTrans for it
        # Avoid circular import
        # pylint: disable=import-outside-toplevel
        from psyclone.autodiff.transformations import ADForwardRoutineTrans
        routine_trans = ADForwardRoutineTrans(self)

        # Transform the Routine
        routine_trans.apply(
            routine, dependent_vars, independent_vars, options=options
        )
        # This adds all necessary entries to self.container and to the maps

        return self.container
