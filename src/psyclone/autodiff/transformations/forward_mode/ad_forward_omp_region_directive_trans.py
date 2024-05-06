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
differentiation of PSyIR OMPRegionDirective nodes."""

from psyclone.psyir.nodes import (
    OMPRegionDirective,
    OMPParallelDirective,
    OMPParallelDoDirective,
    OMPDoDirective,
    Assignment,
    Call,
    Loop,
    IfBlock,
    OMPDefaultClause,
    OMPPrivateClause,
    OMPSharedClause,
    OMPFirstprivateClause,
    Reference,
    Schedule,
)
from psyclone.psyir.symbols import ScalarType

from psyclone.autodiff import zero
from psyclone.autodiff.transformations import ADOMPRegionDirectiveTrans

# TODO: Reduction, Atomic, Barrier, etc.


class ADForwardOMPRegionDirectiveTrans(ADOMPRegionDirectiveTrans):
    """A class for automatic differentation transformations of \
    OMPRegionDirective nodes and derived classes using forward-mode.
    Requires an ADForwardRoutineTrans instance as context, where the \
    derivative symbols can be found.
    """

    def apply(self, omp_region, options=None):
        """Applies the transformation, generating the transformed \
        statement associated with this OMPRegionDirective.
        This is the same type of directive, with a differentiated body and \
        derivative variables using the same clauses as the primal ones.
            
        | Options:
        | - bool 'verbose' : toggles preceding and inline comments around the \
            derivatives of assignment statements in the transformed motion.

        :param omp_region: omp region directive to be transformed.
        :type omp_region: :py:class:`psyclone.psyir.nodes.OMPRegionDirective`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :return: transformed omp parallel directive.
        :rtype: :py:class:`psyclone.psyir.nodes.OMPRegionDirective`
        """
        self.validate(omp_region, options)

        clauses = omp_region.clauses
        transformed_clauses, new_private_derivatives = self.transform_clauses(
            clauses, options
        )

        body = omp_region.dir_body
        transformed_body = self.transform_body(body, options)

        # This is either an OMPDoDirective or OMPParallelDoDirective
        if isinstance(omp_region, (OMPDoDirective, OMPParallelDoDirective)):
            transformed = type(omp_region)(
                children=transformed_body,
                omp_schedule=omp_region.omp_schedule,
                collapse=omp_region.collapse,
                reprod=omp_region.reprod,
            )
            for i, clause in enumerate(transformed_clauses):
                transformed.children[i + 1] = clause

        elif isinstance(omp_region, OMPParallelDirective):
            transformed = OMPParallelDirective(children=transformed_body)
            for i, clause in enumerate(transformed_clauses):
                transformed.children[i + 1] = clause

        else:
            raise NotImplementedError(
                f"Unsupported OMPRegionDirective type '{type(omp_region).__name__}'."
            )
        
        transformed = self.assign_zero_to_new_private_differentials(
            transformed, new_private_derivatives
        )
    
        return transformed

    def transform_clauses(self, clauses, options=None):
        """Transforms all OpenMP clauses of this OMPRegionDirective.
        In forward-mode this replicates the primal variable clause for the \
        associated derivative variable.

        :param clauses: list of OpenMP clauses.
        :type clauses: List[ \
                Union[:py:class:`psyclone.psyir.nodes.OMPPrivateClause`, \
                      :py:class:`psyclone.psyir.nodes.OMPFirstPrivateClause`, \
                      :py:class:`psyclone.psyir.nodes.OMPSharedClause`, \
                      :py:class:`psyclone.psyir.nodes.OMPDefaultClause`]]
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if clauses is of the wrong type.
        :raises TypeError: if an item of clauses is of the wrong type.
        :raises NotImplementedError: if an unsupported clause is included.

        :return: list of transformed clauses and list of new private derivatives
                 to which to assign 0.0 at the beginning of the parallel region.
        :rtype: List[Union[\
                        :py:class:`psyclone.psyir.nodes.OMPPrivateClause`,\
                        :py:class:`psyclone.psyir.nodes.OMPFirstPrivateClause`,\
                        :py:class:`psyclone.psyir.nodes.OMPSharedClause`,\
                        :py:class:`psyclone.psyir.nodes.OMPDefaultClause`]], \
                List[:py:class:`psyclone.psyir.nodes.DataSymbol`]
        """
        if not isinstance(clauses, list):
            raise TypeError(
                f"'clauses' argument should be of type "
                f"'list' but found '{type(clauses).__name__}'."
            )
        for clause in clauses:
            if not isinstance(
                clause,
                (
                    OMPPrivateClause,
                    OMPFirstprivateClause,
                    OMPSharedClause,
                    OMPDefaultClause,
                ),
            ):
                raise TypeError(
                    f"'clauses' argument should be a list of OMP clauses "
                    f"but found an item of type '{type(clause).__name__}'."
                )

        transformed_clauses = []
        new_private_derivatives = []
        for clause in clauses:
            if isinstance(clause, (OMPPrivateClause, OMPFirstprivateClause)):
                symbols = []
                for reference in clause.children:
                    symbol = reference.symbol
                    symbols.append(symbol)
                    if symbol.datatype.intrinsic is ScalarType.Intrinsic.REAL:
                        diff_symbol = (
                            self.routine_trans.data_symbol_differential_map[
                                symbol
                            ]
                        )
                        symbols.append(diff_symbol)
                        new_private_derivatives.append(diff_symbol)
                transformed_clause = type(clause).create(symbols)
                transformed_clauses.append(transformed_clause)
            elif isinstance(clause, OMPFirstprivateClause):
                symbols = []
                for reference in clause.children:
                    symbol = reference.symbol
                    symbols.append(symbol)
                    if symbol.datatype.intrinsic is ScalarType.Intrinsic.REAL:
                        diff_symbol = (
                            self.routine_trans.data_symbol_differential_map[
                                symbol
                            ]
                        )
                        symbols.append(diff_symbol)
                transformed_clause = type(clause).create(symbols)
                transformed_clauses.append(transformed_clause)
            elif isinstance(clause, OMPSharedClause):
                raise NotImplementedError("OMPSharedClause are unused for now.")
                references = []
                for reference in clause.children:
                    symbol = reference.symbol
                    diff_symbol = (
                        self.routine_trans.data_symbol_differential_map[symbol]
                    )
                    references.append(reference.copy())
                    references.append(Reference(diff_symbol))
                transformed_clause = OMPSharedClause(children=references)
                transformed_clauses.append(transformed_clause)
            elif isinstance(clause, OMPDefaultClause):
                transformed_clauses.append(clause.copy())
            else:
                raise NotImplementedError(
                    "Only OMPPrivateClause, "
                    "OMPFirstPrivateClause, "
                    "OMPSharedClause "
                    "and OMPDefaultClause are "
                    "implemented but found "
                    f"{type(clause).__name__}."
                )

        return transformed_clauses, new_private_derivatives

    def transform_body(self, body, options=None):
        """Transforms all statements found in the OMPRegionDirective body.

        :param body: body to transform, as a PSyIR Schedule.
        :type body: :py:class:`psyclone.psyir.nodes.Schedule`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TypeError: if body is of the wrong type.

        :return: list of transformed nodes.
        :rtype: List[:py:class:`psyclone.psyir.nodes.Node`]
        """
        if not isinstance(body, Schedule):
            raise TypeError(
                f"'body' argument should be of type "
                f"'Schedule' but found '{type(body).__name__}'."
            )

        transformed_body = []

        for node in body.children:
            if isinstance(node, Assignment):
                transformed_body.extend(
                    self.routine_trans.assignment_trans.apply(node, options)
                )
            elif isinstance(node, Call):
                transformed_body.extend(
                    self.routine_trans.call_trans.apply(node, options)
                )
            elif isinstance(node, IfBlock):
                transformed_body.append(
                    self.routine_trans.if_block_trans(node, options)
                )
            elif isinstance(node, Loop):
                transformed_body.append(
                    self.routine_trans.loop_trans.apply(node, options)
                )
            else:
                raise NotImplementedError(
                    f"Transformations for "
                    f"'{type(node).__name__}' found in "
                    f"OMPRegionDirective body were not implemented "
                    f"yet."
                )

        return transformed_body
