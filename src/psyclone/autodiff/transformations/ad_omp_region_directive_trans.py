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

"""This module provides an abstract Transformation for automatic differentiation
of PSyIR OMPRegionDirective nodes and its derived classes in both modes.
"""

from abc import ABCMeta, abstractmethod

from psyclone.psyir.nodes import (
    OMPRegionDirective,
    OMPParallelDirective,
    OMPParallelDoDirective,
    OMPDoDirective,
)
from psyclone.psyir.transformations import TransformationError

from psyclone.autodiff.transformations import ADElementTrans


class ADOMPRegionDirectiveTrans(ADElementTrans, metaclass=ABCMeta):
    """An abstract class for automatic differentation transformations of \
    OMPRegionDirective nodes and its derived classes.
    """

    def validate(self, omp_region, options=None):
        """Validates the arguments of the `apply` method.

        :param omp_region: node to be transformed.
        :type omp_region: :py:class:`psyclone.psyir.nodes.OMPRegionDirective`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]

        :raises TransformationError: if omp_region is of the wrong type.
        """
        # pylint: disable=arguments-renamed

        super().validate(omp_region, options)

        if not isinstance(omp_region, OMPRegionDirective):
            raise TransformationError(
                f"'omp_region' argument should be a "
                f"PSyIR 'OMPRegionDirective' but found '{type(omp_region).__name__}'."
            )

        if not isinstance(
            omp_region,
            (OMPDoDirective, OMPParallelDirective, OMPParallelDoDirective),
        ):
            raise NotImplementedError(
                "Only OMPDoDirective, "
                "OMPParallelDoDirective and "
                "OMPParallelDirective were implemented "
                "but found "
                f"'{type(omp_region).__name__}'."
            )

    @abstractmethod
    def apply(self, omp_region, options=None):
        """Applies the transformation, generating the recording and returning \
        motions associated to this OMPRegionDirective.

        :param omp_region: node to be transformed.
        :type omp_region: :py:class:`psyclone.psyir.nodes.OMPRegionDirective`
        :param options: a dictionary with options for transformations, \
                        defaults to None.
        :type options: Optional[Dict[Str, Any]]
        """
        # pylint: disable=arguments-renamed, unnecessary-pass
        pass
