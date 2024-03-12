# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2024, Science and Technology Facilities Council.
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

"""This module provides several enums for describing the motions of automatic \
differentiation and an abstract class to use as a parent of all classes \
derived from the PSyIR Symbol and Node ones.
"""
from abc import ABCMeta, abstractmethod
from enum import Enum


class ADMotion(Enum):
    """Enum describing the 'motion' a PSyIR node belongs to.
    Advancing is the original (non-differentiated) motion.
    Tangent is the result of forward-mode automatic differentiation.
    Recording is the 'forward' motion of reverse-mode automatic \
    differentiation, which computes (and eventually tapes) primal values.
    Returning is the 'backward' motion of reverse-mode automatic \
    differentiation, which effectively computes the adjoint, using taped \
    primal values or recomputations where needed.
    Reversing is the combination of both the recording and returning motions.
    """

    ADVANCING = 1
    TANGENT = 2
    RECORDING = 3
    RETURNING = 4
    REVERSING = 5


class ADPSyIR(object, metaclass=ABCMeta):
    """Abstract class, parent of all ADNode and ADSymbol derived classes,
    simply for convenient typechecking and distinguishing between vanilla PSyIR
    and the AD version.
    """
    _transformation = None

    @abstractmethod
    def __init__(self):
        pass

    # @property
    # def transformed_to(self):
    #     return self._transformed_to
    
    # @property
    # def transformed_from(self):
    #     return self._transformed_from
    
    # def log_transformation(self, ad_trans_dag_node):
    #     if not isinstance(ad_trans_dag_node, ADTransDAGNode):
    #         raise TypeError("")
    #     ad_trans = ad_trans_dag_node.transformation
    #     if not isinstance(ad_trans, self._transformation):
    #         raise TypeError("")
        
    #     source = ad_trans_dag_node.source
    #     target = ad_trans_dag_node.target
    #     if self == source:
    #         self._transformed_to.append(ad_trans_dag_node)
    #     if self == target:
    #         self._transformed_from = ad_trans_dag_node
        


    @classmethod
    def from_psyir(cls, psyir):
        from psyclone.psyir.symbols import DataSymbol, RoutineSymbol
        from psyclone.psyir.nodes import (
            Literal,
            Reference,
            UnaryOperation,
            BinaryOperation,
            Assignment,
            Loop,
            IfBlock,
            Call,
            IntrinsicCall,
            Schedule,
            Routine,
            Range,
            ArrayReference,
        )
        from psyclone.autodiff.psyir.symbols import (
            ADVariableSymbol,
            ADRoutineSymbol,
        )
        from psyclone.autodiff.psyir.nodes import (
            ADLiteral,
            ADReference,
            ADUnaryOperation,
            ADBinaryOperation,
            ADAssignment,
            ADLoop,
            ADIfBlock,
            ADCall,
            ADIntrinsicCall,
            ADSchedule,
            ADRoutine,
            ADRange,
            ADArrayReference,
        )

        psyir_to_AD = {
            DataSymbol: ADVariableSymbol,
            RoutineSymbol: ADRoutineSymbol,
            Literal: ADLiteral,
            Reference: ADReference,
            UnaryOperation: ADUnaryOperation,
            BinaryOperation: ADBinaryOperation,
            Assignment: ADAssignment,
            Loop: ADLoop,
            IfBlock: ADIfBlock,
            Call: ADCall,
            IntrinsicCall: ADIntrinsicCall,
            Schedule: ADSchedule,
            Routine: ADRoutine,
            Range: ADRange,
            ArrayReference: ADArrayReference,
        }
        psyir_type = type(psyir)
        # # Only keep cls subclasses in the hashmap,
        # # so that eg. ADReference.from_psyir(ref) cannot create an ADLoop
        # new_dict = dict()
        # for psyir_type, ad_psyir_type in psyir_to_AD.items():
        #     if issubclass(ad_psyir_type, cls):
        #         new_dict[psyir_type] = ad_psyir_type
        # psyir_to_AD = new_dict
        # # Already an ADPSyIR instance, return it
        # if psyir_type in psyir_to_AD.values():
        #     return psyir
        # PSyIR not in map
        if psyir_type not in psyir_to_AD:
            raise TypeError(f"{psyir_type.__name__}")
        # PSyIR -> ADPSyIR
        ad_type = psyir_to_AD[psyir_type]
        return ad_type.from_psyir(psyir)
