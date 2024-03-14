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

"""This module provides an abstract Transformation for reverse-mode automatic 
differentiation in `psyclone.autodiff`. This is the parent class of all 
`AD...Trans` classes."""

from abc import ABCMeta

from psyclone.psyGen import Transformation


class ADTrans(Transformation, metaclass=ABCMeta):
    """An abstract base class for all `psyclone.autodiff` transformations.
    Default options for `apply` are defined here, as a class attribute, \
    as well as option unpacking from a `dict`.
    """

    def kwargs(func):
        def wrapped_func(*args, **kwargs):
            kwargs.update(zip(func.__code__.co_varnames, args))
            return func(**kwargs)
        return wrapped_func

    @kwargs
    def apply()

    def create_trans_dag(self, **kwargs):
        trans_dag = kwargs.get("parent_trans_dag").new_child(self, **kwargs)
        return trans_dag




    # def create_trans_dag_node(self, node):
    #     transformation_dag_node = ADTransDAGNode(node, self)

