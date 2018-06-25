# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2017-2018, Science and Technology Facilities Council
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
# Authors R. W. Ford and A. R. Porter, STFC Daresbury Lab

'''PSyclone configuration file where system wide properties and
defaults are set.'''

SUPPORTEDAPIS = ["gunghoproto", "dynamo0.1", "dynamo0.3", "gocean0.1",
                 "gocean1.0"]
DEFAULTAPI = "dynamo0.3"
SUPPORTEDSTUBAPIS = ["dynamo0.3"]
DEFAULTSTUBAPI = "dynamo0.3"
DISTRIBUTED_MEMORY = True
REPRODUCIBLE_REDUCTIONS = False
# Ammount to pad the local summation array when REPRODUCIBLE_REDUCTIONS is True
REPROD_PAD_SIZE = 8
# Specify whether we compute annexed dofs when a kernel is written so
# that it iterates over dofs. This is currently only the case for
# builtins. If annexed dofs are computed then in certain cases we
# remove the need for a halo exchange call.
COMPUTE_ANNEXED_DOFS = True
