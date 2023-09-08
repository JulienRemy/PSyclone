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

"""A module to perform tests on the autodiff ADTrans class.
"""

import pytest

from psyclone.autodiff.transformations import ADTrans

# ADTrans is an ABC due to its 'apply' abstractclassmethod.
# Create a derived class for testing purposes.

class ADTransTest(ADTrans):
    def apply(self, node, options=None):
        pass

def test_ad_trans_typecheck_options():
    ad_trans = ADTransTest()
    with pytest.raises(TypeError) as info:
        ad_trans.typecheck_options(1)
    assert (
        "'options' argument should be of "
        "type 'dict' or 'NoneType' but found "
        "'int'." in str(info.value)
    )
    with pytest.raises(TypeError) as info:
        ad_trans.typecheck_options({'key1' : 1, 2: 3})
    assert (
        "'options' argument should be a 'dict' "
        "with keys of type 'str' but found a key "
        "'2' of type 'int'." in str(info.value)
    )

def test_ad_trans_unpack_option_errors():
    ad_trans = ADTransTest()
    ad_trans.default_options = {'other_key' : 2}

    with pytest.raises(TypeError) as info:
        ad_trans.unpack_option(None, None)
    assert (
        "'name' argument should be of "
        "type 'str' but found "
        "'NoneType'." in str(info.value)
    )
    with pytest.raises(TypeError) as info:
        ad_trans.unpack_option('key', 2)
    assert (
        "'options' argument should be of "
        "type 'dict' or 'NoneType' but found "
        "'int'." in str(info.value)
    )
    with pytest.raises(TypeError) as info:
        ad_trans.unpack_option('key', {'key' : 1, 2: 3})
    assert (
        "'options' argument should be a 'dict' "
        "with keys of type 'str' but found a key "
        "'2' of type 'int'." in str(info.value)
    )
    with pytest.raises(KeyError) as info:
        ad_trans.unpack_option('key', {'other_key' : 1})
    assert (
        "'name' argument 'key' "
        "is neither a key of 'options' nor "
        "of 'self.default_options'." in str(info.value)
    )


def test_ad_trans_unpack_option():
    ad_trans = ADTransTest()
    ad_trans.default_options = {'other_key' : 2}

    assert ad_trans.unpack_option('key', {'key' : 1, 'other_key' : 2}) == 1
    assert ad_trans.unpack_option('other_key', {'key' : 1}) == 2
    assert ad_trans.unpack_option('key', {'key' : 1}) == 1
    assert ad_trans.unpack_option('other_key', None) == 2
