.. -----------------------------------------------------------------------------
.. BSD 3-Clause License
..
.. Copyright (c) 2021-2022, Science and Technology Facilities Council.
.. All rights reserved.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions are met:
..
.. * Redistributions of source code must retain the above copyright notice,
..   this list of conditions and the following disclaimer.
..
.. * Redistributions in binary form must reproduce the above copyright notice,
..   this list of conditions and the following disclaimer in the documentation
..   and/or other materials provided with the distribution.
..
.. * Neither the name of the copyright holder nor the names of its
..   contributors may be used to endorse or promote products derived from
..   this software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
.. "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
.. LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
.. FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
.. COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
.. INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
.. BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
.. LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
.. CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
.. LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
.. ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
.. POSSIBILITY OF SUCH DAMAGE.
.. -----------------------------------------------------------------------------
.. Written by J. Remy, Inria

.. only:: html

    .. image:: ../../logo/psyclone_v1.0.png
        :width: 75%
        :align: center
        :alt: PSyclone


PSyclone automatic differentiation module
=========================================

.. only:: html

    PSyclone ``autodiff`` module is PSyclone's **prototype** implementation of 
    source-to-source automatic differentiation. It takes generic Fortran code 
    and applies automatic differentiation in forward- (tangent) or reverse- 
    (adjoint) mode.  

    It is inspired by 
    `Tapenade <https://team.inria.fr/ecuador/en/tapenade/>`_ (see :footcite:t:`tapenade` and :footcite:p:`tapenade-user-guide`), 
    which is also used to perform numerical tests of the transformations, and 
    `OpenAD <https://www.mcs.anl.gov/OpenAD/>`_ (see :footcite:t:`openad`).

    The general approach and transformations rules were adapted from 
    :footcite:t:`griewank-walther`.

.. toctree::
    :hidden:
    :caption: Table of Contents

.. toctree::
    :maxdepth: 2
    :caption: Automatic differentiation

    introduction
    automatic_differentiation

.. toctree::
    :maxdepth: 2
    :caption: PSyclone autodiff module

    getting_started
    implementation
    reverse_mode
    forward_mode

.. when generating latex the index and module index are generated
   automatically and the entries below are rendered as plain text.
.. .. only:: html
	  
..    Indices and tables
..    ==================

..    * :ref:`genindex`
..    * :ref:`modindex`
..    * :ref:`search`

.. footbibliography::