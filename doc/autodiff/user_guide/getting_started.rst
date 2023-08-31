.. -----------------------------------------------------------------------------
.. BSD 3-Clause License
..
.. Copyright (c) 2021-2022, Science and Technology Facilities Council.
.. All rights reserved.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions are met:
..
.. * Redistributions of source code must retain the above copyright notice, this
..   list of conditions and the following disclaimer.
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
.. Written by J. Remy, Université Grenoble Alpes, Inria

.. _getting_started:

Getting started
===============

.. _download:

Download
--------

PSyclone ``autodiff`` module is hosted on GitHub: https://github.com/JulienRemy/PSyclone/tree/automatic_differentiation. 

It is currently an **experimental protoype**. The latest version is the ``automatic_differentiation`` branch.

To download it, clone the repository then checkout the branch using 

.. code-block:: console

    $ git clone https://github.com/JulienRemy/PSyclone.git
    $ git checkout automatic_differentiation

.. _env_dependencies:

Environment and dependencies
----------------------------

Please follow the instructions regarding environments and dependencies 
at https://psyclone.readthedocs.io/en/stable/getting_going.html.

This module also requires NumPy, which can be installed using ``pip``:

.. code-block:: console

    $ pip install numpy

The tutorials also require Jupyter Notebook, which can be installed using ``pip``:

.. code-block:: console

    $ pip install jupyter

.. _installing:

Installing
----------

PSyclone and its ``autodiff`` module can then be installed using ``pip``:

.. code-block:: console

    $ cd <PSYCLONE_HOME>
    $ pip install [--user] .

or using ``setup.py``:

.. code-block:: console

    $ cd <PSYCLONE_HOME>
    $ python setup.py install

.. _autodiff_tutorial:

Tutorial
--------

See the `src/psyclone/autodiff/tutorials/ <https://github.com/JulienRemy/PSyclone/tree/automatic_differentiation/src/psyclone/autodiff/tutorials>`_ directory for a Jupyter Notebook
tutorial detailling the use of the module in reverse-mode.

To open it using Jupyter Notebook:

.. code-block:: console

    $ jupyter-notebook tuto1.ipynb

