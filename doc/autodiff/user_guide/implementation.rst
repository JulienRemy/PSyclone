.. -----------------------------------------------------------------------------
.. BSD 3-Clause License
..
.. Copyright (c) 2021-2023, Science and Technology Facilities Council.
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
.. Written by J. Remy, Inria

.. _implementation:


Implementation
==============

This module performs source-to-source automatic differentiation of a target 
routine, in which dependent variables are differentiated with respect to 
independent variables.

This is implemented in PSyclone by parsing the source code file containing the 
target routine, and eventually the routines it calls, transforming it into a 
PSyIR AST and applying automatic differentiation transformations in either
:ref:`forward-mode<forward_mode>` or :ref:`reverse-mode<reverse_mode>`
to the nodes thus obtained. The resulting PSyIR tree is then written to 
Fortran source code.

.. _implemented_features:

Implemented features
++++++++++++++++++++

For now, only Fortran subroutines *ie.* neither functions nor programs can be 
transformed. 
The implementation only deals with **scalar** variables, which is to say that 
subroutines containing arrays, either as local variables or as arguments cannot 
yet be transformed.  
The statements the target routine (and eventual routines it calls) may contain 
are : 

- assignments (PSyIR ``Assignment`` nodes),
- calls to subroutines (PSyIR ``Call`` nodes).

These statements may contain unary and binary linear or non-linear operations 
(PSyIR ``UnaryOperation`` and ``BinaryOperation`` nodes).

An optional ``verbose`` mode is available, which is especially useful when 
examining the transformed statements and routines in reverse-mode.

Basic simplification and substitution rules can be applied as an optional 
postprocessing step to shorten the transformed code and improve its 
readability.

:ref:`Reverse-mode transformations<reverse_mode>` store overwritten values 
using a :ref:`"tape"<value_tape>` that is 
implemented as a static array, rather than a LIFO stack as in many 
implementations, so that the transformed routines may (someday) be parallelized 
and/or offloaded to GPU.

Also in reverse-mode, three types of 
:ref:`reversal schedules<reversal_schedules>` are available:

- :ref:`split reversal schedules<split_reversal_schedule>`,
- :ref:`joint reversal schedules<joint_reversal_schedule>`,
- :ref:`"link" reversal schedules<link_reversal_schedule>` specifying strong or weak links for all calling-called pairs of routines.


.. _missing_features:

Missing features
++++++++++++++++

What has **not** been implemented includes:

- functions and programs,
- differentiating called routines that are not found in the same file (or ``Container`` node) as the target routine,
- nary operations,
- loops,
- control flow,
- array variables and arguments,
- activity analysis (dependence DAG),
- to-be-recorded (TBR) analysis,
- taping operations results to reduce the computational complexity of the adjoint,
- and much more.

