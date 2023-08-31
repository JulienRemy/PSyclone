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
.. Written by J. Remy, Université Grenoble Alpes, Inria

.. _forward_mode:


Forward-mode AD (tangent)
=========================

The derivative of the Fortran source code is constructed using a source-to-source 
and line-by-line approach, transforming the target routine into a tangent routine, 
which computes the derivatives of the dependent variables with respect to the 
independent variables.

This is implemented in PSyclone by parsing the source code file containing the 
target routine, and eventually the routines it calls, transforming it into a 
PSyIR AST and applying :ref:`forward-mode automatic differentiation 
transformations <forward_transformations>` to the nodes thus obtained. 
The resulting PSyIR tree is then written to Fortran source code.

.. _forward_transformations:

Forward-mode transformations
++++++++++++++++++++++++++++

All forward-mode AD transformations, to be applied to PSyIR nodes,
follow the naming convention ``ADForward[PSyIRNodeSubclass]Trans``.
The one users should use is 
:ref:`ADForwardContainerTrans <forward_container_trans>` class and its 
``apply`` method.

.. _forward_container_trans:

Container transformation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: psyclone.autodiff.transformations.ADForwardContainerTrans
      :members: apply

After parsing the Fortran code file containing the target routine, an
``ADForwardContainerTrans`` instance should be applied to it to perform 
automatic differentiation.  
This in turn applies an 
``ADForwardRoutineTrans`` to the target routine, which goes line-by-line through
the statements found in the ``Routine`` node, applying 
``ADForward[PSyIRNodeSubclass]Trans`` to the statements, etc.

For descriptions of the arguments, see the relevant sections of 
:ref:`ADReverseContainerTrans <reverse_container_trans>`: 
:ref:`target routine<target_routine>`, 
:ref:`dependent variables<dependent_variables>` (to be differentiated) 
and :ref:`independent variables<independent_variables>` (to differentiate with 
respect to).

The transformation returns a PSyIR ``Container`` node containing two 
``Routine`` definitions for:

- the original target routine,
- the transformed tangent routine, which computes of the required derivatives.

If some other routine is called by the target one, the returned ``Container`` 
node also contains the original and tangent definitions for it.


Generating derivatives
++++++++++++++++++++++

The transformations applied to generate derivatives are detailled below.
They mostly follow the guidelines found in :footcite:t:`griewank-walther` 
chapter 6.2, p.123.

.. _operation_derivatives:

Derivatives of operations
-------------------------

.. _unary_operation_derivatives:

Derivatives of unary operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------+------------------------------------+
| Original       | Transformed                        |
+================+====================================+
|``f = +x``      | ``f_d = x_d``                      |
|                |                                    |
|                | ``f = +x``                         |
+----------------+------------------------------------+
|``f = -x``      | ``f_d = -x_d``                     |
|                |                                    |
|                | ``f = -x``                         |
+----------------+------------------------------------+
|``f = SQRT(x)`` | ``f_d = x_d / (2 * SQRT(x))``      |
|                |                                    |
|                | ``f = SQRT(x)``                    |
+----------------+------------------------------------+
|``f = EXP(x)``  | ``f_d = EXP(x) * x_d``             |
|                |                                    |
|                |``f = EXP(x)``                      |
+----------------+------------------------------------+
|``f = LOG(x)``  | ``f_d = x_d / x``                  |
|                |                                    |
|                | ``f = LOG(x)``                     |
+----------------+------------------------------------+
|``f = LOG10(x)``| ``f_d = x_d / (x * LOG(10.0))``    |
|                |                                    |
|                | ``f = LOG10(x)``                   |
+----------------+------------------------------------+
|``f = COS(x)``  | ``f_d = (-SIN(x)) * x_d``          |
|                |                                    |
|                | ``f = COS(x)``                     |
+----------------+------------------------------------+
|``f = SIN(x)``  | ``f_d = COS(x) * x_d``             |
|                |                                    |
|                | ``f = SIN(x)``                     |
+----------------+------------------------------------+
|``f = TAN(x)``  | ``f_d = (1.0 + TAN(x) ** 2) * x_d``|
|                |                                    |
|                | ``f = TAN(x)``                     |
+----------------+------------------------------------+
|``f = ACOS(x)`` | ``f_d = -x_d / SQRT(1.0 - x ** 2)``|
|                |                                    |
|                | ``f = ACOS(x)``                    |
+----------------+------------------------------------+
|``f = ASIN(x)`` | ``f_d = x_d / SQRT(1.0 - x ** 2)`` |
|                |                                    |
|                | ``f = ASIN(x)``                    |
+----------------+------------------------------------+
|``f = ATAN(x)`` | ``f_d = x_d / (1.0 + x ** 2)``     |
|                |                                    |
|                | ``f = ATAN(x)``                    |
+----------------+------------------------------------+
|``f = ABS(x)``  | ``f_d = x / ABS(x) * x_d``         |
|                |                                    |
|                | ``f = ABS(x)``                     |
+----------------+------------------------------------+

.. _binary_operation_derivatives:

Derivatives of binary operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------+---------------------------------------------------------------+
| Advancing motion  | Recording motion                                              |
+===================+===============================================================+
|``f = x + y``      | ``f_d = x_d + y_d``                                           |
|                   |                                                               |
|                   | ``f = x + y``                                                 |
+-------------------+---------------------------------------------------------------+
|``f = x - y``      | ``f_d = x_d - y_d``                                           |
|                   |                                                               |
|                   | ``f = x - y``                                                 |
+-------------------+---------------------------------------------------------------+
|``f = x * y``      | ``f_d = x_d * y + y_d * x``                                   |
|                   |                                                               |
|                   | ``f = x * y``                                                 |
+-------------------+---------------------------------------------------------------+
|``f = x / y``      | ``f_d = (x_d - y_d * x / y) / y``                             |
|                   |                                                               |
|                   | ``f = x / y``                                                 |
+-------------------+---------------------------------------------------------------+
|``f = x ** y``     | ``f_d = x_d * (y * x ** (y - 1)) + y_d * (x ** y * LOG(x))``  |
|                   |                                                               |
|                   | ``f = x ** y``                                                |
+-------------------+---------------------------------------------------------------+

.. _call_derivatives:

Derivatives of calls to subroutines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------+-----------------------------------------+
| Original          | Transformed                             |
+===================+=========================================+
|``call func(x, y)``|``call func_tangent(x, x_d, y, y_d)``    |
+-------------------+-----------------------------------------+

.. footbibliography::