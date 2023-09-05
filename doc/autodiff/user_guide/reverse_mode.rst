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

.. _reverse_mode:


Reverse-mode AD (adjoint)
=========================

The adjoint of the Fortran source code is constructed using a source-to-source 
and line-by-line approach, transforming the :ref:`target (to be transformed) 
routine <target_routine>` into three different routines, one of which computes 
the adjoints of the variables by reversing the order of computation of the 
target routine.  

This is implemented in PSyclone by parsing the source code file containing the 
target routine, and eventually the routines it calls, transforming it into a 
PSyIR AST and applying :ref:`reverse-mode automatic differentiation 
transformations <reverse_transformations>` to the nodes thus obtained. 
The resulting PSyIR tree is then written to Fortran source code.

.. _reverse_transformations:

Reverse-mode transformations
++++++++++++++++++++++++++++

Several transformations, to be applied on PSyIR nodes, have been implemented. 
In reverse-mode, all of them follow the naming convention 
``ADReverse[PSyIRNodeSubclass]Trans``.
The one of most interest for the user is the 
:ref:`ADReverseContainerTrans <reverse_container_trans>` class and its ``apply`` method.

After parsing the Fortran code file containing the target routine, an
``ADReverseContainerTrans`` instance should be applied to it to perform 
automatic differentiation.  
The ``ADReverseContainerTrans.apply`` method in turn applies an 
``ADReverseRoutineTrans`` to the target routine, which goes line-by-line through
the statements found in the ``Routine`` node, applying 
``ADReverse[PSyIRNodeSubclass]Trans`` to the statements, etc.

.. tikz:: Reverse-mode AD transformation call graph
    :libs: graphs, graphs.standard, quotes

      \graph[nodes={draw}, grow down = 1.5cm, branch right = 5cm]{
                        ADReverseContainerTrans ->["(in)dependent vars"] ADReverseRoutineTrans -> {ADReverseAssignmentTrans ->[swap, "LHS adjoint"] ADReverseOperationTrans ->[bend left = 2cm, "parent adjoint"] a / $ $[white] ->[bend left = 2cm, "if composed"] ADReverseOperationTrans, ADReverseCallTrans},
                        ADReverseCallTrans ->["if operation argument"] ADReverseOperationTrans, 
                        ADReverseCallTrans ->[bend right = 0.5cm, swap, "transform called routine", purple] ADReverseRoutineTrans,
                        };    

.. _reverse_container_trans:

Container transformation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: psyclone.autodiff.transformations.ADReverseContainerTrans
      :members: apply


As can be seen, the required arguments include the PSyIR ``[File]Container`` 
node obtained by parsing and transforming the source code, the **names** of the 
:ref:`target routine<target_routine>`, 
:ref:`dependent variables<dependent_variables>` (to be differentiated) 
and :ref:`independent variables<independent_variables>` (to differentiate with 
respect to), as well as the :ref:`reversal schedule<reversal_schedules>` and 
eventual transformation options.

The transformation returns a PSyIR ``Container`` node containing four 
``Routine`` definitions for:

- the *advancing* (original) motion,
- the *recording* motion, which records overwritten values to the tape,
- the *returning* motion, which recovers values from the tape and computes the adjoints of the independent variables,
- the *reversing* motion, which combines the two precedent recording and returning motions and is the one to call in order to differentiate.

If some other routine is called by the target one, the returned ``Container`` 
node also contains four definitions for its different motions.


.. _target_routine:

Target routine
--------------

The target routine is the Fortran routine in which to differentiate the 
:ref:`dependent variables<dependent_variables>` with respect to the 
:ref:`independent variables<independent_variables>`.  
The routines it may call will also be differentiated iff they can be found in 
the ``[File]Container`` being transformed.

.. _dependent_variables:

Dependent variables
-------------------

The dependent variables are those to differentiate. 
Their intent in the target routine must be compatible with their values being 
returned, *ie.* they cannot be intent(in) arguments of the target routine.

.. _independent_variables:

Independent variables
---------------------

The independent variables are those to differentiate with respect to. 
Their intent in the target routine must be compatible with their values being 
provided as arguments, *ie.* they cannot be intent(out) arguments of the target 
routine.

.. _reversal_schedules:

Reversal schedules 
------------------

Reversal schedules (see :footcite:t:`griewank-walther` chapter 12.2, p.265) 
specify the way a transformed routine may call other 
transformed routines.  
They are implemented as 3 subclasses of ``ADReversalSchedule``.

As an example let us consider the target routine ``foo`` calling subroutines 
``bar`` and ``qux``.

.. code-block:: fortran

    subroutine foo(x, y)
        call bar(x, y)
        call qux(x, y)
    end subroutine foo

And let us denote:

- advancing :math:`\square` routine the original,
- recording :math:`\Large\triangleright` routine the one recording values to the tape,
- returning :math:`\Large\triangleleft` routine the one computing the adjoints,
- reversing :math:`\Large\triangleright\triangleleft` routine the one combining the two preceding. Call it to differentiate.

.. _split_reversal_schedule:

Split reversal schedule
***********************

In split reversal, children (called) routines follow the recording or returning
motion of their parent (calling) routine.

By doing so, the computational complexity is kept as low as possible but values 
stored to the tape in the recording motion of the called routines need to be 
kept until they are called in returning motion, thus using a possibly 
large amount of memory.

.. tikz:: Split reversal schedule
    :libs: graphs, graphs.standard, quotes

    \graph[trie, simple, nodes={draw}, grow down = 0.8cm, branch right = 2cm]{
    foorev / {foo {\Large$\triangleright\triangleleft$}} -> 
    {foorec / {foo {\Large$\triangleright$}} -> {barrec / {bar {\Large$\triangleright$}}[orange], quxrec / {qux {\Large$\triangleright$}}[purple]}, 
    fooret / {foo {\Large$\triangleleft$}} -> {barret / {bar {\Large$\triangleleft$}}, quxret / {qux {\Large$\triangleleft$}}}
    },
    };    

.. _joint_reversal_schedule:

Joint reversal schedule
***********************

In joint reversal, all children (called) routines advance without recording when
their parent (calling) routine is recording and reverse (record then immendiatly
return) when their parent routine is returning.

On the one hand, this reversal schedule uses a smaller tape overall, as the 
values used in adjoining the called routines do no need to be stored longer than
for them to be reversed. On the other hand, called subroutines computations are
repeated, with increases the computational complexity of the adjoint program.

.. tikz:: Joint reversal schedule
    :libs: graphs, graphs.standard, quotes

    \graph[trie, simple, nodes={draw}, grow down = 0.8cm, branch right = 2cm]{
    foorev / {foo {\Large$\triangleright\triangleleft$}} -> 
    {foorec / {foo {\Large$\triangleright$}} -> {baradv / {bar $\square$}[orange], quxadv / {qux $\square$}[purple]}, 
    fooret / {foo {\Large$\triangleleft$}} -> 
        {barrev / {bar {\Large$\triangleright\triangleleft$}} ->
            {barrec / {bar {\Large$\triangleright$}}[orange], barret / {bar {\Large$\triangleleft$}}}, 
        quxrev / {qux {\Large$\triangleright\triangleleft$}} ->
            {quxrec / {qux {\Large$\triangleright$}}[purple], quxret / {qux {\Large$\triangleleft$}}}}
    },
    };    


.. _link_reversal_schedule:

"Link" reversal schedule
************************

A third possibility is to specify *strong* or *weak* links for each 
caller-called pair of routines, where strong links behave as in split reversal
and weak links as in joint reversal.

Below is an illustration of our toy example with ``foo-bar`` a strong link 
and ``foo-qux`` a weak link.

.. tikz:: Link reversal schedule with foo-bar a strong link and foo-qux a weak link
    :libs: graphs, graphs.standard, quotes

    \graph[trie, simple, nodes={draw}, grow down = 0.8cm, branch right = 2cm]{
    foorev / {foo {\Large$\triangleright\triangleleft$}} -> 
    {foorec / {foo {\Large$\triangleright$}} -> {barrec / {bar {\Large$\triangleright$}}[orange], quxadv / {qux $\square$}[purple]}, 
    fooret / {foo {\Large$\triangleleft$}} -> 
        {barret / {bar {\Large$\triangleleft$}},
        quxrev / {qux {\Large$\triangleright\triangleleft$}} ->
            {quxrec / {qux {\Large$\triangleright$}}[purple], quxret / {qux {\Large$\triangleleft$}}}}
    },
    };  



.. _value_tape:

Value tape
++++++++++

Prevalues of overwritten variables are recorded and restored from a *value tape*,
implemented as a static array.
The transformations themselves employ an ``ADValueTape`` to generate recording 
and restoring statements to and from the value tape array.

.. autoclass:: psyclone.autodiff.tapes.ADValueTape
      :members: record, restore

.. _adjoints:

Generating adjoints
+++++++++++++++++++

The transformations applied to generate adjoints are detailled below.
They mostly follow the guidelines found in :footcite:t:`griewank-walther` 
chapter 6.2, pp.125-126.

Internally, the transformations used are ``ADReverseAssignmentTrans``,
``ADReverseOperationTrans`` and ``ADReverseCallTrans``, depending on the PSyIR
node being transformed. 
These all return two separate lists of PSyIR statements, used respectively in 
extending the recording and returning routines being generated.

.. _operation_adjoints:

Adjoints of operations
~~~~~~~~~~~~~~~~~~~~~~

.. _unary_operation_adjoints:

Adjoints of unary operations
----------------------------

+-------------------+-----------------------+---------------------------------------------------+
| Advancing motion  | Recording motion      | Returning motion                                  |
+===================+=======================+===================================================+
|``f = +x``         | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = +x``            |   ``x_adj = x_adj + f_adj``                       |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = -x``         | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = -x``            |   ``x_adj = x_adj - f_adj``                       |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = SQRT(x)``    | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = SQRT(x)``       |   ``x_adj = x_adj + f_adj / (2 * SQRT(x))``       |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = EXP(x)``     | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = EXP(x)``        |   ``x_adj = x_adj + f_adj * EXP(x)``              |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = LOG(x)``     | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = LOG(x)``        |   ``x_adj = x_adj + f_adj / x``                   |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = LOG10(x)``   | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = LOG10(x)``      |   ``x_adj = x_adj + f_adj / (x * LOG(10.0))``     |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = COS(x)``     | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = COS(x)``        |   ``x_adj = x_adj - f_adj * SIN(x)``              |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = SIN(x)``     | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = SIN(x)``        |   ``x_adj = x_adj + f_adj * COS(x)``              |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = TAN(x)``     | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = TAN(x)``        |   ``x_adj = x_adj + f_adj * (1.0 + TAN(x) ** 2)`` |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = ACOS(x)``    | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = ACOS(x)``       |   ``x_adj = x_adj - f_adj / SQRT(1.0 - x ** 2)``  |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = ASIN(x)``    | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = ASIN(x)``       |   ``x_adj = x_adj + f_adj / SQRT(1.0 - x ** 2)``  |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = ATAN(x)``    | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = ATAN(x)``       |   ``x_adj = x_adj + f_adj / (1.0 + x ** 2)``      |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+
|``f = ABS(x)``     | ``value_tape(i) = f`` |   ``f = value_tape(i)``                           |
|                   |                       |                                                   |
|                   | ``f = ABS(x)``        |   ``x_adj = x_adj + f_adj * (x / ABS(x))``        |
|                   |                       |                                                   |
|                   |                       |   ``f_adj = 0.0``                                 |
+-------------------+-----------------------+---------------------------------------------------+

*Note*: some of these adjoints computations, 
explicitly those for ``SQRT``, ``EXP``, ``TAN`` and ``ABS``,
could reuse the (post)value of ``f`` before restoring its prevalue from the 
value tape rather than recompute it (see :footcite:t:`griewank-walther` 
table 4.8, p.68). This is not implemented yet.

.. _binary_operation_adjoints:

Adjoints of binary operations
-----------------------------

+-------------------+-----------------------+---------------------------------------------+
| Advancing motion  | Recording motion      | Returning motion                            |
+===================+=======================+=============================================+
|``f = x + y``      | ``value_tape(i) = f`` |``f = value_tape(i)``                        |
|                   |                       |                                             |
|                   | ``f = x + y``         |``x_adj = x_adj + f_adj``                    |
|                   |                       |                                             |
|                   |                       |``y_adj = y_adj + f_adj``                    |
|                   |                       |                                             |
|                   |                       |``f_adj = 0.0``                              |
+-------------------+-----------------------+---------------------------------------------+
|``f = x - y``      | ``value_tape(i) = f`` |``f = value_tape(i)``                        |
|                   |                       |                                             |
|                   | ``f = x - y``         |``x_adj = x_adj + f_adj``                    |
|                   |                       |                                             |
|                   |                       |``y_adj = y_adj - f_adj``                    |
|                   |                       |                                             |
|                   |                       |``f_adj = 0.0``                              |
+-------------------+-----------------------+---------------------------------------------+
|``f = x * y``      | ``value_tape(i) = f`` |``f = value_tape(i)``                        |
|                   |                       |                                             |
|                   | ``f = x * y``         |``x_adj = x_adj + f_adj * y``                |
|                   |                       |                                             |
|                   |                       |``y_adj = y_adj + f_adj * x``                |
|                   |                       |                                             |
|                   |                       |``f_adj = 0.0``                              |
+-------------------+-----------------------+---------------------------------------------+
|``f = x / y``      | ``value_tape(i) = f`` |``f = value_tape(i)``                        |
|                   |                       |                                             |
|                   | ``f = x / y``         |``x_adj = x_adj + f_adj / y``                |
|                   |                       |                                             |
|                   |                       |``y_adj = y_adj - f_adj * x / y ** 2``       |
|                   |                       |                                             |
|                   |                       |``f_adj = 0.0``                              |
+-------------------+-----------------------+---------------------------------------------+
|``f = x ** y``     | ``value_tape(i) = f`` |``f = value_tape(i)``                        |
|                   |                       |                                             |
|                   | ``f = x ** y``        |``x_adj = x_adj + f_adj * y * x ** (y - 1)`` |
|                   |                       |                                             |
|                   |                       |``y_adj = y_adj + f_adj * x ** y * LOG(x)``  |
|                   |                       |                                             |
|                   |                       |``f_adj = 0.0``                              |
+-------------------+-----------------------+---------------------------------------------+

*Note*: some of these adjoints computations, 
explicitly those for ``/`` and ``**``
could reuse the (post)value of ``f`` before restoring its prevalue from the 
value tape rather than recompute it (see :footcite:t:`griewank-walther` 
table 4.8, p.68). This is not implemented yet.

.. _composed_operations_adjoints:

The cases detailled above are the simpler ones, of assigning the result of an 
operation to a variable.

When composed operations are present, an adjoint variable is declared for the 
adjoint of the operation itself and used to increment the adjoints of its 
operands.

The transformation option ``inline_operation_adjoints`` allows the user to 
choose whether these operation adjoints should be substituted in further 
computations of adjoints as a postprocessing step, 
iff they only appear once on the RHS of an assignment.

As an example, consider the following computation involving composed operations
and the associated adjoints computations, without and with substitution.
*Note*: taping assignments are omitted below.

+---------------------+-------------------------------------+---------------------------------------------+
| Composed operation  | Adjoints, without substitution      | Adjoints, with substitution                 |
+=====================+=====================================+=============================================+
|``f = EXP(x) + z``   | ``op_adj = f_adj``                  |``z_adj = z_adj + f_adj``                    |
|                     |                                     |                                             |
|                     | ``z_adj = z_adj + f_adj``           |``x_adj = x_adj + f_adj * EXP(x)``           |
|                     |                                     |                                             |
|                     | ``x_adj = x_adj + op_adj * EXP(x)`` |``f_adj = 0.0``                              |
|                     |                                     |                                             |
|                     | ``f_adj = 0.0``                     |                                             |
+---------------------+-------------------------------------+---------------------------------------------+

.. _iterative_assignments:

Adjoints of iterative assignments 
---------------------------------

In the case of iterative assignments *ie.* where the LHS variable of the 
assignment is also present on the RHS, additional care must be taken to avoid 
incorrect computations of the LHS adjoint by assigning to it last rather than 
incrementing its value as in the general case detailled above 
(see :footcite:t:`griewank-walther` chapter 5.1, p.93).

As an example consider the following adjoint:

+-------------------+-----------------------+--------------------------+
| Advancing motion  | Recording motion      | Returning motion         |
+===================+=======================+==========================+
|``f = 2 * f + x``  | ``value_tape(i) = f`` |``f = value_tape(i)``     |
|                   |                       |                          |
|                   | ``f = 2 * f + x``     |``x_adj = x_adj + f_adj`` |
|                   |                       |                          |
|                   |                       |``f_adj = f_adj * 2``     |
+-------------------+-----------------------+--------------------------+

.. _call_adjoints:

Adjoints of calls to subroutines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The adjoints of calls to subroutines depend on the 
:ref:`reversal schedule <reversal_schedules>` that is used.

Whether the prevalues of the arguments are recorded and restored from the tape 
depend on their intent in the called subroutine, which determines whether their 
value might be overwritten by it or not.

Operations as subroutine call arguments are also transformed.  

Split reversal schedule
-----------------------

+-------------------+----------------------------+--------------------------------------------+
| Advancing motion  | Recording motion           | Returning motion                           |
+===================+============================+============================================+
|``call func(x, y)``|[``value_tape(i) = x``]     |[``x = value_tape(i)``]                     |
+-------------------+----------------------------+--------------------------------------------+
|                   |[``value_tape(i + 1) = y``] |[``y = value_tape(i + 1)``]                 |
+-------------------+----------------------------+--------------------------------------------+
|                   |``call func_recording(x,y)``|``call func_returning(x, x_adj, y, y_adj)`` |
+-------------------+----------------------------+--------------------------------------------+

Joint reversal schedule
-----------------------

+-------------------+----------------------------+--------------------------------------------+
| Advancing motion  | Recording motion           | Returning motion                           |
+===================+============================+============================================+
|``call func(x, y)``|[``value_tape(i) = x``]     |[``x = value_tape(i)``]                     |
+-------------------+----------------------------+--------------------------------------------+
|                   |[``value_tape(i + 1) = y``] |[``y = value_tape(i + 1)``]                 |
+-------------------+----------------------------+--------------------------------------------+
|                   |``call func(x,y)``          |``call func_reversing(x, x_adj, y, y_adj)`` |
+-------------------+----------------------------+--------------------------------------------+

.. footbibliography::
