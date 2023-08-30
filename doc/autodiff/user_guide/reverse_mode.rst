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

.. _reverse_mode:


Reverse-mode automatic differentiation (adjoint)
================================================

The adjoint of the Fortran source code is constructed using a source-to-source 
and line-by-line approach, transforming the :ref:`target (to be transformed) 
routine <target_routine>` into three different routines, one of which computes 
the adjoints of the variables by reversing the order of computation of the 
target routine.  

This is implemented in PSyclone by parsing the source code file containing the 
target routine, and eventually the routines it calls, transforming it into a 
PSyIR AST and applying :ref:`reverse-mode automatic differentiation 
transformations <transformations>` 
to the nodes thus obtained. The resulting PSyIR tree is then written to 
Fortran source code.

.. _transformations:

Reverse-mode automatic differentiation transformations
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Several transformations, to be applied on PSyIR nodes, have been implemented. 
In reverse-mode, all of them follow the naming convention 
``ADReverse[PSyIRNodeSubclass]Trans``.
The one of most interest for the user is the 
:ref:`ADReverseContainerTrans <container_trans>` class and its ``apply`` method.

.. _container_trans:

Container transformation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: psyclone.autodiff.transformations.ADReverseContainerTrans
      :members: apply

After parsing the Fortran code file containing the target routine, an
``ADReverseContainerTrans`` instance should be applied to it to perform 
automatic differentiation.  
The ``ADReverseContainerTrans.apply`` method in turn applies an 
``ADReverseRoutineTrans`` to the target routine, which goes line-by-line through
the statements found in the ``Routine`` node, applying 
``ADReverse[PSyIRNodeSubclass]Trans`` to the statements, etc.

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
- the *returning* motion, which recovers values from the tape and computes the 
adjoints of the independent variables,
- the *reversing* motion, which combines the two precedent recording and 
returning motions and is the one to call in order to differentiate.

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

Reversal schedules specify the way a transformed routine may call other 
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
----------

.. autoclass:: psyclone.autodiff.tapes.ADValueTape
      :members: record, restore




