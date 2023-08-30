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

.. _automatic_differentiation:

Automatic differentiation
=========================

Simply put, the aim of automatic differentiation is to automatically obtain the
derivatives of somes variables output by an existing program with respect to 
some of its input variables.

It avoids resorting to symbolic differentiation, which is error-prone when done
manually and quickly of excessive complexity when applied automatically, or 
finite differences, which are inexact.

To gain an intuition of the way this is achieved, consider a program computing 
return values of variables :math:`y_j` from values of arguments :math:`x_i` 
through intermediate values :math:`v_k`, where each value is obtained from its 
direct predecessors through *elemental* operations 
:math:`(+, \times, /, \exp, \sf{etc.})`.

.. _notations:

Let us denote:
 - independent variables: :math:`(x_i)_i, i \in \{1, 2, ..., n\}`,
 - dependent variables: :math:`(y_j)_j, j \in \{1, 2, ..., m\}`,
 - intermediate values: :math:`(v_k)_k` which may or not be assigned to variables in the program,
 - relation: :math:`i \prec j` if :math:`v_j` depends on :math:`v_i`, *eg.* below :math:`1 \prec 5`.
 - predecessors: :math:`u_j := (v_i)_{i \prec j}` *eg.* below :math:`u_5 = \begin{pmatrix} v_1 \\ v_2 \end{pmatrix}`
 - operation: :math:`\varphi_j: u_j \mapsto v_j` *eg.* below :math:`v_5 = \varphi_5(u_5)`

.. tikz:: Example program
    :libs: graphs, graphs.standard, quotes

    \graph[math nodes={circle, draw}, branch down = 0.5cm, grow right = 1.5cm]{  
    x_1 = v_{-2} -> {v_1 / {v_1}[olive] ->[olive, swap, "$\varphi_5$"] v_5 / {v_5}[orange] -> {v_8 = y_1, v_9 = y_2}, v_2 / {v_2}[olive] -> {v_6 -> v_9 = y_2}},
    v_2 ->[olive] v_5,
    x_1 = v_{-2} ->[bend left = 0.5cm] v_8 = y_1,
    x_2 = v_{-1} -> {v_2, v_3 -> v_7 -> v_{10} = y_3},
    x_3 = v_0 -> {v_3, v_4 -> v_{10} = y_3, v_7},
    %x_3 = v_0 ->[bend right = 0.5cm] v_{10} = y_3
    };

Since all programs can be reduced to sequential elemental operations in this 
fashion, automatic differentiation allows to compute 
:math:`\dfrac{\partial y_j}{\partial x_i}(x_1, \ldots, x_n)`
by differentiating operations :math:`\varphi_k : u_k \mapsto v_k` and using 
the chain rule.

It comes in two main flavors, usually called 
:ref:`forward- or tangent-mode <forward_math>` and 
:ref:`reverse- or adjoint-mode <reverse_math>`.

.. _forward_math:

Forward- or tangent-mode
------------------------

Using the notations introduced :ref:`above <notations>`, forward-mode automatic
differentiations allows to compute all derivatives w.r.t. a **single** 
independent variable :math:`d \in (x_i)_i`.

Let us denote the derivatives w.r.t. :math:`d` as

.. math::

    \dot{v}_i = \dfrac{\partial v_i}{\partial d}

such that the chain rule writes

.. math::

    \dot{v}_j = \sum_{i \prec j} \dfrac{\partial \varphi_j}{\partial v_i}(u_j) \dot{v}_i

Forward-mode automatic differentation is equivalent to applying substitutions in
the order indicated by the arrow in 

.. math::
    \dot{v}_{k+2} = \overleftarrow{\dfrac{\partial v_{k+2}}{\partial v_{k+1}} \underbrace{\left( \frac{\partial v_{k+1}}{\partial v_{k}} {\dot{v}_{k}} \right)}_{\dot{v}_{k+1}}}

Initializing :math:`\dot{d} = 1` and :math:`\dot{v}_k = 0, \forall v_k \neq d`,

we obtain in a **single** evaluation :math:`\left( \frac{\partial y_j}{\partial d}(x_1, \ldots x_n)\right)_j = J((x_i)_i) (0 \ldots \dot{d} \ldots 0)^T`

where :math:`J` is the Jacobian matrix :math:`J = \nabla \begin{pmatrix} y_1(x_1, \ldots, x_n) \\ \vdots \\ y_m(x_1, \ldots, x_n) \end{pmatrix}`.

Advantages and inconvenients
****************************

Forward-mode is easy to implement as derivatives can be computed in the same 
order of computation as that of the original program.  

If there are less independent than dependent variables, its complexity
is lower than that of the reverse- or adjoint-mode. But frequently, and maybe 
even more so in oceanographic models, the number of inputs greatly exceeds the 
number of outputs, requiring many repeated evaluations, one for each input or 
independent variable to differentiate with respect to.


.. _reverse_math:

Reverse- or adjoint-mode
------------------------

Using the notations introduced :ref:`above <notations>`, reverse-mode automatic
differentiations allows to compute all derivatives of a **single** 
dependent variable :math:`z \in (y_j)_j`.

Let us denote the adjoints w.r.t. :math:`z` as

.. math::

   \bar{v}_i = \dfrac{\partial z}{\partial v_i}

such that the chain rule writes

.. math::

    \bar{v}_i = \sum_{\mathbf{{j \succ i}}} \mathbf{\bar{v}_j} \dfrac{\partial \varphi_j}{\partial v_i}(\mathbf{\overset{?}{u_j}})

where bold font is used to highlight how the value of the adjoint :math:`\bar{v}_i` 
depends on **successors** of :math:`v_i`.

Reverse-mode automatic differentation is equivalent to applying substitutions in
the order indicated by the arrow in 

.. math::

    \overrightarrow{\underbrace{\left( \bar{v}_{k} \dfrac{\partial v_{k}}{\partial v_{k-1}} \right)}_{\bar{v}_{k-1}} \dfrac{\partial v_{k-1}}{\partial v_{k-2}} } = \bar{v}_{k-2}

Initializing :math:`\bar{z} = 1` and :math:`\bar{v}_k = 0, \forall v_k \neq z`,

we obtain in a **single** evaluation :math:`\left( \frac{\partial z}{\partial x_i}(x_1, \ldots, x_n)\right)_i = \nabla^T z(x_1, \ldots, x_n) = (0 \ldots \bar{z} \ldots 0) J(x_1, \ldots, x_n)`.

Advantages and inconvenients
****************************

Reverse-mode is quite a lot more complicated to implement than forward-mode as 
adjoints need to be computed in the reversed order of computation compared to 
that of the original program as illustated in the 
:ref:`example below <reverse_example>`.  

If there are less dependent than independent variables, as is often the case, 
its complexity is lower than that of the forward- or tagent-mode. 

However, when non-linearities are present, reverse-mode also requires running 
the original program and recording overwritten values, and eventually some 
the results of some operations, when they appear in the computations of some 
adjoints. 
This add further complications compared to forward-mode and requires using 
a persistent "tape", which needs to be kept in memory, or recomputing values 
as many times as they are required.

.. _reverse_example:

A simple example in reverse-mode with non-linearities
*****************************************************

Let us consider the simple computations displayed below and illustate how to 
compute the adjoints 
:math:`\bar{x}_1 = \dfrac{\partial z}{\partial x_1}` 
and :math:`\bar{x}_2 = \dfrac{\partial z}{\partial x_2}`
for a chosen dependent variable :math:`z \in \{y_1, y_2\}`.

.. tikz:: Simple program example
    :libs: graphs, graphs.standard, quotes

    \graph[nodes={draw}, branch down = 1cm, grow right = 4cm]{
    x1 / $x_1$[red] ->[red] {v1 / $v_1 = x_1^{{2}}$ -> v3 / {$v_3 = {\exp}(x_1^2)$} -> y1 / {$y_1 = \exp(x_1^2) - 3 * x_1 + x_2$}, 
                    v2 / $v_2 = 3 * x_1$ -> v4 / $v_4 = 3 * x_1 + x_2$[purple] ->[purple] {y1, y2 / $y_2 = x_2 {*} (3 * x_1 + x_2)$}},
    3[black] -> v2,
    x2 / $x_2$[olive] ->[olive] v4,
    x2 ->[olive, bend right = 0.2cm] y2
    
    };   

.. tikz:: Reverse-mode example
    :libs: graphs, graphs.standard, quotes

    \graph[nodes={draw}, branch down = 1cm, grow right = 4cm]{
    x1b1 / {$\bar{x}_1 +$ $= \bar{v}_1 * {2 * x_1}$}[red] <-[red] v1b / {$\bar{v}_1 +$ $= \bar{v}_3 * {v_3}$} <- v3b / {$\bar{v}_3 +$ $= \bar{y}_1 * 1$} <- y1b / {$\bar{y}_1$},
    x1b2 / {$\bar{x}_1 +$ $= \bar{v}_2 * 3$}[red] <-[red] v2b / {$\bar{v}_2 +$ $= \bar{v}_4 * 1$} <-[white] {v4b1 / {$\bar{v}_4 +$ $= \bar{y}_1 * (-1)$}[purple] <-[white] {y1b, y2b / {$\bar{y}_2$}}, v4b2 / {$\bar{v}_4 +$ $= \bar{y}_2 * {x_2}$}[purple]},
    v4b1 <-[purple] y1b,
    v2b <- v4b1,
    v2b <- v4b2,
    v4b2 <-[purple] y2b,
    x2b1 / {$\bar{x}_2 +$ $= \bar{v}_4 * 1$}[olive] <-[olive] v4b1,
    x2b2 / {$\bar{x}_2 +$ $= \bar{y}_2 * {v_4}$}[olive] <-[olive, bend right = 0.5cm] y2b
    };    

Initialize with :math:`\forall i, \bar{x}_i = 0, \forall k, \bar{v}_k = 0 \text{ and choose } (\bar{y}_1 = 1, \bar{y}_2 = 0) \text{ \textbf{or} } (\bar{y}_1 = 0, \bar{y}_2 = 1)`
to obtain the adjoints.