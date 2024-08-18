# PSyclone: automatic differentiation

# !!! THIS IS AN EXPERIMENTAL FORK !!! #
### It is intended only as a prototype ###
### Currently in development and not thoroughly tested ###

# Introduction #

Welcome to PSyclone `autodiff` module fork, a prototype implementation of forward- and reverse-mode automatic differentation in PSyclone.  

Please see [psyclone-autodiff on ReadTheDocs](https://psyclone-autodiff.readthedocs.io/) for more information.

For PSyclone itself, see [GitHub](https://github.com/stfc/PSyclone) and [ReadTheDocs](http://psyclone.readthedocs.io).  

# Description #

This implements forward- and reverse-mode automatic differentiation using source-to-source transformations.  
Compared to other tools, it uses a static or dynamic array as a tape to record and restore values rather than a LIFO stack.

What has been implemented (but **not** necessarily tested):  
- transforming subroutines containing:
    - assignments,
    - calls to subroutines,
    - (some) unary and binary operations,
- scalar and array variables and arguments,
- nested calls to subroutines,
- simplification of the transformed subroutines as a postprocessing step,
- in reverse-mode:
    - three different types of reversal schedules,
    - recording and restoring values from a tape.
- conditional branches, including taping the condition and restoring it in reverse-mode.
- loops, with explicit tape indexing in reverse-mode.

What has *not* been implemented:
- functions and programs,
- nary operations,
- activity analysis (dependence DAG),
- to-be-recorded (TBR) analysis,
- and much more.

A talk slide deck can be found in the `src/psyclone/autodiff/documents` subdirectory.

# Installing #

Instructions for installing from source can be found on ReadTheDocs in the [getting started section](https://psyclone-autodiff.readthedocs.io/en/latest/getting_started.html).

# Documentation #

The documentation is not yet hosted online.  
It can be generated using the instructions found in the `doc/reference_guide/` subdirectory.  
The Doxygen documentation built in `doc/reference_guide/build/html/_static/html/index.html` is suggested over the Sphinx one.

# Tutorial #

A tutorial may be found in the `src/psyclone/autodiff/tutorial` subdirectory.  

# Tests #
 
Tests include unit tests for *some* of the transformations and numerical comparisons to [Tapenade](https://team.inria.fr/ecuador/en/tapenade/) 3.16.

