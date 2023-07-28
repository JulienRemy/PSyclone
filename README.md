# !!! THIS IS AN EXPERIMENTAL FORK !!! #
### It is intended only as a prototype ###
### Currently in development and not thoroughly tested ###

# Introduction #

Welcome to PSyclone `autodiff` fork, a prototype implementation of reverse-mode automatic differentation in PSyclone.   
For PSyclone itself, see [GitHub](https://github.com/stfc/PSyclone) and [ReadTheDocs](http://psyclone.readthedocs.io).  

# Description #

This implements reverse-mode automatic differentiation using source-to-source transformations.  
Compared to other tools, it uses a static array as a tape to record and restore values rather than a LIFO stack.

What has been implemented (but **not** necessarily tested):  
- transforming subroutines containing:
    - assignments,
    - calls to subroutines,
    - (some) unary and binary operations,
- scalar variables and arguments (no arrays),
- recording and restoring values from a tape,
- nested calls to subroutines,
- three different types of reversal schedules,
- simplification of the adjoint subroutine as a postprocessing step.  

What has *not* been implemented:
- functions and programs,
- nary operations,
- loops,
- control flow,
- array variables and arguments,
- activity analysis (dependence DAG),
- to-be-recorded (TBR) analysis,
- and much more.

A talk slide deck can be found in the `src/psyclone/autodiff/documents` subdirectory.

# Documentation #

The documentation is not yet hosted online.  
It can be generated using the instructions found in the `doc/reference_guide/` subdirectory.  
The Doxygen documentation built in `doc/reference_guide/build/html/_static/html/index.html` is suggested over the Sphinx one.

# Tutorial #

A tutorial may be found in the `src/psyclone/autodiff/tutorial` subdirectory.  

# Tests #

For now tests are all in the `src/psyclone/autodiff/tests` subdirectory and not in the `tests` one.  

