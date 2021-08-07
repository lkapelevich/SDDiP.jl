# SDDiP

This code is no longer maintained.You should use https://github.com/odow/SDDP.jl instead, which supports integer variables.

[![Build Status](https://travis-ci.org/lkapelevich/SDDiP.jl.svg?branch=master)](https://travis-ci.org/lkapelevich/SDDiP.jl)

[![Coverage Status](https://coveralls.io/repos/lkapelevich/SDDiP.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/lkapelevich/SDDiP.jl?branch=master)

[![codecov.io](http://codecov.io/github/lkapelevich/SDDiP.jl/coverage.svg?branch=master)](http://codecov.io/github/lkapelevich/SDDiP.jl?branch=master)

This repository contains old code moved to and maintained in [`SDDP.jl`](https://github.com/odow/SDDP.jl "SDDP.jl").

SDDiP is a version of SDDP for integer local or state variables.
If a model includes integer variables, we cannot compute cut coefficients in the way we normally do in SDDP (by querying duals from an LP solver). In SDDiP, each stage problem is solved via its Lagrangian dual in the backward pass.

The implementation of SDDiP in this package comes from the paper by *Zhou, J., Ahmed, S., Sun, X.A.* (2016): Nested Decomposition of Multistage Stochastic
Integer Programs with Binary State Variables.

## Installation
```julia
Pkg.clone("https://github.com/lkapelevich/SDDiP.jl")
```

This package is built upon [@odow](https://github.com/odow/ "")'s SDDP package, [`SDDP.jl`](https://github.com/odow/SDDP.jl "SDDP.jl"):
```julia
Pkg.clone("https://github.com/odow/SDDP.jl")
```

## Usage
Note there are some examples in the *examples* folder.

The first step is to become familiar with `SDDP.jl`. For this, have a look at the `SDDP.jl` [documentation](https://odow.github.io/SDDP.jl/latest/ "SDDP.jl latest").

### Calling the solver
Using `SDDP.jl`, a user should include a call to `setSDDiPsolver!` at the end of a stage problem definition:

```julia
setSDDiPsolver!(sp::JuMP.Model; method=KelleyMethod(), pattern=Pattern(), MIPsolver=sp.solver, LPsolver=MIPsolver)
```
The keyword arguments are as follows.
* `method`: An `AbstractLagrangianMethod` object from that defines how the Lagrangian will be solved. See the readme in *src/Lagrangian*.
* `pattern`: Allows us to choose to add Lagrangian cuts, Benders cuts, or strengthened Benders cuts every so often. See **Pattern** below.
* `MIPsolver`: The solver that will be used every time a MIP is solved.
* `LPsolver`: The solver that is used ever time an LP is solved.

### Declaring state variables
We can declare state variables using the `@binarystate` macro  (like [@state](https://github.com/odow/ "")).

There are four required arguments. The first three come from `@state`:
* A stage problem object,
* State at the end of a stage (written like a JuMP variable),
* State at the start of a stage.

The fourth argument is special for SDDiP:
* Type of variable: binary (Bin), integer (Int), or continuous (Cont).

For example:
```julia
@binarystate(sp, 1 <= x[i=1:3] <= 10, x0 == ones(3)[i] , Int)
```

Although our state may be described by variables that are binary, integer, or continuous,
only binary variables are ever registered as state variables in the SDDP model.

We will compute the binary expansion of integer and continuous variables
(as described in *Zhou, J., et. al.* (2016)).

For continuous variables, we can also specify a precision for binary expansion as
an optional fifth argument.

For example:
```julia
@binarystate(sp, 1 <= x[i=1:3] <= 10, x0 == ones(3)[i] , Cont, 0.01)
```

If a precision is not specified, the default precision is 0.1.

#### Pattern
The Pattern function has the definition:
```julia
Pattern(;benders=0, strengthened_benders=0, integer_optimality=0, lagrangian=1)
```
with, for example,
```
Pattern(benders=0, strengthened_benders=1, lagrangian=4)
```
meaning that in every cycle of 5 iterations, we should add 1 strengthened Benders cut and 4 Lagrangian cuts.
