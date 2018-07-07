# Lagrangian.jl

Solver for optimal Lagrange multipliers in linear and mixed integer problems.

## Overview
MIPs and LPs with complicating constraints can be solved via their Lagrangian dual. A complicating constraint is a constraint we relax and move into the objective.

We provide the primal problem as a JuMP model and its complicating constraints as JuMP constraint references to Lagrangian.jl, in order to compute optimal Lagrange multipliers and a Lagrangian bound for the primal problem.

We also need to provide some other information.

#### Information about the problem:

* Relaxed RHSs of complicating constraints
   * Complicating constraints aren't really removed, we just change their RHS to something that will ensure they aren't active
   * If you have ≥ constraints, pick a very small number (e.g. -M), and the constraint will become LHS ≥ -M inside the solver
   * If you have ≤ or = constraints, pick a very big number (e.g. M), and the constraint will become LHS ≤ M inside the solver

All this should be stored in a `LinearProgramData` object.

#### Information for the solver:

 * An initial bound for the dual problem
    * If a primal is a maximisation problem, pick a very small number to be the lower bound of the dual
    * If a primal is a minimisation problem, pick a very large number to be the upper bound of the dual
 * Information specific to the method you are using. See Solve Methods.

This should be stored in an `AbstractLagrangianMethod` object.

## Usage

1) Create a Lagrangian Method object. There are two types at the moment.
* LevelMethod
* SubgradientDescent

See **Solve methods**.

2) Create a `LinearProgramData` object to describe your problem.

```julia
    LinearProgramData(obj::QuadExpr, constraints::Vector{<:ConstraintRef}, relaxed_bounds::Vector{Float64}; method=LevelMethod(), problem_class=LinearProgram())

Creates a `LinearProgramData` object for calling `lagrangiansolve!`.

# Arguments
* obj:               Objective of the linear program without relaxation.
* constraints:       A vector of type `JuMP.ConstraintRef` with the contraints to be relaxed.
* relaxed_bounds:    A RHS that will replace the RHS of the constraint being relaxed (choose a big number for ≤ or = constraints, and something small for ≥).
* method:            Solving parameters, of type `AbstractLagrangianMethod`.
* problem_class:     To overload how the primal problem is solved.
```
Notice that the method (from (1)) is one of the inputs.

3) Call the solve function.

```julia
lagrangiansolve!{M<:AbstractLagrangeinMethod}(l::LinearProgramData{M}, m::JuMP.Model, π::Vector{Float64})
```
Here `l` is the object from (2) and `m` is the primal model.
`π` is an initial guess for the duals. The vector π is modified in place.
`lagrangiansolve!` returns the optimal objective and a status. Possible statuses are:

 * :optimal - this means that we converged on a solution
 * :iterationLimit - this means that the solution did not converge after the maximum number of iterations specified

## Solve methods
In `SDDiP.jl`, the value of the field `initialbound` for all the methods below
will be ignored. This is because we can find the actual objective of the
Lagrangian dual of any mixed-integer subproblem simply by solving
it (it is only the dual multipliers we are interested in finding). We will
always set `initialbound` to the actual objective of our subproblems.

In general, `Lagrangian.jl` can be used to find the objective of LPs/MIPs with complicating constraints that are never explicitly included and
an `initialbound` will affect the method.


#### Kelley's Cutting Planes
Passing a `KelleyMethod` type object to `LinearProgramData` means we solve the Lagrangian dual with Kelley's cutting-plane method.

The definition of the constructor is:
```julia
KelleyMethod(; initialbound=0.0, tol=Unit(1e-6), maxit=10_000)
```
The arguments are as follows.

* initialbound:   Initial upper or lower bound for the Lagrangian dual,
* tol:            Tolerance for stopping, see **Tolerances**,
* maxit:          Maximum number of iterations before terminating the method.

#### Level Method
Passing a `LevelMethod` type object to `LinearProgramData` means we solve the Lagrangian
using the level method (Lemarechal, Nemirovskii, Nesterov, 1992).

The definition of the constructor is:
```julia
LevelMethod(; initialbound::Float64, level=0.5, tol=Unit(1e-6), quadsolver=UnsetSolver(), maxit=10_000)
```
The arguments are as follows.

* initialbound:   Initial upper or lower bound for the Lagrangian dual,
* level:          Level method parameter 0 ≥ λ ≥ 1,
* tol:            Tolerance for stopping, see **Tolerances**,
* quadsolver:     A quadratic solver,
* maxit:          Maximum number of iterations before terminating the method.

#### Subgradient Descent
Passing a `SubgradientDescent` type object to `LinearProgramData` means we solve the
Lagrangian using a subgradient descent with *Polyak's* stepsizes.

The definition of the constructor is:
```julia
SubgradientMethod(; initialbound::Float64, tol=Unit(1e-6), wait=30, maxit=10_000)
```

The arguments are as follows.

* initialbound:   Initial upper or lower bound for the Lagrangian dual,
* tol:            Tolerance for stopping, see **Tolerances**,
* wait:           How many iterations with no improvement we allow before halving the stepsize,
* maxit:          Maximum number of iterations before terminating the method.

#### Tolerances
You can pass through the stopping tolerance as:
 - `tol = Absolute(val)` to stop when within absolute tolerance of value `val`;
 - `tol = Relative(val)` to stop when within relative tolerance of `val`;
 - `tol = Unit(val)` to stop when within unit tolerance of `val`.

#### Expected Changes
* Lagrangian multipliers are currently the opposite sign (+/-) to the LP duals.
* Some better defaults for initial duals, relaxed constraints, dual bounds
