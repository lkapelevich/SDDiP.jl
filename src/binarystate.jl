# This file includes modified source code from https://github.com/odow/SDDP.jl
# as at 5b3ba3910f6347765708dd6e7058e2fcf7d13ae5

#  Copyright 2017, Oscar Dowson and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

uberror(x) =  error("You must provide an upper bound on $(x).
    The lower the bound, the smaller the statespace.")

function anonymousstate!(sp::JuMP.Model, nstates::Int, initial::Vector{<:Real},
                             nameout::AbstractString, namein::AbstractString)
    statein  = @variable(sp, [i=1:nstates], basename="_bin_$namein", start=initial[i])
    stateout = @variable(sp,   [1:nstates], basename="_bin_$nameout")
    SDDP.statevariable!(sp, statein, stateout)
    stateout, statein
end

function binarystate!(sp::JuMP.Model, xout::JuMP.Variable, xin::JuMP.Variable)

    ub = sp.colUpper[xout.col]
    isinf(ub) && uberror(xout)

    nstates = bitsrequired(Int(ub))
    init = binexpand(Int(getvalue(xin)), length=nstates)

    v, v0 = anonymousstate!(sp, nstates, init, JuMP.getname(xout), JuMP.getname(xin))
    @constraint(sp, xout == bincontract(v))
    @constraint(sp, xin  == bincontract(v0))
end

function binarystate!(sp::JuMP.Model, xout::JuMP.Variable, xin::JuMP.Variable, eps::Float64)

    ub = sp.colUpper[xout.col]
    isinf(ub) && uberror(xout)

    nstates = bitsrequired(ub, eps)
    init = binexpand(float(getvalue(xin)), eps, length=nstates)

    v, v0 = anonymousstate!(sp, nstates, init, JuMP.getname(xout), JuMP.getname(xin))
    @constraint(sp, xout == bincontract(Float64, v, eps))
    @constraint(sp, xin  == bincontract(Float64, v0, eps))
end

function binarystate!(sp::JuMP.Model, xout::Array{JuMP.Variable}, xin::Array{JuMP.Variable})
    @assert size(xout) == size(xin)
    for i = 1:length(xin)
        binarystate!(sp, xout[i], xin[i])
    end
end

function binarystate!(sp::JuMP.Model, xout::Array{JuMP.Variable}, xin::Array{JuMP.Variable}, eps::Float64)
    @assert size(xout) == size(xin)
    for i = 1:length(xin)
        binarystate!(sp, xout[i], xin[i], eps)
    end
end

function binarystate!(sp::JuMP.Model, xout::JuMP.JuMPArray, xin::JuMP.JuMPArray)
    @assert length(keys(xin)) == length(keys(xout))
    for key in keys(xin)
        binarystate!(sp, xout[key...], xin[key...])
    end
end

function binarystate!(sp::JuMP.Model, xout::JuMP.JuMPArray, xin::JuMP.JuMPArray, eps::Float64)
    @assert length(keys(xin)) == length(keys(xout))
    for key in keys(xin)
        binarystate!(sp, xout[key...], xin[key...], eps)
    end
end

"""
    @binarystate(sp, stateleaving, stateentering, vartype)

# Description

Define a new state variable in the SDDiP subproblem `sp`.

# Arguments
 * `sp`               the subproblem
 * `stateleaving`     any valid JuMP `@variable` syntax to define the value of the state variable at the end of the stage
 * `stateentering`    any valid JuMP `@variable` syntax to define the value of the state variable at the beginning of the stage
 *  `vartype`         type of variable: choices are Bin, Int, or Cont
 *  `epsilon`         the accuracy of binary expansion for continuous variables (default is always 0.1 if not specified)

# Examples
 * `@binarystate(sp, 0 <= x[i=1:3] <= 1,  x0==rand([0 1], 3)[i], Bin )`
 * `@binarystate(sp,      y        <= 1,  y0==0.5              , Cont, 0.01)`
 * `@binarystate(sp,      z        <= 10, z0==5                , Int )`
"""
macro binarystate(sp, x, x0, vtype_args...)

    # Model
    sp = esc(sp)

    # eps and type
    if length(vtype_args) == 0
        vtype = :(Bin)
    elseif length(vtype_args) == 1
        vtype = vtype_args[1]
        (vtype == :Cont) && (eps = 0.1)
    elseif length(vtype_args) == 2
        vtype = vtype_args[1]
        @assert vtype == :Cont
        eps = vtype_args[2]
    else
        error("Type of variable $(x) not properly specified in @binarystate.")
    end

    # If it's already a binary variable, just call @state
    if vtype == :Bin
        return quote
            # Ordinary @state call
            $(Expr(:macrocall, Symbol("@state"), sp, esc(x), esc(x0)))
        end

    # Otherwise prepare for reformulation
    elseif vtype in [:Int, :Cont]

        # Get variables exactly as in SDDP.jl
        @assert x0.head == :call && x0.args[1] == :(==) # must be ==
        _, symin, init = x0.args
        if SDDP.is_comparison(x)
            if length(x.args) == 5
                xin = SDDP._copy(x.args[3])
            elseif length(x.args) == 3
                xin = SDDP._copy(x.args[2])
            else
                error("Unknown format for $(x)")
            end
        else
            xin = SDDP._copy(x)
        end
        if isa(xin, Expr)                   # x has indices
            xin.args[1] = symin             # so just change the name
        else                                # its just a Symbol
            xin = symin                     # so change the Symbol
        end

        # The names the user chose will be dummy variables in the model
        varin, varout = gensym(), gensym()

        code = quote
            $varout = $(Expr(:macrocall, Symbol("@variable"), sp, esc(x)))
            $varin  = $(Expr(:macrocall, Symbol("@variable"), sp, esc(xin), Expr(SDDP.KW_SYM, SDDP.START, esc(init))))
        end

        if vtype == :Int
            # Integer case
            push!(code.args, quote
                binarystate!($sp, $varout, $varin)
            end)
        else
            # Continuous case
            push!(code.args, quote
                binarystate!($sp, $varout, $varin, $(esc(eps)))
            end)
        end

        push!(code.args, quote
            $varout, $varin
        end)
        return code

    else
        error("You must specify a variable type for $(x) because you are calling @binarystate.
                Choices are Bin, Int, or Cont.")
    end # vtype
end # function

"""
    @binarystates(sp, begin
        stateleaving1, stateentering1, vartype
        stateleaving2, stateentering2, vartype
    end)
# Description
Define a new state variables in the subproblem `sp`.
# Arguments
* `sp`               the subproblem
* `stateleaving`     any valid JuMP `@variable` syntax to define the value of the state variable at the end of the stage
* `stateentering`    any valid JuMP `@variable` syntax to define the value of the state variable at the beginning of the stage
*  `vartype`         type of variable: choices are Bin, Int, or Cont
*  `epsilon`         the accuracy of binary expansion for continuous variables (default is always 0.1 if not specified)

# Usage
    @binarystates(sp, begin
        0 <= x[i=1:3] <= 1,  x0==rand([0 1], 3)[i], Bin
             y        <= 1,  y0==0.5              , Cont, 0.01
             z        <= 10, z0==5                , Int
     end)
"""
macro binarystates(m, b)
    @assert b.head == :block || error("Invalid syntax for @binarystates.")
    code = quote end
    for line in b.args
        if !Base.Meta.isexpr(line, :line)
            if line.head == :tuple && 3 <= length(line.args) <= 4
                escargs = []
                for ex in line.args
                    push!(escargs, esc(ex))
                end
                push!(code.args, Expr(:macrocall, Symbol("@binarystate"), esc(m), escargs...))
            else
                error("Unknown arguments in @binarystates.")
            end
        end
    end
    push!(code.args, :(nothing))
    return code
end
