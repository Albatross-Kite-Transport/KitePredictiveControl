# function get_control_function(model, input)
#     f_ip, dvs, psym, io_sys = ModelingToolkit.generate_control_function(IRSystem(model), input)
#     # any(ModelingToolkit.is_alg_equation, equations(io_sys)) && error("Systems with algebraic equations are not supported")
#     return (f_ip, dvs, psym, io_sys)
# end

"""
    di_derivatives(f::Function; backend) -> Tuple{Function,Function}

Return a tuple of functions that evaluate the gradient and Hessian of `f` using
DifferentiationInterface.jl with any given `backend`.
"""
function di_derivatives(f::Function; backend)
    function ∇f(g::AbstractVector{T}, x::Vararg{T,N}) where {T,N}
        DifferentiationInterface.gradient!(splat(f), g, backend, collect(x))
        return
    end
    function ∇²f(H::AbstractMatrix{T}, x::Vararg{T,N}) where {T,N}
        H_dense =
            DifferentiationInterface.hessian(splat(f), backend, collect(x))
        for i in 1:N, j in 1:i
            H[i, j] = H_dense[i, j]
        end
        return
    end
    return ∇f, ∇²f
end

function generate_f_h(kite::KPS4_3L, input, output, Ts)
    get_y = getu(kite.integrator, output)

    sys = kite.prob.f.sys
    state = unknowns(sys)
    nu = length(input)
    nx = length(state)
    ny = length(output)

    setu! = setp(kite.prob, [input[i] for i in 1:nu])
    integrator = kite.integrator

    function make_default_creator(state)
        keys = collect(state)
        return x -> Dict(k => x[i] for (i, k) in enumerate(keys))
    end
    create_default = make_default_creator(state)
    solver = QBDF()
    
    """
    Nonlinear discrete dynamics. Takes in complex state and returns simple state_plus
    """
    function f!(x_plus, x, u, _, _)
        norm_before = norm(integrator.u)
        reinit!(integrator, x; t0=1.0, tf=1.0 + Ts)
        @assert norm_before != norm(integrator.u)
        setu!(integrator, u)
        OrdinaryDiffEq.step!(integrator, Ts, true)
        !successful_retcode(integrator.sol) && @show u
        # @assert successful_retcode(integrator.sol)
        x_plus .= integrator.u
        return nothing
    end

    "Observer function"
    function h!(y, x, _, _)
        reinit!(integrator, x; t0=1.0, tf=1.0 + Ts)
        y .= get_y(integrator)
        return nothing
    end

    return (f!, h!, nx, nu, ny)
end
