# module KitePredictiveControl

using ModelPredictiveControl
using ModelingToolkit
using ModelingToolkit: D_nounits as D, t_nounits as t
using KiteModels, Plots, Serialization, JuliaSimCompiler, OrdinaryDiffEq, ForwardDiff, RuntimeGeneratedFunctions, LinearAlgebra
using JuMP, DAQP, MadNLP
daqp = Model(DAQP.Optimizer, add_bridges=false)

# export run_controller

include("mtk_interface.jl")

set_data_path(joinpath(pwd(), "data"))
if ! @isdefined kite
    kite::KPS4_3L = KPS4_3L(KCU(se("system_3l.yaml")))
    kite.torque_control = true
    KiteModels.init_sim!(kite; prn=true, torque_control=true)

end
kite_model, inputs = model!(kite, kite.pos, kite.vel)
kite_model = complete(kite_model)
outputs = vcat(
    vcat(kite_model.flap_angle), 
    reduce(vcat, collect(kite_model.pos[:, 1:kite.num_flap_C])), 
    reduce(vcat, collect(kite_model.pos[:, kite.num_flap_D+1:kite.num_A])),
    vcat(kite_model.tether_length)
)
initial_outputs = vcat(
    vcat(kite.flap_angle), 
    reduce(vcat, kite.pos[1:kite.num_flap_C]), 
    reduce(vcat, kite.pos[kite.num_flap_D+1:kite.num_A]),
    vcat(kite.tether_lengths)
)

(f_ip, dvs, psym, io_sys) = get_control_function(kite_model, inputs)
f!, (h!, nu, ny, nx, vu, vy, vx) = generate_f_h(io_sys, inputs, outputs, f_ip, dvs, psym)

[defaults(io_sys)[kite_model.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in 1:kite.num_flap_C-1]
[defaults(io_sys)[kite_model.pos[j, i]] = kite.pos[i][j] for j in 1:3 for i in kite.num_flap_D+1:kite.num_A]
[defaults(io_sys)[kite_model.flap_angle[i]] = kite.flap_angle[i] for i in 1:2]
[defaults(io_sys)[kite_model.tether_length[i]] = kite.tether_lengths[i] for i in 1:3]

x_0 = JuliaSimCompiler.initial_conditions(io_sys, defaults(io_sys), psym)[1]

Ts = 0.1
N = 600
model = setname!(NonLinModel(f!, h!, Ts, nu, nx, ny, solver=RungeKutta(4; supersample=Int(1e5*Ts))); u=vu, x=vx, y=vy)
setstate!(model, x_0)

println("linear mpc")
max = 200.0
gain = 10e3
Hp, Hc = 40, 1
umin, umax = fill(-max, nu), fill(max, nu)
Mwt = fill(0.0, ny)
output_idxs = [findfirst(x -> x == string(kite_model.tether_length[i]), model.yname) for i in 1:3]
Mwt[output_idxs] .= 1.0
Nwt = fill(0.1, nu)

if !@isdefined linmodel
    println("linearize")
    @time linmodel = ModelPredictiveControl.linearize(model)
    println("sanity check")
    u = [100, 100, -100]
    res = sim!(linmodel, N, u; x_0 = x_0)
    display(plot(res; plotx=1:3, ploty=false, plotu=false))
end

α=0.01; σQ=[0.1, 1.0]; σR=[5.0]; nint_u=[1]; σQint_u=[0.1]
estim = KalmanFilter(linmodel; nint_u = fill(1, nu), σQint_u=fill(0.1, nu), σQ = fill(0.1, nx), σR = fill(5.0, ny))
mpc = LinMPC(estim; Hp, Hc, Mwt=Mwt, Nwt=Nwt, Cwt=Inf)
mpc = setconstraint!(mpc; umin=umin, umax=umax)
@time res = sim!(mpc, N, initial_outputs, x_0 = x_0, lastu = [0, 0, -100.0])
display(plot(res; plotx=false, ploty=output_idxs, plotu=true, plotxwithx̂=1:3))

# xhat should be 86

# println("nonlinear stepping")
# function sim_adapt!(mpc, model, N, ry, plant, x_0, x̂_0, y_step=[0, 0])
#     U_data, Y_data, Ry_data = zeros(plant.nu, N), zeros(plant.ny, N), zeros(plant.ny, N)
#     setstate!(plant, x_0)
#     initstate!(mpc, [100, 100, -100], plant())
#     setstate!(mpc, x̂_0)
#     for i = 1:N
#         @show i
#         y = plant() + y_step
#         x̂ = preparestate!(mpc, y)
#         @show x̂
#         u = moveinput!(mpc, ry)
#         setmodel!(mpc, model)
#         U_data[:,i], Y_data[:,i], Ry_data[:,i] = u, y, ry
#         updatestate!(mpc, u, y) # update mpc state estimate
#         updatestate!(plant, u)  # update plant simulator
#     end
#     res = SimResult(mpc, U_data, Y_data; Ry_data)
#     return res
# end

# x̂_0 = vcat(x_0, wanted_outputs)
# res_slin = sim_adapt!(nmpc, model, N, wanted_outputs, model, x_0, x̂_0)
# display(plot(res_slin))

nothing

# end