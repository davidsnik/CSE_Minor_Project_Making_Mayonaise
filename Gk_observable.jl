
using Random
using StaticArrays
using Molly
using GLMakie
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity, visualize, Langevin
using Unitful
using DelimitedFiles
using Statistics
using LinearAlgebra: norm, dot
using GeometryOps: convex_hull
using Colors: RGB
using GeoInterface
using StatsBase  # for autocov
using FFTW 


# Optional progress bars
const HAS_PROGRESSMETER = Ref(false)
try
    @eval import ProgressMeter
    HAS_PROGRESSMETER[] = true
catch
    @warn "ProgressMeter.jl not available; progress bars will be disabled."
end
include(joinpath(@__DIR__, "system_and_simulation_functions.jl"))
# MyCoordinatesLogger, run_and_collect!, build_emulsion_system
include(joinpath(@__DIR__, "plotting_functions.jl"))
# visualize_with_progress, plot_hull_points, visualize_soft_spheres_with_progress

# Setting random seed for reproducibility
Random.seed!(1234)

# minimum-image in 2D periodic box [0,L)
@inline function min_image(dr::SVector{2,Float64}, L::Float64)
    return dr .- L .* round.(dr ./ L)
end


# function shear_stress_xy(sys::Molly.System, L::Float64, inter)
#     A = L^2
#     N = length(sys)

#     # compute COM drift
# vx̄ = mean(getindex.(sys.velocities, 1))
# vȳ = mean(getindex.(sys.velocities, 2))

# s_kin = 0.0
# @inbounds for i in 1:N
#     m  = sys.atoms[i].mass
#     vx = sys.velocities[i][1] - vx̄
#     vy = sys.velocities[i][2] - vȳ
#     s_kin += m * vx * vy
# end
#     # virial (pairs)
#     s_vir = 0.0
#     rc = (inter.cutoff isa Molly.DistanceCutoff) ? inter.cutoff.dist_cutoff : Inf

#     @inbounds for i in 1:N-1
#         xi = sys.coords[i]
#         ai = sys.atoms[i]
#         for j in i+1:N
#             xj = sys.coords[j]
#             aj = sys.atoms[j]

#             dr = min_image(xj - xi, L)
#             r  = norm(dr)
#             if r == 0.0 || r > rc
#                 continue
#             end

#             # force on i due to j:
#             fij = Molly.force(inter, dr, ai, aj, 1.0)
#             s_vir += dr[1] * fij[2]
#         end
#     end

#     return 1*(s_kin + s_vir) / A
# end

function kbT_2D_from_velocities(vels, masses, bulk_ids)

    "version that removes COM drift test"
    vx̄ = mean(getindex.(vels[bulk_ids], 1))
    vȳ = mean(getindex.(vels[bulk_ids], 2))
    T_sum = 0.0
    for i in bulk_ids
        vx = vels[i][1] - vx̄
        vy = vels[i][2] - vȳ
        T_sum += masses[i] * (vx^2 + vy^2)
    end
    kbT = T_sum / (2length(bulk_ids))
    return kbT
end
"version that computes G(t) from shear stress series and velocities"
function compute_GreenKubo_Gt(sigma_xy_series, mass, v, bulk_ids, box_side)
    A = box_side^2
    N = length(bulk_ids)

    T_sum = 0.0
    @inbounds for i in bulk_ids
        vx, vy = v[i][1], v[i][2]
        T_sum += mass[i] * (vx^2 + vy^2)
    end
    kbT = T_sum / (2 * N)      # 2D: 2 dof per particle

    lags = 0:(length(sigma_xy_series)-1)
    C = autocov(sigma_xy_series, lags)
    G = (A / kbT) .* C
    return G
end
"Compute autocorrelation function of sgima_xy series with blocks (just did it tot est)"
function compute_autocorrelation(sigma_xy_series; maxlag=100_000, nblocks=1)
    N = length(sigma_xy_series)
    Lb = Int(floor(N / nblocks))
    maxlag = min(maxlag, Lb-1)
    Csum = zeros(Float64, maxlag+1)
    for b in 1:nblocks
        i1 = (b-1)*Lb + 1
        i2 = b*Lb
        x = @view sigma_xy_series[i1:i2]
        Csum .+= autocov(x, 0:maxlag; demean=true)
    end
    return Csum ./ nblocks
end
"Version that takes kbT directly as input"
function compute_GreenKubo_Gt_from_kbT(sigma_xy_series, box_side, kbT; maxlag=100_000)
    A = box_side^2
    maxlag = min(maxlag, length(sigma_xy_series) - 1)  # limit to actual data length
    C = autocov(sigma_xy_series, 0:maxlag; demean=false)
    G = (A / kbT) .* C
    # lags = 0:(length(sigma_xy_series)-1)
    # C = autocov(sigma_xy_series, lags)   # we’ll truncate/taper next
    # return (A / kbT) .* C
    return G
end
"cosine taper only on the last `frac` of the signal"
function tail_taper_window(N; frac=0.2)
    w = ones(Float64, N)
    m = max(2, Int(round(frac * N)))
    n = 0:(m-1)
    taper = 0.5 .* (1 .+ cos.(π .* n ./ (m-1))) 
    w[end-m+1:end] .= taper
    return w
end


"Smoothly average G(t) over blocks to reduce noise"
function greenkubo_G_blockavg(sigma_xy_series, box_side, kbT; maxlag=100_000, nblocks=1)
    A = box_side^2
    N = length(sigma_xy_series)
    Lb = Int(floor(N / nblocks))
    maxlag = min(maxlag, Lb-1)

    Gsum = zeros(Float64, maxlag+1)
    for b in 1:nblocks
        i1 = (b-1)*Lb + 1
        i2 = b*Lb
        x = @view sigma_xy_series[i1:i2]
        C = autocov(x, 0:maxlag; demean=true)
        Gsum .+= (A / kbT) .* C
    end
    return Gsum ./ nblocks
end


mutable struct StrideSeriesLogger{T}
    stride::Int
    data::Vector{T}
end

StrideSeriesLogger(::Type{T}, stride::Int) where {T} = StrideSeriesLogger{T}(stride, T[])
StrideSeriesLogger(stride::Int) = StrideSeriesLogger(Float64, stride)

Base.values(logger::StrideSeriesLogger) = logger.data

"sigma_xy observable from Molly pressure tensor (fast)"
@inline function obs_sigma_xy(sys::Molly.System, buffers, neighbors, step_n::Int;
                              n_threads::Integer=Threads.nthreads(), kwargs...)
    P = Molly.pressure(sys, neighbors, step_n, buffers; n_threads=n_threads)
    return 0.5*(P[1, 2] + P[2, 1])  # ensure symmetry
end


function Molly.log_property!(logger::StrideSeriesLogger{T},
                             sys, buffers, neighbors=nothing, step_n::Int=0;
                             n_threads::Integer=Threads.nthreads(), kwargs...) where {T}
    (logger.stride > 1 && step_n % logger.stride != 0) && return

    # # fallback if neighbors aren't passed (usually they are)
    # if neighbors === nothing
    #     neighbors = Molly.find_neighbors(sys; n_threads=n_threads)
    # end
    push!(logger.data, T(obs_sigma_xy(sys, buffers, neighbors, step_n; n_threads=n_threads)))
    return
end


function GreenKubo_fft(G0, dt)
    N = length(G0)
    n = 0:(N-1)
    window = tail_taper_window(N; frac=0.2) # cosine taper on last 20%
    #window=1 #test with no window
    Gw = G0 .* window #Apply window
    F = dt .* rfft(Gw)                      # ≈ ∫ G(t) e^{-i ω t} dt
    ω = 2π .* (0:length(F)-1) ./ (N * dt)         # angular frequency grid
    Gstar = 1im.* ω .* F
    Gprime =  real(Gstar)
    Gloss  = imag(Gstar)
    return ω, Gprime, Gloss
end

"Progress bar wrapper for simulate!"
function simulate_with_progress!(sys, sim, nsteps;
                                 chunk=5_000,
                                 desc="simulate",
                                 run_loggers=:skipzero)

    if !HAS_PROGRESSMETER[]
        simulate!(sys, sim, nsteps; run_loggers=run_loggers)
        return
    end

    prog = ProgressMeter.Progress(nsteps; desc=desc)
    steps_done = 0

    while steps_done < nsteps
        n = min(chunk, nsteps - steps_done)

        simulate!(sys, sim, n; run_loggers=run_loggers)

        steps_done += n
        ProgressMeter.next!(prog; step=n)
    end

    ProgressMeter.finish!(prog)
    return
end

using InteractiveUtils


#region ------- Simulation Parameters -----
# -------- General system parameters --------
volume_fraction_oil = 0.75                   
box_side            = 5.0
temperature         = 1.0                   # temp in reduced units

# ------------- Droplet settings ------------
#n_droplets = 20               # number of droplets is now based on volume fraction and droplet size
droplet_radius     = 0.1 
droplet_mass       = 100.0
energy_strength    = 0.1
#energy_strength    = 1.0
cutoff             = 5*droplet_radius    # cutoff is disabled for droplet_radius>0.3 because then the number of droplets is small (so there is no need to use neoighbors list)
minimal_dist_frac  = 0.75                  # minimal  distance between initial droplet centers as fraction of 2*R

# ------------ Simulation settings ----------

"dt_sample is the time interval between stress samples.  So in our system n_integrator=T/dt is number of integrator steps.
 Then every stress_every steps we sample stress.
 So our acctual usable amount of time samples for GK is n_integrator/stress_every.
 Maxlag_GT determines how many of these samples we use to compute G(t).
The actual tmax for G(t) is maxlag_Gt * dt_sample.
    nblcoks is the amoutn of blovks used for averaging to reduce noise in G(t).
The Nyquist frequency is π/dt_sample. 
The maximum frequency resolution is 2π/(maxlag_Gt*dt_sample)=2pi/(tmax)."
#max_lags *dt is time signal length for G(t)
dt = 0.00025                    # integrator time step
maxlag_Gt = 100000        # max lag time for G(t) calculation
T = 2000                              # CHANGE TIME HERE
nblocks=8
nsteps = Int(round(T/dt))
stress_every = 5                          # sample every x steps (adjust if too heavy)
dt_sample = stress_every * dt
# Desired output fps; we will subsample to approximate this while keeping duration = T
desired_fps = 100
realistic_time = true          # if true, the video plays in real time and desired fps is disregarded; if false, output fps matches desired fps 
output_folder = "soft_spheres_mp4" # folder to save visualization output (mp4, temp plot, energy plot)
#endregion
#region ------- System Initialization -----
volume_oil   = volume_fraction_oil * box_side^2
volume_water = (1.0 - volume_fraction_oil) * box_side^2
droplet_area = min(pi*droplet_radius^2, volume_oil)         # area of ONE droplet
if droplet_area == volume_oil
    println("Warning: Only enough oil for one droplet.")
end
n_droplets::Int = floor(Int, volume_oil / droplet_area)
println("Number of droplets: ", n_droplets)
println("Total oil area: ", n_droplets * droplet_area)
println("Real volume fraction of oil: ", (n_droplets * droplet_area) / box_side^2)


droplets = [Atom(mass=droplet_mass, σ=2*droplet_radius, ϵ=energy_strength) for _ in 1:n_droplets]
boundary = Molly.RectangularBoundary(SVector{2,Float64}(box_side, box_side))
init_coord = Molly.place_atoms(n_droplets, boundary, min_dist=2.0*droplet_radius*minimal_dist_frac)
v0_bulk = [random_velocity(droplet_mass, temperature; dims=2) for _ in 1:n_droplets]

if droplet_radius > 0.3
    pairwise_inter = (Molly.SoftSphere(use_neighbors=false),)
else
    pairwise_inter = (Molly.SoftSphere(cutoff=Molly.DistanceCutoff(cutoff), use_neighbors=true),)
end
#endregion
#region ------- Plot the force function -----
rs = range(0.1*droplet_radius, 4*droplet_radius, length=500) # Distance range 
force_magnitudes = Float64[]
forces_diff_str = Vector{Float64}[]
energy_strs= [energy_strength,0.1, 1.0, 10.0, 100.0]  # Different energy strengths to compare

for s in energy_strs
    for r in rs
        atom1 = Atom(mass=1.0, σ=2*droplet_radius, ϵ=s)
        atom2 = Atom(mass=1.0, σ=2*droplet_radius, ϵ=s)
        
        # Molly's force function expects a displacement vector (dr)
        dr = SVector(r, 0.0) 
        
        # Call the Molly force function directly
        # We pass 1.0 as force_units to get raw numbers back
        f_vec = force(pairwise_inter[1], dr, atom1, atom2, 1.0)
        
        # Store the magnitude (norm) of the force vector
        push!(force_magnitudes, norm(f_vec))
    end
    push!(forces_diff_str, copy(force_magnitudes))
    empty!(force_magnitudes)
end

# 5. Plot using GLMakie
fig = Figure(size=(800, 500))
ax = Axis(fig[1, 1], 
    title = "Soft Sphere Force Function (via Molly.jl)",
    xlabel = "Distance (r)", 
    ylabel = "Force Magnitude (F)",
    yscale = log10 # Log scale is helpful because the 1/r^13 rise is very steep
)
 colors = [:red, :green, :blue, :orange, :purple]
for (i, s) in enumerate(energy_strs)
    lines!(ax, rs, forces_diff_str[i], label="ϵ = $s", color=colors[i])
end
vlines!(ax, [droplet_radius], color=:black, linestyle=:dash, label="Sigma (Diameter)")

# Add a limit to avoid the graph looking empty due to the infinite rise at r=0
ylims!(ax, 1e-3, 1e4) 
axislegend(ax)
outdir = joinpath(@__DIR__, output_folder)
isdir(outdir) || mkpath(outdir) # Create directory if it doesn't exist
save(joinpath(outdir, "soft_sphere_force_plot.png"), fig)
println("Force plot saved as 'soft_sphere_force_plot.png'.")
#endregion
#region ------- Build the system and run simulation -----

if droplet_radius > 0.3
  

    sys = System(
        atoms = droplets,
        coords = init_coord,
        velocities = v0_bulk,
        boundary = boundary,
        pairwise_inters= pairwise_inter,
        loggers = (coords=MyCoordinatesLogger(10000, dims=2),),
        #loggers =(),
        energy_units = Unitful.NoUnits,
        force_units = Unitful.NoUnits
    )
else
    cellListMap_finder = Molly.CellListMapNeighborFinder(
        eligible=trues(n_droplets, n_droplets),
        dist_cutoff=cutoff,
        x0=init_coord,
        unit_cell = boundary,
        n_steps = 5, # update neighbors more often so moving walls stay accurate
        dims = 2,
    )
    sys = System(
        atoms = droplets,
        coords = init_coord,
        velocities = v0_bulk,
        boundary = boundary,
        pairwise_inters= pairwise_inter,
        neighbor_finder = cellListMap_finder,
        loggers = (coords=MyCoordinatesLogger(1, dims=2),),
        energy_units = Unitful.NoUnits,
        force_units = Unitful.NoUnits
    )
end

sys = Molly.System(
    atoms = sys.atoms,
    coords = sys.coords,
    velocities = sys.velocities,
    boundary = sys.boundary,
    pairwise_inters = sys.pairwise_inters,
    neighbor_finder = sys.neighbor_finder,
    loggers = (
        coords = MyCoordinatesLogger(1, dims=2),
        sigma_xy = StrideSeriesLogger(stress_every),    # logger for σ_xy observable
    ),
    energy_units = Unitful.NoUnits,
    force_units  = Unitful.NoUnits
)
simulator = VelocityVerlet(dt = dt,coupling = Molly.AndersenThermostat(temperature,1.0))


# 1) Equilibration: thermostat ON
n_eq = Int(round(0.3 * nsteps))          # x% equilibration, in our case 0.2 or 0.3 of total time
sim_eq = VelocityVerlet(dt=dt, coupling=Molly.AndersenThermostat(temperature, 1.0))

# Run equilibration WITHOUT recording stress
simulate_with_progress!(sys, sim_eq, n_eq; chunk=5_000, desc="Equilibration", run_loggers=false)
#Test
##Tests
# vx̄ = mean(getindex.(sys.velocities, 1))
# vȳ = mean(getindex.(sys.velocities, 2))
# P  = sum(sys.atoms[i].mass .* sys.velocities[i] for i in 1:length(sys))
#@show vx̄ vȳ P mean(sys.loggers.sigma_xy.data)
Molly.remove_CM_motion!(sys) # remove CM motion before production created by thermostat
empty!(sys.loggers.sigma_xy.data)


# 2) Production: thermostat OFF (this is what we use for Green–Kubo)
n_prod = nsteps - n_eq
sim_prod = VelocityVerlet(dt=dt)  

sigma_xy_series = Float64[]
t_series = Float64[]

# simulate!(sys, sim_prod, n_prod; run_loggers=:skipzero)
simulate_with_progress!(sys, sim_prod, n_prod;chunk=5_000, desc="Production", run_loggers=:skipzero)


sigma_xy_series = values(sys.loggers.sigma_xy)

#Test
# vx̄ = mean(getindex.(sys.velocities, 1))
# vȳ = mean(getindex.(sys.velocities, 2))
# P  = sum(sys.atoms[i].mass .* sys.velocities[i] for i in 1:length(sys))
# @show vx̄ vȳ P mean(sigma_xy_series)
dt_sample = stress_every * dt
t_series = (0:length(sigma_xy_series)-1) .* dt_sample

using StatsBase  # autocov

# after we have sigma_xy_series, dt_sample, box_side
masses = [sys.atoms[i].mass for i in 1:length(sys)]
bulk_ids = 1:length(sys)

kbT_kin = kbT_2D_from_velocities(sys.velocities, masses, bulk_ids)
kbT = kbT_kin
burn_frac = 0.2  # fraction of sigmaxy data to discard as burn-in, due to remaining transients
burn = Int(floor(burn_frac * length(sigma_xy_series)))

sigma_use = sigma_xy_series[burn+1:end]
sigma_use .-= mean(sigma_use)   # de-mean AFTER burn-in

# remove mean stress BEFORE autocov
# sigma_fluct = sigma_xy_series .- mean(sigma_xy_series)  #Without burn-in
sigma_fluct = sigma_use #- mean(sigma_use)  #With burn-in
Molly.remove_CM_motion!(sys)  # once, right before GK production

#sigma_fluct =sigma_xy_series #if using block averegaing

#G = compute_GreenKubo_Gt_from_kbT(sigma_fluct, box_side, kbT; maxlag=maxlag_Gt) #no block averaging
#C = compute_autocorrelation(sigma_fluct; maxlag=maxlag_Gt,nblocks=4) # autocorrelation function
G = greenkubo_G_blockavg(sigma_fluct, box_side, kbT; maxlag=maxlag_Gt, nblocks=nblocks) #block averaging to reduce noise
ω, Gp, Gpp = GreenKubo_fft(G, dt_sample)

# keep only positive, non-zero frequencies (avoid ω=0 point)
N = length(ω)
half = Int(floor(N/2))
ωp  = ω[2:end]
Gp_ = Gp[2:end]
Gpp_= Gpp[2:end]
fig2 = Figure(size=(800, 400))
ax2 = Axis(fig2[1,1], xlabel="t", ylabel="G(t)")
# tG = (0:length(G)-1) .* dt_sample
tG = (0:length(G)-1) .* dt_sample #time array for G(t)
lines!(ax2, tG, G)
save(joinpath(outdir, "G_of_t.png"), fig2)
fig3 = Figure(size=(900, 450))

ax3 = Axis(fig3[1,1],
    xlabel="ω",
    ylabel="Modulus",
    xscale=log10
)

lines!(ax3, ωp, Gp_, label="G'(ω)")
lines!(ax3, ωp, Gpp_, label="G''(ω)")
axislegend(ax3, position=:rb)

save(joinpath(outdir, "Gp_Gpp_vs_omega.png"), fig3)

"This checkes type stability of functions, ours are all type stable:"
# @code_warntype compute_GreenKubo_Gt_from_kbT(sigma_xy_series, box_side, kbT)
# @code_warntype GreenKubo_fft(G, dt)

ωN = π/dt_sample



using GLMakie

# --- Build time array if not already stored ---
# If you already pushed t_series during sampling, skip this
if !@isdefined(t_series) || length(t_series) != length(sigma_xy_series)
    t_series = (0:length(sigma_xy_series)-1) .* dt_sample
end

# --- Plot σ_xy(t) ---
fig = Figure(size=(900, 400))
ax = Axis(
    fig[1, 1],
    xlabel = "Time",
    ylabel = "σ_xy",
    title  = "Shear stress σ_xy(t)"
)

lines!(ax, t_series, sigma_xy_series)

# --- Save ---
outdir = joinpath(@__DIR__, output_folder)
isdir(outdir) || mkpath(outdir)

save(joinpath(outdir, "sigma_xy_vs_time.png"), fig)
@show mean(sigma_xy_series)
println("Saved shear stress plot to sigma_xy_vs_time.png")
