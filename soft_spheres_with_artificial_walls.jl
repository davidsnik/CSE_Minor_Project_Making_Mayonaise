
using Random
using StaticArrays
using Molly
using GLMakie  # Make sure you have this loaded
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity, visualize, Langevin
using Unitful
using DelimitedFiles
using Statistics
using LinearAlgebra: norm, dot
using GeometryOps: convex_hull
using Colors: RGB
using GeoInterface

# Optional progress bars
const HAS_PROGRESSMETER = Ref(false)
try
    @eval import ProgressMeter
    HAS_PROGRESSMETER[] = true
catch
    @warn "ProgressMeter.jl not available; progress bars will be disabled."
end

#region -------- Include function files --------
#include(joinpath(@__DIR__, "functions.jl"))

# After each include there is a list of functions made available.
include(joinpath(@__DIR__, "wrapping_and_confining_functions.jl"))
# wrap_x, wrap_delta, periodic_confinement_y!,
include(joinpath(@__DIR__, "beads_placing_functions.jl"))
# generate_droplet_centers, generate_multi_droplet, generate_multi_droplet_and_matrix, generate_outside_droplets
include(joinpath(@__DIR__, "forces_and_energy_functions.jl"))
# wrap_delta, emulsion_force_magnitude, emulsion_potential, Molly.pairwise_force, Molly.force, emulsion_energy, current_temperature
include(joinpath(@__DIR__, "wall_and_shear_functions.jl"))  
# make_rough_wall_particles, make_shear_profile, shear_state, wall_sign, wall_displacement_from_shear 
# wall_speed_from_shear_rate, average_wall_gap,wall_velocity_from_shear, apply_wall_drag!, enforce_wall_motion!
include(joinpath(@__DIR__, "system_and_simulation_functions.jl"))
# MyCoordinatesLogger, run_and_collect!, build_emulsion_system
include(joinpath(@__DIR__, "plotting_functions.jl"))
# visualize_with_progress, plot_hull_points
include(joinpath(@__DIR__, "cluster_functions.jl"))
# detect_clusters_from_file, analyze_clusters_simple, visualize_soft_spheres_with_progress

#endregion


# Setting random seed for reproducibility
Random.seed!(1234)

#region ------- Simulation Parameters -----
# -------- General system parameters --------
volume_fraction_oil = 0.3                  
box_side            = 5.0
temperature         = 1.0                   # reduced temp to limit initial speeds

# ------------- Droplet settings ------------
#n_droplets = 20               # number of droplets is now based on volume fraction and droplet size
droplet_radius     = 0.1
droplet_mass       = 1.0
energy_strength    = 1.0
cutoff             = 2.5*droplet_radius    # use a modest LJ/soft-sphere cutoff ~2.5 sigma
minimal_dist_frac  = 1                  # minimal distance between initial droplet centers as fraction of 2*R
skipping_neighbors_threshold = 0.3  # if droplet diameter is larger than this, skip neighbor list in Molly system
# -------------- Wall settings --------------
# Rough wall configuration; motion is set later by the shear profile.
enable_walls = true             # set false to disable walls and use periodic y instead
periodic_y_mode = !enable_walls
wall_mass      = 10*droplet_mass
wall_radius    = 1.0 * droplet_radius     # give walls a sensible size to avoid singular kicks
wall_energy_strength = 0.5 * energy_strength  # moderate wall interaction

# Time-dependent shear strain gamma(t). Supply any lambda you like here; gammȧ(t)
# is approximated numerically inside make_shear_profile.
shear_freq = 0.00                      # cycles per unit time; adjust as needed
gamma_amplitude = 0.0                # peak strain
gamma_phase = 0.0
gamma_fn = t -> gamma_amplitude * sin(2*pi*shear_freq*t + gamma_phase)

# ------------ Simulation settings ----------
enable_energy_logging = true     # set false to skip energy calc/logging for speed
enable_langevin = true          # thermostat toggle
gamma_langevin = 0.5             # friction coefficient for Langevin thermostat
dt = 1e-5
T = 10.0                              # CHANGE TIME HERE
nsteps = Int(round(T/dt))
# Desired output fps; we will subsample to approximate this while keeping duration = T
desired_fps = 60
realistic_time = false          # if true, the video plays in real time and desired fps is disregarded; if false, output fps matches desired fps 
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


droplets = [Molly.Atom(mass=droplet_mass, σ=2*droplet_radius, ϵ=energy_strength, atom_type=1) for _ in 1:n_droplets]
# Boundary for placement; Molly's RectangularBoundary here is periodic flag-less. Non-periodic y is handled by walls.
boundary = Molly.RectangularBoundary(SVector{2,Float64}(box_side, box_side))
# Place atoms away from walls: y in [2r, box_side-2r], x in [2r, box_side-2r]
function place_away_from_walls(n, box_side, r, min_dist)
    coords = SVector{2,Float64}[]
    max_attempts = 100000
    attempts = 0
    while length(coords) < n && attempts < max_attempts
        attempts += 1
        x = 2r + rand()*(box_side - 4r)
        y = 2r + rand()*(box_side - 4r)
        p = SVector{2,Float64}(x, y)
        if all(norm(p - q) > min_dist for q in coords)
            push!(coords, p)
        end
    end
    if length(coords) < n
        error("Could not place atoms without overlap after $max_attempts attempts")
    end
    return coords
end
init_coord = place_away_from_walls(n_droplets, box_side, droplet_radius, 2.0*droplet_radius*minimal_dist_frac)
n_bulk   = n_droplets
v0_bulk = [random_velocity(droplet_mass, temperature; dims=2) for _ in 1:n_droplets]

pairwise_inter = (Molly.SoftSphere(use_neighbors=false),)
#endregion

#region -------- Initialize wall interactions --------
if enable_walls
    wall_up = SoftsphereWall(
        box_side,      # Wall at y=box_side
        wall_radius,          
        wall_energy_strength, 
        max(wall_radius * 3, cutoff)     # Cutoff
    )
    wall_down = SoftsphereWall(
        0.0,      # Wall at y=0
        wall_radius,          
        wall_energy_strength, 
        max(wall_radius * 3, cutoff)     # Cutoff
    )
    
    walls = (wall_up, wall_down,)

else
    walls = ()
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
#region -------- Shear profile setup --------
shear_profile = make_shear_profile(gamma_fn = gamma_fn)

# Precompute a reference shear rate over the simulation span to normalize drag strength.
rate_samples = [abs(shear_profile.gamma_rate(t)) for t in range(0, stop=T, length=101)]
gamma_rate_ref = maximum(rate_samples)
gamma_rate_ref = gamma_rate_ref <= eps(Float64) ? 1.0 : gamma_rate_ref

#endregion
#region ------- Build the system and run simulation -----

sys = build_soft_emulsion_system_with_artificial_walls(
    init_coord,
    droplets,
    box_side,
    cutoff,
    v0_bulk,
    pairwise_inter,
    walls,
    nsteps;
    n_threashold = skipping_neighbors_threshold,
)
#endregion
#region -------- Energy and temperature logging setup --------
energy_history = Float64[]
time_history = Float64[]
temperature_history = Float64[]
temperature_time = Float64[]
if enable_energy_logging
    # initial energies (t=0)
    E0 = total_energy(sys)
    push!(energy_history, E0)
    push!(time_history, 0.0)
end
temp0 = current_temperature(sys, n_bulk)  # proxy temperature (k_B=1)
push!(temperature_history, temp0)
push!(temperature_time, 0.0)
#endregion
#region -------- Setup pre_step! and post_step! functions --------

# disabled pre wall as the wall is just a force right now 
# pre_wall! = isempty(wall_indices) ? nothing : (step_idx -> begin
#     t = step_idx == 0 ? 0.0 : (step_idx - 1) * dt
#     enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
# end)

post_wall! = enable_walls ? nothing :
   (step_idx -> begin
    t = step_idx * dt
    #enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
    #apply_wall_drag!(sys, n_bulk, wall_y_top_ref, wall_y_bot_ref, shear_profile, t, wall_gap, box_side,
    #                  dt=dt, rate_ref=gamma_rate_ref)
    #confine_bulk_y!(sys, n_bulk, wall_y_bot_ref + cutoff, wall_y_top_ref - cutoff)
    

    if enable_energy_logging && (step_idx % save_every == 0 || step_idx == nsteps)
        Etot = total_energy(sys)
        push!(energy_history, Etot)
        push!(time_history, step_idx * dt)
    end
    if step_idx % save_every == 0 || step_idx == nsteps
        temp_inst = current_temperature(sys, n_bulk)
        push!(temperature_history, temp_inst)
        push!(temperature_time, step_idx * dt)
    end
    end)

# Simple hard confinement in y to keep particles inside [0, box_side]
function confine_y!(sys::Molly.System, y_min::Float64, y_max::Float64)
    @inbounds for i in eachindex(sys.coords)
        p = sys.coords[i]; v = sys.velocities[i]
        if p[2] < y_min
            sys.coords[i] = SVector(p[1], y_min + (y_min - p[2]))
            sys.velocities[i] = SVector(v[1], -v[2])
        elseif p[2] > y_max
            sys.coords[i] = SVector(p[1], y_max - (p[2] - y_max))
            sys.velocities[i] = SVector(v[1], -v[2])
        end
    end
end

# Periodic-y hook (used when walls are disabled).
post_periodic! = periodic_y_mode ? (step_idx -> begin
    apply_periodic_y!(sys, n_bulk, box_side)
    
    if enable_energy_logging && (step_idx % save_every == 0 || step_idx == nsteps)
        Etot = total_energy(sys)
        push!(energy_history, Etot)
        push!(time_history, step_idx * dt)
    end
    if step_idx % save_every == 0 || step_idx == nsteps
        temp_inst = current_temperature(sys, n_bulk)
        push!(temperature_history, temp_inst)
        push!(temperature_time, step_idx * dt)
    end
end) : nothing

#pre_hook  = periodic_y_mode ? nothing       : pre_wall!
post_hook = periodic_y_mode ? post_periodic! : post_wall!
# Always confine after each step to keep beads in the box.
post_confine! = (step_idx -> confine_y!(sys, wall_radius, box_side - wall_radius))
#endregion
#region -------- Run the simulation and collect coordinates -------- 
simulator = enable_langevin ?
    Langevin(dt = dt, temperature = temperature, friction = gamma_langevin) :
    VelocityVerlet(dt = dt)

# Compute save_every from desired_fps; if not enough steps to reach target fps, save every frame
# frames_target = desired_fps * T; save_every ~ nsteps / frames_target
frames_target = max(1, Int(round(desired_fps * T)))
save_every = max(1, Int(floor(nsteps / frames_target)))

sim_time = @elapsed begin
    coords_history = run_and_collect!(
        sys,
        simulator,
        nsteps;
        save_every=save_every,
        post_step! = (s -> begin
            post_hook === nothing || post_hook(s)
            post_confine!(s)
        end),
    )
end
println("Simulation completed in $sim_time seconds.")
if isempty(coords_history) || isempty(coords_history[1])
    error("No particle coordinates recorded; check system initialization.")
end
#endregion
#region ------- Visualize the results -----
if realistic_time
    frames = length(coords_history)
    fps = max(1, Int(round(frames / T)))
else
    fps = desired_fps 
end
frames = length(coords_history)
visualize_soft_spheres_with__artificial_wall(
    coords_history,
    boundary,
    joinpath(outdir, "sim_soft_spheres_aw_vf$(round(volume_fraction_oil, digits=2))_d$(round(droplet_radius, digits=2)).mp4");
    box_side=box_side,
    framerate=fps,
    droplet_radius=droplet_radius,
    n_droplets=n_droplets,
)

# Plot energy vs time
if enable_energy_logging && !isempty(energy_history)
    figE = GLMakie.Figure(size=(600,400))
    axE = GLMakie.Axis(figE[1,1]; xlabel="time", ylabel="total energy")
    GLMakie.lines!(axE, time_history, energy_history, color=:black)
    energy_outfile = joinpath(outdir, "energy_vs_time.png")
    GLMakie.save(energy_outfile, figE)
    println("Saved energy plot to ", energy_outfile)
end

# Plot temperature vs time
if !isempty(temperature_history)
    figT = GLMakie.Figure(size=(600,400))
    axT = GLMakie.Axis(figT[1,1]; xlabel="time", ylabel="kinetic temperature")
    GLMakie.lines!(axT, temperature_time, temperature_history, color=:red)
    temp_outfile = joinpath(outdir, "temperature_vs_time.png")
    GLMakie.save(temp_outfile, figT)
    println("Saved temperature plot to ", temp_outfile)
end
#endregion
