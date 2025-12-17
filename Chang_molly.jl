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
include(joinpath(@__DIR__, "hull_and_area_functions.jl"))  
# apply_soft_hull_wall!, local_idx_global, apply_inter_droplet_repulsion!, apply_area_constraint!
include(joinpath(@__DIR__, "system_and_simulation_functions.jl"))
# MyCoordinatesLogger, run_and_collect!, build_emulsion_system
include(joinpath(@__DIR__, "plotting_functions.jl"))
# visualize_with_progress, plot_hull_points
include(joinpath(@__DIR__, "cluster_functions.jl"))
# detect_clusters_from_file, analyze_clusters_simple

#endregion

#region -------- Define system and simulation parameters --------
# -------- Interaction parameters --------
# defining repulsion parameters, much more stable behaviour with bigger a_oil_water and smaller a_water_water/a_oil_oil
const a_water_water = 15.0 # 5 <- better
const a_oil_oil     = 15 # 5 <- better
const a_oil_water   = 40.0 # 100 <- better

const a_wall_wall   = 15.0
const a_wall_water  = 40.0
const a_wall_oil    = 40.0

const a_surface_oil = 25.0;
const a_surface_water = 25.0;
const a_surface_surface = 50;
const a_surface_wall = 40;

# Map bead types to interaction strengths
const A_MAP = Dict{Tuple{Int,Int},Float64}(
    (1,1)     => a_oil_oil,
    (2,2) => a_water_water,
    (3,3)   => a_wall_wall,
    (1,2)   => a_oil_water,
    (2,1)   => a_oil_water,
    (1,3)    => a_wall_oil,
    (3,1)    => a_wall_oil,
    (2,3)  => a_wall_water,
    (3,2)  => a_wall_water,
    (4,4)     => a_surface_surface,
    (4,1)     => a_surface_oil,
    (1,4)     => a_surface_oil,
    (4,2)     => a_surface_water,
    (2,4)     => a_surface_water,
    (4,3)     => a_surface_wall,
    (3,4)     => a_surface_wall,
)

# Setting random seed for reproducibility
Random.seed!(1234)

# -------- General system parameters --------
volume_fraction_oil = 0.9        # VF
box_side            = 5.0
cutoff              = 5/32.2
density_number      = 100.0         # particles per unit area
temperature = 273                   # temp in kelvin

# ------------- Droplet settings ------------
n_droplets = 20
enable_hulls = true                 # toggle droplet hulls (this includes bonding along hulls)
surface_concentration = 0.1         # fraction of particles on droplet surface
bond_stiffness_main = 100.0         # Bond stiffness for main hull bonds (i and i+1)
bond_stiffness_god = 50.0           # Bond stiffness for center bead bonds
bond_stiffness_secondary = 100.0    # Bond stiffness for bending bonds (i and i+2)
bond_stiffness_fourth = 100.0       # Bond stiffness for bending bonds (i and i+4)
droplets_repulsion = 0.000          # Repulsion strength between droplets to avoid interpenetration
light_spring_stiffness = 50.0       # Spring stiffness for light springs connecting hull beads to droplet center
enable_God_bead = true              # Toggle God bead bonds

# -------------- Wall settings --------------
# Rough wall configuration; motion is set later by the shear profile.
enable_walls = true                 # set false to disable walls and use periodic y instead
periodic_y_mode = !enable_walls
n_per_wall     = 600
wall_y_offset  = 0.5 * cutoff
wall_roughness = 0.8 * cutoff

# Time-dependent shear strain gamma(t). Supply any lambda you like here; gammȧ(t)
# is approximated numerically inside make_shear_profile.
shear_freq = 0.01                      # cycles per unit time; adjust as needed
gamma_amplitude = 0.01                 # peak strain
gamma_phase = 0.0
gamma_fn = t -> gamma_amplitude * sin(2*pi*shear_freq*t + gamma_phase)

# ------------ Simulation settings ----------
enable_energy_logging = true     # set false to skip energy calc/logging for speed
enable_langevin = true          # thermostat toggle
gamma_langevin = 0.5             # friction coefficient for Langevin thermostat
dt = 0.0001
T = 10000*dt                           #CHANGE TIME HERE
nsteps = Int(round(T/dt))
# Desired output fps; we will subsample to approximate this while keeping duration = T
desired_fps = 100
realistic_time = false          # if true, the video plays in real time and desired fps is disregarded; if false, output fps matches desired fps 
output_folder = "Bouncy_droplets_with_walls" # folder to save visualization output (mp4, temp plot, energy plot)
# -------------------------------------------
#endregion
#region -------- Initialize wall particles --------
if enable_walls
    x_top, x_bot = make_rough_wall_particles(
        SVector{2,Float64},
        box_side,
        n_per_wall;
        y_offset = wall_y_offset,
        y_amp = wall_roughness,
    )
    walls = vcat(x_top, x_bot)
    wall_bases = [SVector(w[1], w[2]) for w in walls]
    wall_sides = vcat(fill(:top, length(x_top)), fill(:bottom, length(x_bot)))
    # Absolute coordinates with bottom at y=0.
    wall_y_bot_ref = wall_y_offset
    wall_y_top_ref = box_side - wall_y_offset
    wall_gap = wall_y_top_ref - wall_y_bot_ref
else
    walls = SVector{2,Float64}[]
    wall_bases = SVector{2,Float64}[]
    wall_sides = Symbol[]
    wall_y_bot_ref = 0.0
    wall_y_top_ref = box_side
    wall_gap = wall_y_top_ref - wall_y_bot_ref
end
#endregion
#region -------- Initialize droplet particles and hulls --------
volume_oil   = volume_fraction_oil * box_side^2
volume_water = (1.0 - volume_fraction_oil) * box_side^2
n_oil::Int   = ceil(volume_oil * density_number)
n_water::Int = ceil((box_side^2 - volume_oil) * density_number)
droplet_area = volume_oil / n_droplets          # area of ONE droplet
R_droplet    = sqrt(droplet_area / pi)

centers = generate_droplet_centers(
    n_droplets,
    box_side,
    R_droplet;
    margin_factor = 1.0,
)


x0_oil, R_used, droplets_matrix, n_per, hull_poly, hulls_idx, hulls_vec, light_spring_interactions = generate_multi_droplet_and_matrix(
    n_oil,
    n_water,
    centers,
    droplet_area,
    surface_concentration,
    light_spring_stiffness ,
)

x0_water = generate_outside_droplets(
    n_water,
    centers,
    R_used,
    box_side,
)

droplet_ranges = let r = Vector{UnitRange{Int}}(); offset = 0
    for np in n_per
        push!(r, (offset+1):(offset+np))
        offset += np
    end
    r
end

x0_emulsion = vcat(x0_oil, x0_water)

n_bulk   = length(x0_emulsion)   # = n_oil + n_water
n_walls  = length(walls)
v0_bulk  = [random_velocity(1.0, temperature; dims=2) for _ in 1:n_bulk]
v0_walls = [SVector{2,Float64}(0.0, 0.0) for _ in 1:n_walls]

if enable_hulls
    surface_particles_idx = local_idx_global(hulls_idx, n_per)
    # Check if there are any duplicate surface indices
    all_surface_indices = vcat(surface_particles_idx)

    if length(all_surface_indices) != length(unique(all_surface_indices))
        @warn "Duplicate surface particle indices detected!"
    else
        println("No duplicate surface particle indices.")
    end

#endregion
#region -------- Setup bonding along droplet hulls --------

    bond_is = Int[]
    bond_js = Int[]
    bond_params = Vector{HarmonicBond}()

    bend_is = Int[]
    bend_js = Int[]
    bend_params = Vector{HarmonicBond}()

    bend4_is = Int[]
    bend4_js = Int[]
    bend4_params = Vector{HarmonicBond}()

    if enable_God_bead 
        God_bond_is = Int[]
        God_bond_js = Int[]
        God_bond_params = Vector{HarmonicBond}()
    end
        
    for (k, droplet_range) in enumerate(droplet_ranges)
        center_idx = droplet_range[1]  # God bead index is the first index inside the droplet
        surface_particles = surface_particles_idx[k]
        for i in 1:(length(surface_particles)-1)  # Exclude last which is a duplicate of the first

            if enable_God_bead
                r0 = norm(x0_oil[surface_particles[i]] - x0_oil[center_idx])
                push!(God_bond_is, center_idx)
                push!(God_bond_js, surface_particles[i])
                push!(God_bond_params, HarmonicBond(k=bond_stiffness_god, r0=r0))
            end

            r0_neighor = norm(x0_oil[surface_particles[i]] - x0_oil[surface_particles[i+1]])
            push!(bond_is, surface_particles[i])
            push!(bond_js, surface_particles[i+1])
            push!(bond_params, HarmonicBond(k=bond_stiffness_main, r0=r0_neighor))

            ro_neighbor2 = norm(x0_oil[surface_particles[i]] - x0_oil[surface_particles[mod1(i+2, length(surface_particles)-1)]])
            push!(bend_is, surface_particles[i])
            push!(bend_js, surface_particles[mod1(i+2, length(surface_particles)-1)])
            push!(bend_params, HarmonicBond(k=bond_stiffness_secondary, r0=ro_neighbor2))

            ro_neighbor4 = norm(x0_oil[surface_particles[i]] - x0_oil[surface_particles[mod1(i+4, length(surface_particles)-1)]])
            push!(bend4_is, surface_particles[i])
            push!(bend4_js, surface_particles[mod1(i+4, length(surface_particles)-1)])
            push!(bend4_params, HarmonicBond(k=bond_stiffness_fourth, r0=ro_neighbor4))
        end
    end
    standard = InteractionList2Atoms(bond_is, bond_js, bond_params)
    bend2 = InteractionList2Atoms(bend_is, bend_js, bend_params)
    bend4 = InteractionList2Atoms(bend4_is, bend4_js, bend4_params)
    if enable_God_bead
        god_interactions = InteractionList2Atoms(God_bond_is, God_bond_js, God_bond_params)
        specific_inter_lists = (standard, bend2, bend4, god_interactions, light_spring_interactions)# hull_angles) # Combine bond and angle lists
    else
        specific_inter_lists = (standard, bend2, bend4)# hull_angles) # Combine bond and angle lists
    end
else 
    specific_inter_lists
end



#endregion
#region -------- Shear profile setup --------
shear_profile = make_shear_profile(gamma_fn = gamma_fn)

# Precompute a reference shear rate over the simulation span to normalize drag strength.
rate_samples = [abs(shear_profile.gamma_rate(t)) for t in range(0, stop=T, length=101)]
gamma_rate_ref = maximum(rate_samples)
gamma_rate_ref = gamma_rate_ref <= eps(Float64) ? 1.0 : gamma_rate_ref

#endregion
#region -------- Build the Molly system --------
sys, wall_range = build_emulsion_system(
    x0_emulsion,
    walls,
    n_oil,
    n_water,
    box_side,
    cutoff,
    v0_bulk,
    v0_walls,
    all_surface_indices,
    surface_particles_idx,
    specific_inter_lists,
    nsteps;
    periodic_y=periodic_y_mode,
)
wall_indices = collect(wall_range)
#endregion
#region -------- Energy and temperature logging setup --------
energy_history = Float64[]
time_history = Float64[]
temperature_history = Float64[]
temperature_time = Float64[]
if enable_energy_logging
    # initial energies (t=0)
    E0, _, _ = emulsion_energy(sys, cutoff, box_side; n_bulk_only=n_bulk)
    push!(energy_history, E0)
    push!(time_history, 0.0)
end
temp0 = current_temperature(sys, n_bulk)  # proxy temperature (k_B=1)
push!(temperature_history, temp0)
push!(temperature_time, 0.0)
#endregion
#region -------- Setup pre_step! and post_step! functions --------
pre_wall! = isempty(wall_indices) ? nothing : (step_idx -> begin
    t = step_idx == 0 ? 0.0 : (step_idx - 1) * dt
    enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
end)

post_wall! = isempty(wall_indices) ?
    (step_idx -> begin 
        if enable_hulls
            # No moving walls; still keep droplets intact (soft hull)
            apply_soft_hull_wall!(
                sys, droplet_ranges, box_side;
                hulls_idx = hulls_idx, dt = dt,
                k_wall = bond_stiffness_main, buffer = 0.5 * cutoff
            )
        end
    end) :
   (step_idx -> begin
    t = step_idx * dt
    enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
    apply_wall_drag!(sys, n_bulk, wall_y_top_ref, wall_y_bot_ref, shear_profile, t, wall_gap, box_side,
                     dt=dt, rate_ref=gamma_rate_ref)
    confine_bulk_y!(sys, n_bulk, wall_y_bot_ref + cutoff, wall_y_top_ref - cutoff)
    if enable_hulls
        # Soft hull wall to keep droplets intact but deformable
        apply_soft_hull_wall!(
            sys, droplet_ranges, box_side;
            hulls_idx = hulls_idx, dt = dt,
            k_wall = bond_stiffness_main, buffer = 0.5 * cutoff/1.5
        )
    end

    if enable_energy_logging && (step_idx % save_every == 0 || step_idx == nsteps)
        Etot, _, _ = emulsion_energy(sys, cutoff, box_side; n_bulk_only=n_bulk)
        push!(energy_history, Etot)
        push!(time_history, step_idx * dt)
    end
    if step_idx % save_every == 0 || step_idx == nsteps
        temp_inst = current_temperature(sys, n_bulk)
        push!(temperature_history, temp_inst)
        push!(temperature_time, step_idx * dt)
    end
    end)

# Periodic-y hook (used when walls are disabled).
post_periodic! = periodic_y_mode ? (step_idx -> begin
    apply_periodic_y!(sys, n_bulk, box_side)
    if enable_hulls
        # No moving walls; still keep droplets intact (soft hull)
        apply_soft_hull_wall!(
            sys, droplet_ranges, box_side;
            hulls_idx = hulls_idx, dt = dt,
            k_wall = bond_stiffness_main, buffer = 0.5 * cutoff
    )
    end
    if enable_energy_logging && (step_idx % save_every == 0 || step_idx == nsteps)
        Etot, _, _ = emulsion_energy(sys, cutoff, box_side; n_bulk_only=n_bulk)
        push!(energy_history, Etot)
        push!(time_history, step_idx * dt)
    end
    if step_idx % save_every == 0 || step_idx == nsteps
        temp_inst = current_temperature(sys, n_bulk)
        push!(temperature_history, temp_inst)
        push!(temperature_time, step_idx * dt)
    end
end) : nothing

pre_hook  = periodic_y_mode ? nothing       : pre_wall!
post_hook = periodic_y_mode ? post_periodic! : post_wall!
#endregion
#region -------- Run the simulation and collect coordinates -------- 
simulator = enable_langevin ?
    Langevin(dt = dt, temperature = temperature, friction = gamma_langevin) :
    VelocityVerlet(dt = dt)

# Compute save_every from desired_fps; if not enough steps to reach target fps, save every frame
# frames_target = desired_fps * T; save_every ~ nsteps / frames_target
frames_target = max(1, Int(round(desired_fps * T)))
save_every = 1 # max(1, Int(floor(nsteps / frames_target)))
if save_every <= 0
    save_every = 1
end

sim_time = @elapsed begin
    coords_history = run_and_collect!(
        sys,
        simulator,
        nsteps;
        save_every=save_every,
        pre_step! = pre_hook,
        post_step! = post_hook,
    )
end
println("Simulation completed in $sim_time seconds.")
if isempty(coords_history) || isempty(coords_history[1])
    error("No particle coordinates recorded; check system initialization.")
end
#endregion
#region -------- Visualization --------
# color each droplet differently 
droplet_colors = [RGB(rand(), rand(), rand()) for _ in 1:length(n_per)]
oil_colors = vcat([fill(droplet_colors[k], n_per[k]) for k in 1:length(n_per)]...)
water_colors = [RGB(0, 0, 1) for _ in 1:n_water] # all water particles blue
wall_colors = [RGB(0.5, 0.5, 0.5) for _ in 1:n_walls] # all wall particles gray
colors = vcat(oil_colors, water_colors, wall_colors)

# Compute framerate so that video duration equals T seconds
if realistic_time
    frames = length(coords_history)
    fps = max(1, Int(round(frames / T)))
else
    fps = desired_fps 
end

# Report effective fps and subsampling for traceability
println("Desired fps: ", desired_fps, ", save_every: ", save_every, ", frames saved: ", frames, ", effective fps: ", fps)

# Wrap x (and y if periodic) for visualization.
side_x = sys.boundary.side_lengths[1]
wrap_frame(frame) = periodic_y_mode ?
    [SVector(mod(p[1], side_x), mod(p[2], box_side)) for p in frame] :
    [SVector(mod(p[1], side_x), p[2]) for p in frame]
coords_history_wrapped = [wrap_frame(frame) for frame in coords_history]

# Ensure output dir exists
outdir = joinpath(@__DIR__, output_folder)
isdir(outdir) || mkpath(outdir) # Create directory if it doesn't exist

# Build filename
fname = "wall_emulsion_harmonic_dt$(dt)_ow$(a_oil_water)_ww$(a_water_water)_so$(a_surface_oil)_ss$(a_surface_surface)_$(n_droplets)_T$(T)_vol$(volume_fraction_oil)_sc$(surface_concentration).mp4"
outfile = joinpath(outdir, fname)

if enable_hulls
    visualize_with_progress(
        coords_history_wrapped,
        sys.boundary,
        outfile;
        color = colors,
        markersize = 1.0,
        framerate = fps,
        droplet_ranges = droplet_ranges,
        box_side = box_side,
        hulls_idx = hulls_idx,
        droplet_colors = droplet_colors,
    )
else
    visualize_with_progress(
        coords_history_wrapped,
        sys.boundary,
        outfile;
        color = colors,
        markersize = 1.0,
        framerate = fps,
        box_side = box_side,
        droplet_colors = droplet_colors,
    )
end
println("Visualization saved to ", outfile)

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
#region -------- Cluster analysis on snapshot --------
# -----------------------
# Take snapshot at a chosen physical time
# -----------------------

# Time between saved frames
dt_frame = save_every * dt

# Choose snapshot time (physical time)
t_snap = 4.0   # seconds
k_snap = round(Int, t_snap / dt_frame) + 1
k_snap = clamp(k_snap, 1, length(coords_history_wrapped))

# Extract snapshot
frame_snap  = coords_history_wrapped[k_snap]
oil_snapshot = frame_snap[1:n_oil]

println("Snapshot taken at t = ",
        (k_snap - 1) * dt_frame,
        " with ",
        n_oil,
        " oil beads.")

# -----------------------
# Write snapshot to file
# -----------------------

snap_dir = joinpath(@__DIR__, "snapshots")
isdir(snap_dir) || mkpath(snap_dir)

snap_name = "oil_snapshot.txt"
snap_file = joinpath(snap_dir, snap_name)

data = hcat(
    [p[1] for p in oil_snapshot],
    [p[2] for p in oil_snapshot]
)

writedlm(snap_file, data)
println("Oil snapshot written to ", snap_file)

# -----------------------
# Cluster detection
# -----------------------

wrap_x(dx, L) = dx - L * round(dx / L)



clusters = detect_clusters_from_file(
    snap_file;
    box_side = box_side,
    cutoff   = cutoff
)

data = readdlm(snap_file)
coords = [(data[i,1], data[i,2]) for i in axes(data, 1)]

n_clusters, beads_per_cluster, avg_radius =
    analyze_clusters_simple(clusters, cutoff)

println("Clusters: ", n_clusters,
        " | beads per cluster: ", beads_per_cluster)

println("Average effective radius: ", avg_radius)
#endregion