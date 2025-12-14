using Random
using StaticArrays
using Molly
using GLMakie  # Make sure you have this loaded
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity, visualize, InteractionList3Atoms, force
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

include(joinpath(@__DIR__, "functions.jl"))
include(joinpath(@__DIR__, "wrapping_and_confining_functions.jl"))  # very basic, save to include first
include(joinpath(@__DIR__, "forces_functions.jl"))                  # Custom pairwise interaction defined here - has to be included before system building
include(joinpath(@__DIR__, "beads_placing_functions.jl"))           # has to be included before system building
include(joinpath(@__DIR__, "wall_and_shear_functions.jl"))          # has to be included before system building
include(joinpath(@__DIR__, "hull_and_area_functions.jl"))           # has to be included before system building
include(joinpath(@__DIR__, "system_and_simulation_functions.jl"))
include(joinpath(@__DIR__, "plotting_functions.jl"))
include(joinpath(@__DIR__, "cluster_functions.jl"))

#Function that generates a random vector of type VecType within a specified range
function random_vec(::Type{VecType},range) where VecType 
    dim = length(VecType)
    T = eltype(VecType)
    p = VecType(
        range[begin] + rand(T)*(range[end]-range[begin]) for _ in 1:dim
    )
    return p
end

# defining repulsion parameters, much more stable behaviour with bigger a_oil_water and smaller a_water_water/a_oil_oil
const a_water_water = 15.0 # 5 <- better
const a_oil_oil     = 20 # 5 <- better
const a_oil_water   = 90.0 # 100 <- better

const a_wall_wall   = 15.0
const a_wall_water  = 40.0
const a_wall_oil    = 40.0

const a_surface_oil = 25.0;
const a_surface_water = 80;
const a_surface_surface = 100;
const a_surface_wall = 80;

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
volume_fraction_oil = 0.7           # VF
box_side            = 5.0
cutoff              = 5/32.2*1.5
density_number      = 200.0         # particles per unit area
temperature = 273                   # temp in kelvin

# ------------- Droplet settings ------------
n_droplets = 100
surface_concentration = 0.5         # fraction of particles on droplet surface
k_area = 10.0                       # Area constraint stiffness (Tune this value)
bond_stiffness_main = 200.0         # Bond stiffness for main hull bonds (i and i+1)
bond_stiffness_secondary = 200.0    # Bond stiffness for bending bonds (i and i+2)
bond_stiffness_fourth = 100.0       # Bond stiffness for bending bonds (i and i+4)

# -------------- Wall settings --------------
# Rough wall configuration; motion is set later by the shear profile.
n_per_wall     = 0
wall_y_offset  = 0.5 * cutoff
wall_roughness = 0.8 * cutoff

# Time-dependent shear strain gamma(t). Supply any lambda you like here; gammȧ(t)
# is approximated numerically inside make_shear_profile.
shear_freq = 0                      # cycles per unit time; adjust as needed
gamma_amplitude = 0                 # peak strain
gamma_phase = 0.0
gamma_fn = t -> gamma_amplitude * sin(2*pi*shear_freq*t + gamma_phase)

# ------------ Simulation settings ----------
dt = 0.001
T = 5 #CHANGE TIME HERE
nsteps = Int(round(T/dt))
# Desired output fps; we will subsample to approximate this while keeping duration = T
desired_fps = 15
# -------------------------------------------


# ------- Initialize wall particles --------
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

# ------- Initialize droplet particles and hulls --------
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

x0_oil, R_used, droplets_matrix, n_per, hull_poly, hulls_idx, hulls_vec = generate_multi_droplet_and_matrix(
    n_oil,
    n_water,
    centers,
    droplet_area,
    surface_concentration,
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
v0_bulk  = [random_vec(SVector{2,Float64},(-0.1,0.1)) for _ in 1:n_bulk]
v0_walls = [SVector{2,Float64}(0.0, 0.0) for _ in 1:n_walls]

# # Basic usage
# fig = plot_hull_points(x0_oil, hulls_idx, n_per)
# display(fig)

# # With box boundaries
# fig = plot_hull_points(x0_oil, hulls_idx, n_per; box_side=box_side)
# display(fig)

# # Only hull points, no background
# fig = plot_hull_points(x0_oil, hulls_idx, n_per; show_all_oil=false)
# display(fig)

surface_particles_idx = local_idx_global(hulls_idx, n_per)

# Check if there are any duplicate surface indices
all_surface_indices = vcat(surface_particles_idx)
@show all_surface_indices
if length(all_surface_indices) != length(unique(all_surface_indices))
    @warn "Duplicate surface particle indices detected!"
else
    println("No duplicate surface particle indices.")
end
# Plot surface particles (hulls) for verification
fig = Figure()
ax = Axis(fig[1, 1]; 
          title="Surface Particles", 
          xlabel="x", 
          ylabel="y", 
          aspect=DataAspect())
for (k, surface_idx) in enumerate(surface_particles_idx)
    surface_points = [x0_oil[i] for i in surface_idx]
    scatter!(ax, [p[1] for p in surface_points], [p[2] for p in surface_points]; 
             markersize=6, label=(k==1 ? "Surface Particles" : nothing))
end
display(fig)

# -------- Setup bonding along droplet hulls --------
# Setup harmonic bonds along the droplet surfaces (hulls)
bond_is = Int[]
bond_js = Int[]
bond_params = Vector{HarmonicBond}()

bend_is = Int[]
bend_js = Int[]
bend_params = Vector{HarmonicBond}()

bend4_is = Int[]
bend4_js = Int[]
bend4_params = Vector{HarmonicBond}()


for ring in surface_particles_idx
    length(ring) < 2 && continue
    for i in 1:(length(ring)-1)
        a = ring[i]
        b = ring[i+1]
        b2 = ring[mod1(i+2, length(ring)-1)] # next to next particle for bending
        b4 = ring[mod1(i+4, length(ring)-1)] # next to next to next particle for bending
        push!(bond_is, a)
        push!(bond_js, b)
        push!(bend_is, a)
        push!(bend_js, b2)
        push!(bend4_is, a)
        push!(bend4_js, b4)
        r0 = norm(x0_oil[b] - x0_oil[a])
        r0_bend = norm(x0_oil[b2] - x0_oil[a])
        r0_bend2 = norm(x0_oil[b4] - x0_oil[a])
        push!(bond_params, HarmonicBond(k=bond_stiffness_main, r0=r0))
        push!(bend_params, HarmonicBond(k=bond_stiffness_secondary, r0=r0_bend))
        push!(bend4_params, HarmonicBond(k=bond_stiffness_fourth, r0=r0_bend2))
    end
    # close the ring
    a = ring[end]
    b = ring[1]
    b2 = ring[2]
    b4 = ring[4]
    push!(bond_is, a)
    push!(bond_js, b)
    push!(bend_is, a)
    push!(bend_js, b2)
    push!(bend4_is, a)
    push!(bend4_js, b4)
    r0 = norm(x0_oil[b] - x0_oil[a])
    r0_bend = norm(x0_oil[b2] - x0_oil[a])
    r0_bend2 = norm(x0_oil[b4] - x0_oil[a])
    push!(bond_params, HarmonicBond(k=bond_stiffness_main, r0=r0))
    push!(bend_params, HarmonicBond(k=bond_stiffness_secondary, r0=r0_bend))
    push!(bend4_params, HarmonicBond(k=bond_stiffness_fourth, r0=r0_bend2))
end

# Create specific interactions list
hull_bonds = InteractionList2Atoms(bond_is, bond_js, bond_params)
bend_bonds = InteractionList2Atoms(bend_is, bend_js, bend_params)
bend4_bonds = InteractionList2Atoms(bend4_is, bend4_js, bend4_params)

specific_inter_lists = (hull_bonds, bend_bonds, bend4_bonds)# hull_angles) # Combine bond and angle lists

# ------- Shear profile setup --------
shear_profile = make_shear_profile(gamma_fn = gamma_fn)

# Precompute a reference shear rate over the simulation span to normalize drag strength.
rate_samples = [abs(shear_profile.gamma_rate(t)) for t in range(0, stop=T, length=101)]
gamma_rate_ref = maximum(rate_samples)
gamma_rate_ref = gamma_rate_ref <= eps(Float64) ? 1.0 : gamma_rate_ref



# ------- Build the Molly system --------
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
    bond_is,
    bond_js,
    surface_particles_idx,
    specific_inter_lists,
    nsteps;
    periodic_y=false,
)
wall_indices = collect(wall_range)

# ------- Calculate target areas for area constraint --------
# Calculate the initial (target) area for each droplet
target_areas = Vector{Float64}()
for k in 1:length(hulls_vec)
    # hulls_vec is convex_hull_coords (initial hull coordinates) from generate_multi_droplet_and_matrix
    hull_coords = hulls_vec[k]
    push!(target_areas, polygon_area(hull_coords))
end

# ------- Setup pre_step! and post_step! functions --------
pre_wall! = isempty(wall_indices) ? nothing : (step_idx -> begin
    t = step_idx == 0 ? 0.0 : (step_idx - 1) * dt
    enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
end)
post_wall! = isempty(wall_indices) ?
    (step_idx -> begin
        # No moving walls; still keep droplets intact (soft hull)
        apply_soft_hull_wall!(
            sys, droplet_ranges, box_side;
            hulls_idx = hulls_idx, dt = dt,
            k_wall = bond_stiffness_main, buffer = 0.5 * cutoff
        )
        apply_area_constraint!(
            sys, surface_particles_idx, target_areas, k_area, box_side;
            dt = dt,
        )
        
    end) :
    (step_idx -> begin
        t = step_idx * dt

        # Update wall motion and apply near-wall dynamics
        enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
        apply_wall_drag!(
            sys, n_bulk, wall_y_top_ref, wall_y_bot_ref,
            shear_profile, t, wall_gap, box_side;
            dt = dt, rate_ref = gamma_rate_ref
        )
        confine_bulk_y!(sys, n_bulk, wall_y_bot_ref + cutoff/1.5, wall_y_top_ref - cutoff/1.5)

        # Soft hull wall to keep droplets intact but deformable
        apply_soft_hull_wall!(
            sys, droplet_ranges, box_side;
            hulls_idx = hulls_idx, dt = dt,
            k_wall = bond_stiffness_main, buffer = 0.5 * cutoff/1.5
        )
        apply_area_constraint!(
            sys, surface_particles_idx, target_areas, k_area, box_side;
            dt = dt,
        )
        
    end)

# ------- Run the simulation and collect coordinates --------    
simulator = VelocityVerlet(
    dt = dt
)

# Compute save_every from desired_fps; if not enough steps to reach target fps, save every frame
# frames_target = desired_fps * T; save_every ~ nsteps / frames_target
frames_target = max(1, Int(round(desired_fps * T)))
save_every = max(1, Int(floor(nsteps / frames_target)))
if save_every <= 0
    save_every = 1
end
sim_time = @elapsed begin
    coords_history = run_and_collect!(
        sys,
        simulator,
        nsteps;
        save_every=save_every,
        pre_step! = pre_wall!,
        post_step! = post_wall!,
    )
end
println("Simulation completed in $sim_time seconds.")
if isempty(coords_history) || isempty(coords_history[1])
    error("No particle coordinates recorded; check system initialization.")
end

# ------- Visualization --------
# color each droplet differently 
droplet_colors = [RGB(rand(), rand(), rand()) for _ in 1:length(n_per)]
oil_colors = vcat([fill(droplet_colors[k], n_per[k]) for k in 1:length(n_per)]...)
water_colors = [RGB(0, 0, 1) for _ in 1:n_water] # all water particles blue
wall_colors = [RGB(0.5, 0.5, 0.5) for _ in 1:n_walls] # all wall particles gray
colors = vcat(oil_colors, water_colors, wall_colors)

# Compute framerate so that video duration equals T seconds
frames = length(coords_history)
fps = max(1, Int(round(frames / T)))

# Report effective fps and subsampling for traceability
println("Desired fps: ", desired_fps, ", save_every: ", save_every, ", frames saved: ", frames, ", effective fps: ", fps)

# Wrap x for visualization; leave y unwrapped to inspect shear profile.
side_x = sys.boundary.side_lengths[1]
wrap_frame(frame) = [SVector(mod(p[1], side_x), p[2]) for p in frame]
coords_history_wrapped = [wrap_frame(frame) for frame in coords_history]

# Ensure output dir exists
outdir = joinpath(@__DIR__, "Bouncy_droplets")
isdir(outdir) || mkpath(outdir) # Create directory if it doesn't exist

# Build filename
fname = "emulsion_harmonic_dt$(dt)_ow$(a_oil_water)_ww$(a_water_water)_so$(a_surface_oil)_ss$(a_surface_surface)_$(n_droplets)_T$(T)_vol$(volume_fraction_oil)_sc$(surface_concentration).mp4"
outfile = joinpath(outdir, fname)

visualize_with_progress(
    coords_history_wrapped,
    sys.boundary,
    outfile;
    color = colors,
    markersize = 2.0,
    framerate = fps,
    droplet_ranges = droplet_ranges,
    box_side = box_side,
    hulls_idx = hulls_idx,
    droplet_colors = droplet_colors,
)
println("Visualization saved to ", outfile)

# --------------Cluster analysis on snapshot --------------
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
    cutoff   = cutoff/1.5
)

data = readdlm(snap_file)
coords = [(data[i,1], data[i,2]) for i in axes(data, 1)]

n_clusters, beads_per_cluster, avg_radius =
    analyze_clusters_simple(clusters, cutoff/1.5)

println("Clusters: ", n_clusters,
        " | beads per cluster: ", beads_per_cluster)

println("Average effective radius: ", avg_radius)
