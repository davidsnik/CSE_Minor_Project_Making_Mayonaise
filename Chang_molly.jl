using Random
using StaticArrays
using Molly
using GLMakie  # Make sure you have this loaded
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity, visualize
using Unitful
using DelimitedFiles
using Statistics
using LinearAlgebra: norm
# Optional progress bars
const HAS_PROGRESSMETER = Ref(false)
try
    @eval import ProgressMeter
    HAS_PROGRESSMETER[] = true
catch
    @warn "ProgressMeter.jl not available; progress bars will be disabled."
end

include(joinpath(@__DIR__, "functions.jl"))
# Functions are now available: generate_droplet_centers, generate_oil_particles_in_droplets,
# generate_water_particles, make_rough_wall_particles

#Function that generates a random vector of type VecType within a specified range
function random_vec(::Type{VecType},range) where VecType 
    dim = length(VecType)
    T = eltype(VecType)
    p = VecType(
        range[begin] + rand(T)*(range[end]-range[begin]) for _ in 1:dim
    )
    return p
end

# Custom pairwise interaction with cutoff
struct EmulsionInter{T} <: PairwiseInteraction
    cutoff::T
end
Molly.use_neighbors(::EmulsionInter) = true # use neighbor lists; x-wrapping handled in force

# Pairwise scalar force law (depends only on scalar distance between particles "r")
function Molly.pairwise_force(inter::EmulsionInter, r, params)
    a_ij, repulsive = params
    if r > inter.cutoff
        return 0.0
    # huge repulsive force is temporarily disabled
    # elseif repulsive && r < 1e-12
    #     return 3000 * (1 - r/inter.cutoff)
    else
        return a_ij * (1 - r/inter.cutoff)
    end
end

# Vector force along minimum distance vector between particles i and j
function Molly.force(inter::EmulsionInter,
                     vec_ij,
                     atom_i::Molly.Atom,
                     atom_j::Molly.Atom,
                     force_units,
                     special,
                     coord_i,
                     coord_j,
                     boundary::Molly.RectangularBoundary,
                     velocity_i,
                     velocity_j,
                     step_n)

    # Recompute displacement to wrap only in x; do not wrap in y to avoid asymmetric image artifacts.
    dx = wrap_x(coord_j[1] - coord_i[1], boundary.side_lengths[1])
    dy = coord_j[2] - coord_i[2]
    disp = SVector(dx, dy)
    r = norm(disp)
    # Guard against r = 0 to avoid division by zero
    if r <= eps(eltype(disp))
        return zero(disp)
    end

    # Choose indexes corresponding to atom types 1-> oil, 2-> water 3-> wall
    ti = atom_i.atom_type  
    tj = atom_j.atom_type
    a_ij = get(A_MAP, (ti, tj), 0.0) # assign repulsion coefficient based on the atom types

    repulsive = a_ij == a_oil_oil # specify if there should be strong surfactant like repulsion betwen two particles 
    params = (a_ij, repulsive)

    fmag = Molly.pairwise_force(inter, r, params) # get scalar force magnitude
    return fmag * disp / r 
end

# Custom logger to record coordinates without units (taken from Molly source code)
function MyCoordinatesLogger(T, n_steps::Integer; dims::Integer=3)
    return Molly.GeneralObservableLogger(
        Molly.coordinates_wrapper,
        Array{SArray{Tuple{dims}, T, 1, dims}, 1},
        n_steps,
    )
end
MyCoordinatesLogger(n_steps::Integer; dims::Integer=3) = MyCoordinatesLogger(Float64, n_steps; dims=dims)

# Fallback manual trajectory collector with subsampling by save_every
# Saves the initial frame, then every save_every steps, and always the final frame.
function run_and_collect!(sys::Molly.System, simulator, nsteps::Integer;
                          save_every::Integer=1,
                          pre_step!::Union{Nothing,Function}=nothing,
                          post_step!::Union{Nothing,Function}=nothing)
    save_every = max(1, save_every)
    hist = Vector{Vector{SVector{2,Float64}}}()

    if pre_step! !== nothing
        pre_step!(0)
    end
    push!(hist, copy(sys.coords))

    # Progress bar for simulation
    local p = nothing
    if HAS_PROGRESSMETER[]
        p = ProgressMeter.Progress(nsteps; desc="Simulating", dt=0.2)
    end

    for s in 1:nsteps
        if pre_step! !== nothing
            pre_step!(s)
        end
        simulate!(sys, simulator, 1)
        if post_step! !== nothing
            post_step!(s)
        end
        if s % save_every == 0 || s == nsteps
            push!(hist, copy(sys.coords))
        end
        if p !== nothing
            ProgressMeter.next!(p)
        end
    end
    return hist
end

function build_emulsion_system(x0_bulk::Vector{SVector{2,Float64}},
                               x0_walls::Vector{SVector{2,Float64}},
                               n_oil::Int, n_water::Int,
                               box_side::Float64, cutoff::Float64,
                               bulk_velocities,
                               wall_velocities,
                               nsteps::Integer; periodic_y::Bool=false)

    n_bulk_local = n_oil + n_water
    n_walls = length(x0_walls)
    n_total = n_bulk_local + n_walls
    wall_range = n_walls == 0 ? (0:-1) : ((n_bulk_local+1):n_total)

    # Molly.RectangularBoundary does not accept per-dimension periodic flags; use the
    # standard constructor (periodic in all dimensions) and handle x wrapping manually
    # for wall motion via wrap_x.
    # Periodic in x; standard span in y (we ignore y wrapping in force calculation).
    y_span = box_side
    boundary = Molly.RectangularBoundary(SVector{2,Float64}(box_side, y_span))

    atoms = Vector{Molly.Atom}(undef, n_total)
    for i in 1:n_oil
        atoms[i] = Molly.Atom(mass=1.0, atom_type=1)
    end
    for i in (n_oil+1):(n_oil+n_water)
        atoms[i] = Molly.Atom(mass=1.0, atom_type=2)
    end
    for i in (n_bulk_local+1):n_total
        atoms[i] = Molly.Atom(mass=1.0, atom_type=3)
    end

    coords_all = Vector{SVector{2,Float64}}(undef, n_total)
    coords_all[1:n_bulk_local] = x0_bulk
    if n_walls > 0
        coords_all[wall_range] = x0_walls
    end

    velocities = Vector{SVector{2,Float64}}(undef, n_total)
    velocities[1:n_bulk_local] = bulk_velocities
    if n_walls > 0
        wall_velocities = isempty(wall_velocities) ? [SVector{2,Float64}(0.0, 0.0) for _ in 1:n_walls] : wall_velocities
        velocities[wall_range] = wall_velocities
    end

    eligible = trues(n_total, n_total)
    @inbounds for i in 1:n_total
        eligible[i,i] = false
    end
    if n_walls > 0
        eligible[wall_range, wall_range] .= false  # keep rough wall beads rigid relative to each other
    end

    cellListMap_finder = Molly.CellListMapNeighborFinder(
        eligible=eligible,
        dist_cutoff=cutoff,
        x0=coords_all,
        unit_cell = boundary,
        n_steps = 5, # update neighbors more often so moving walls stay accurate
        dims = 2,
    )

    # Build Molly system that will be solved by a simulator
    sys = Molly.System(
        atoms = atoms,
        coords = coords_all,
        boundary = boundary,
        velocities = velocities,
        pairwise_inters = (MyPairwiseInter=EmulsionInter(cutoff),),
        neighbor_finder = cellListMap_finder,
        loggers = (coords=MyCoordinatesLogger(nsteps, dims=2),),
        energy_units = Unitful.NoUnits,
        force_units = Unitful.NoUnits
    )
    return sys, wall_range
end

# Update wall bead positions and velocities to follow the current shear state.
function enforce_wall_motion!(sys::Molly.System,
                              wall_indices::AbstractVector{Int},
                              wall_bases::AbstractVector{SVector{2,Float64}},
                              wall_sides::AbstractVector{Symbol},
                              shear::ShearProfile,
                              t::Real,
                              gap::Real,
                              box_side::Real)
    gamma, gamma_rate = shear_state(shear, t)

    @inbounds for (idx, base, side) in zip(wall_indices, wall_bases, wall_sides)
        disp = wall_displacement_from_shear(gamma, gap, side)
        speed = wall_speed_from_shear_rate(gamma_rate, gap, side)
        x = wrap_x(base[1] + disp, box_side)
        sys.coords[idx] = SVector(x, base[2])
        sys.velocities[idx] = SVector(speed, 0.0)
    end
end

# Keep bulk particles inside the gap in y by reflecting if they cross the walls.
function confine_bulk_y!(sys::Molly.System, n_bulk::Int, y_min::Float64, y_max::Float64)
    @inbounds for i in 1:n_bulk
        pos = sys.coords[i]
        vel = sys.velocities[i]
        if pos[2] > y_max
            new_y = 2y_max - pos[2]
            sys.coords[i] = SVector(pos[1], new_y)
            sys.velocities[i] = SVector(vel[1], -vel[2])
        elseif pos[2] < y_min
            new_y = 2y_min - pos[2]
            sys.coords[i] = SVector(pos[1], new_y)
            sys.velocities[i] = SVector(vel[1], -vel[2])
        end
    end
end

# Apply tangential drag near the walls to reduce slip. Drag is stronger closer to the wall.
function apply_wall_drag!(sys::Molly.System,
                          n_bulk::Int,
                          wall_y_top::Float64,
                          wall_y_bot::Float64,
                          shear::ShearProfile,
                          t::Real,
                          gap::Real,
                          box_side::Real;
                          dt::Real,
                          drag_coeff::Float64 = 1.5)
    gamma, gamma_rate = shear_state(shear, t)
    y_max = wall_y_top
    y_min = wall_y_bot
    v_top = wall_speed_from_shear_rate(gamma_rate, gap, :top)
    v_bot = wall_speed_from_shear_rate(gamma_rate, gap, :bottom)

    @inbounds for i in 1:n_bulk
        pos = sys.coords[i]
        vel = sys.velocities[i]

        dist_top = max(0.0, y_max - pos[2])
        dist_bot = max(0.0, pos[2] - y_min)

        # Linear weights across the gap to cover the whole bulk and cancel at midplane.
        w_top = max(0.0, 1.0 - dist_top / gap)
        w_bot = max(0.0, 1.0 - dist_bot / gap)

        w_sum = w_top + w_bot
        if w_sum == 0
            continue
        end

        v_target = (w_top * v_top + w_bot * v_bot) / w_sum
        drag = (vel[1] - v_target) * w_sum
        vx = vel[1] - drag_coeff * drag * dt
        sys.velocities[i] = SVector(vx, vel[2])
    end
end

function detect_clusters_from_file( #cluster detection andrej
    filename::String;
    box_side::Float64,
    cutoff::Float64,
    factor::Float64 = 1.2
)
    data = readdlm(filename)
    N = size(data, 1)

    coords = [(data[i,1], data[i,2]) for i in 1:N]
    r2 = (factor * cutoff)^2

    visited = falses(N)
    clusters = Vector{Vector{Int}}()

    for i in 1:N
        visited[i] && continue

        cluster = Int[]
        stack = [i]
        visited[i] = true

        while !isempty(stack)
            p = pop!(stack)
            push!(cluster, p)

            xi, yi = coords[p]

            for j in 1:N
                visited[j] && continue

                xj, yj = coords[j]
                dx = wrap_x(xj - xi, box_side)
                dy = yj - yi

                if dx*dx + dy*dy < r2
                    visited[j] = true
                    push!(stack, j)
                end
            end
        end

        push!(clusters, cluster)
    end

    return clusters
end

function analyze_clusters_simple(clusters, cutoff::Float64)
    bead_radius = cutoff / 2
    bead_area   = π * bead_radius^2

    n_clusters = length(clusters)

    beads_per_cluster = [length(c) for c in clusters]

    radii = [
        sqrt((length(c) * bead_area) / π)
        for c in clusters
    ]

    avg_radius = mean(radii)

    return n_clusters, beads_per_cluster, avg_radius
end

# defining repulsion parameters, much more stable behaviour with bigger a_oil_water and smaller a_water_water/a_oil_oil
const a_water_water = 5.0 # 5 <- better
const a_oil_oil     = 5.0 # 5 <- better
const a_oil_water   = 100.0 # 100 <- better

const a_wall_wall   = 25.0
const a_wall_water  = 80.0
const a_wall_oil    = 150.0

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
)

# Setting random seed for reproducibility
Random.seed!(1234)

volume_fraction_oil = 0.3 # VF
box_side            = 32.2
cutoff              = 1.0
density_number      = 3.0    # particles per unit area
temperature = 273 #temp in kelvin

# Rough wall configuration; motion is set later by the shear profile.
n_per_wall     = 600
wall_y_offset  = 0.5 * cutoff
wall_roughness = 0.8 * cutoff

volume_oil   = volume_fraction_oil * box_side^2
volume_water = (1.0 - volume_fraction_oil) * box_side^2
n_oil::Int   = ceil(volume_oil * density_number)
n_water::Int = ceil((box_side^2 - volume_oil) * density_number)

n_droplets = 4

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

# Generate initial positions for oil and water droplets
x0_oil, x0_water = if volume_fraction_oil <= 0.5
    droplet_area = volume_oil / n_droplets          # area of ONE droplet
    R_droplet    = sqrt(droplet_area / pi)

    centers = generate_droplet_centers(
        n_droplets,
        box_side,
        R_droplet;
        margin_factor = 1.5,
    )

    x0_oil, R_used = generate_multi_droplet(
        n_oil,
        centers,
        droplet_area,
    )

    x0_water = generate_outside_droplets(
        n_water,
        centers,
        R_used,
        box_side,
    )

    x0_oil, x0_water
else
    droplet_area = volume_water / n_droplets
    R_droplet    = sqrt(droplet_area / pi)

    centers = generate_droplet_centers(
        n_droplets,
        box_side,
        R_droplet;
        margin_factor = 1.5,
    )

    x0_water, R_used = generate_multi_droplet(
        n_water,
        centers,
        droplet_area,
    )

    x0_oil = generate_outside_droplets(
        n_oil,
        centers,
        R_used,
        box_side,
    )

    x0_oil, x0_water
end

x0_emulsion = vcat(x0_oil, x0_water)

n_bulk   = length(x0_emulsion)   # = n_oil + n_water
n_walls  = length(walls)
v0_bulk  = [random_vec(SVector{2,Float64},(-0.1,0.1)) for _ in 1:n_bulk]
v0_walls = [SVector{2,Float64}(0.0, 0.0) for _ in 1:n_walls]

dt = 0.001
T = 5.0 #CHANGE TIME HERE
nsteps = Int(round(T/dt))

# Time-dependent shear strain gamma(t). Supply any lambda you like here; gammȧ(t)
# is approximated numerically inside make_shear_profile.
shear_freq = 0.5            # cycles per unit time; adjust as needed
gamma_amplitude = 0.5       # peak strain
gamma_phase = 0.0
gamma_fn = t -> gamma_amplitude * sin(2*pi*shear_freq*t + gamma_phase)
shear_profile = make_shear_profile(gamma_fn = gamma_fn)

# Desired output fps; we will subsample to approximate this while keeping duration = T
desired_fps = 60

# Compute save_every from desired_fps; if not enough steps to reach target fps, save every frame
# frames_target = desired_fps * T; save_every ~ nsteps / frames_target
frames_target = max(1, Int(round(desired_fps * T)))
save_every = max(1, Int(floor(nsteps / frames_target)))
if save_every <= 0
    save_every = 1
end

sys, wall_range = build_emulsion_system(
    x0_emulsion,
    walls,
    n_oil,
    n_water,
    box_side,
    cutoff,
    v0_bulk,
    v0_walls,
    nsteps;
    periodic_y=false,
)
wall_indices = collect(wall_range)

pre_wall! = isempty(wall_indices) ? nothing : (step_idx -> begin
    t = step_idx == 0 ? 0.0 : (step_idx - 1) * dt
    enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
end)
post_wall! = isempty(wall_indices) ? nothing : (step_idx -> begin
    t = step_idx * dt
    enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
    apply_wall_drag!(sys, n_bulk, wall_y_top_ref, wall_y_bot_ref, shear_profile, t, wall_gap, box_side, dt=dt)
    confine_bulk_y!(sys, n_bulk, wall_y_bot_ref + cutoff, wall_y_top_ref - cutoff)
end)

simulator = VelocityVerlet(
    dt = dt
)

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
colors = [if i <= n_oil
            :yellow
        elseif i <= n_oil + n_water
            :blue
        else
            :gray
        end for i in 1:length(sys.coords)]

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
outdir = joinpath(@__DIR__, "Molly_mp4")
isdir(outdir) || mkpath(outdir) # Create directory if it doesn't exist

# Build filename
fname = "emulsion_molly_shear_velocity_verlet_dt$(dt)_aw$(a_oil_water)_ww$(a_water_water)_target$(desired_fps)_eff$(fps)_se$(save_every).mp4"
outfile = joinpath(outdir, fname)

# Custom renderer with progress bar for video writing
function visualize_with_progress(
    coords_history,
    boundary::Molly.RectangularBoundary,
    outfile;
    color,
    markersize::Real=1.0,
    framerate::Integer=30,
)
    frames = length(coords_history)
    isempty(coords_history) && error("No coordinates recorded; nothing to visualize.")

    xmin = Inf; xmax = -Inf; ymin = Inf; ymax = -Inf
    for frame in coords_history
        for p in frame
            x = p[1]; y = p[2]
            if isfinite(x) && isfinite(y)
                xmin = min(xmin, x); xmax = max(xmax, x)
                ymin = min(ymin, y); ymax = max(ymax, y)
            end
        end
    end
    if !isfinite(xmin) || !isfinite(ymin)
        error("Coordinates contain no finite values; cannot visualize trajectory.")
    end
    span_x = xmax - xmin
    span_y = ymax - ymin
    pad = 0.1 * max(max(span_x, span_y), 1.0)

    fig = GLMakie.Figure(size=(800,800))
    ax = GLMakie.Axis(fig[1,1];
        limits = (xmin - pad, xmax + pad, ymin - pad, ymax + pad),
        aspect = GLMakie.DataAspect(),
    )

    first_frame = coords_history[1]
    N = length(first_frame)
    xs = GLMakie.Observable([first_frame[i][1] for i in 1:N])
    ys = GLMakie.Observable([first_frame[i][2] for i in 1:N])

    # Scale marker sizes by particle type to make walls/oil more visible.
    msizes = [i <= n_oil ? 6.0 : i <= n_oil + n_water ? 4.0 : 3.5 for i in 1:N]

    GLMakie.scatter!(ax, xs, ys; color=color, markersize=msizes)

    # Progress for rendering
    local p = nothing
    if HAS_PROGRESSMETER[]
        p = ProgressMeter.Progress(frames; desc="Rendering", dt=0.2)
    end

    GLMakie.record(fig, outfile, 1:frames; framerate=framerate) do i
        ci = coords_history[i]
        @inbounds for k in 1:N
            xs[][k] = ci[k][1]
            ys[][k] = ci[k][2]
        end
        GLMakie.notify(xs); GLMakie.notify(ys)
        if p !== nothing
            ProgressMeter.next!(p)
        end
    end
end

visualize_with_progress(
    coords_history_wrapped,
    sys.boundary,
    outfile;
    color = colors,
    markersize = 1.0,
    framerate = fps,
)

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
