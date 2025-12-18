
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
include(joinpath(@__DIR__, "system_and_simulation_functions.jl"))
# MyCoordinatesLogger, run_and_collect!, build_emulsion_system
include(joinpath(@__DIR__, "plotting_functions.jl"))
# visualize_with_progress, plot_hull_points, visualize_soft_spheres_with_progress

# Setting random seed for reproducibility
Random.seed!(1234)

#region ------- Simulation Parameters -----
# -------- General system parameters --------
volume_fraction_oil = 0.8                   
box_side            = 5.0
temperature         = 273                   # temp in kelvin

# ------------- Droplet settings ------------
#n_droplets = 20               # number of droplets is now based on volume fraction and droplet size
droplet_radius     = 0.1 
droplet_mass       = 100.0
energy_strength    = 0.00001
cutoff             = 5*droplet_radius    # cutoff is disabled for droplet_radius>0.3 because then the number of droplets is small (so there is no need to use neoighbors list)
minimal_dist_frac  = 0.5                  # minimal distance between initial droplet centers as fraction of 2*R

# ------------ Simulation settings ----------
dt = 0.001
T = 10000*dt                              # CHANGE TIME HERE
nsteps = Int(round(T/dt))
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
        loggers = (coords=MyCoordinatesLogger(1, dims=2),),
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
simulator = VelocityVerlet(dt = dt,coupling = Molly.AndersenThermostat(temperature,1.0))

sim_time = @elapsed begin
    coords_history = run_and_collect!(
        sys,
        simulator,
        nsteps;
        save_every=1,
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
visualize_soft_spheres_with_progress(
    coords_history,
    boundary,
    joinpath(outdir, "sim_soft_spheres_vf$(round(volume_fraction_oil, digits=2))_d$(round(droplet_radius, digits=2)).mp4");
    box_side=box_side,
    framerate=fps,
    droplet_radius=droplet_radius,
)
#endregion