using Random
using StaticArrays
using Molly
using GLMakie  # Make sure you have this loaded
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity, visualize
using Unitful
using LinearAlgebra: norm
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
Molly.use_neighbors(::EmulsionInter) = true # this specifies if simulation is computing forces between every pair (false) or only neighbors (true)

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

    r = norm(vec_ij) # wraping is handled by Molly simulator, so not needed here
    disp = vec_ij
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

function build_emulsion_system(x0_all::Vector{SVector{2,Float64}},
                               n_oil::Int, n_water::Int,
                               box_side::Float64, cutoff::Float64, velocities)

# The commented out function below is an earlier version of build_emulsion_system with the walls - maybe useful, if not needed, can be deleted.
# Build system: assign atom types so EmulsionInter can choose a_ij
# function build_emulsion_system(x0_all::Vector{SVector{2,Float64}},
#                                n_oil::Int, n_water::Int, n_topwall::Int, n_botwall::Int,
#                                box_side::Float64, cutoff::Float64, velocities)

    boundary = Molly.RectangularBoundary(box_side) # define periodic 2D box
    atoms = Vector{Molly.Atom}(undef, n_oil+n_water) # initialize atoms vector
    
    # oil
    for i in 1:n_oil
        atoms[i] = Molly.Atom(mass=1.0, atom_type=1) 
       
    end
    # water
    for i in (n_oil+1):(n_oil+n_water)
        atoms[i] = Molly.Atom(mass=1.0, atom_type=2)
        
    end
    # old walls implementation - delete if not needed
    # for i in (n_oil+n_water+1):length(x0_all)
    #     atoms[i] = Molly.Atom(mass=1.0, atom_type=3)
    # end
    
    # Define neighbor finder using CellListMap (exactly the same as in our previous code)
    cellListMap_finder = Molly.CellListMapNeighborFinder(
        eligible=trues(n_bulk, n_bulk), # all bulk particles interact
        dist_cutoff=cutoff,
        x0=x0_all[1:n_bulk],
        unit_cell = boundary,
        n_steps = 1, # update neighbor list every step
        dims = 2,
    )
    
    # Build Molly system that will be solved by a simulator
    sys = Molly.System(
        atoms = atoms,
        coords = x0_all[1:n_bulk],
        boundary = boundary,
        velocities = velocities,
        pairwise_inters = (MyPairwiseInter=EmulsionInter(cutoff),),
        neighbor_finder = cellListMap_finder,
        loggers = (coords=MyCoordinatesLogger(1,dims=2),),
        energy_units = Unitful.NoUnits,
        force_units = Unitful.NoUnits
    )
    return sys
end
# defining repulsion parameters, much more stable behaviour with bigger a_oil_water and smaller a_water_water/a_oil_oil
const a_water_water = 25.0 # 5 <- better
const a_oil_oil     = 25.0 # 5 <- better
const a_oil_water   = 80.0 # 100 <- better

const a_wall_wall   = 25.0
const a_wall_water  = 25.0
const a_wall_oil    = 80.0

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

volume_fraction_oil = 0.3
box_side            = 32.2
cutoff              = 1.0
density_number      = 3.0    # particles per unit area

volume_oil   = volume_fraction_oil * box_side^2
volume_water = (1.0 - volume_fraction_oil) * box_side^2
n_oil::Int   = ceil(volume_oil * density_number)
n_water::Int = ceil((box_side^2 - volume_oil) * density_number)

n_droplets = 4

# Commented out wall generation - can be deleted if not needed
# n_per_wall = 400
# x_top, x_bot = make_rough_wall_particles(SVector{2,Float64}, box_side, n_per_wall;
#                                             y_offset = 0.0)
# walls = vcat(x_top, x_bot)

# Generate initial positions for oil and water droplets, the same as in our old code (walls commented out)
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
        #walls,             # <- avoid walls too
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
        #walls,             # <- avoid walls too
    )

    x0_oil, x0_water
end

x0_emulsion = vcat(x0_oil, x0_water)
#x0_all_u      = vcat(x0_emulsion, walls)
x0_all = [SVector(x[1], x[2]) for x in x0_emulsion]

n_bulk   = length(x0_emulsion)   # = n_oil + n_water
v0_all = [random_vec(SVector{2,Float64},(-0.1,0.1)) for _ in 1:n_bulk]

# walls implementation - delete if not needed
#n_wall   = length(walls)
# n_total  = n_bulk + n_wall
# n_topwall  = length(x_top)
# n_botwall  = length(x_bot)
# n_total    = length(x0_all)
#v0_all = [SVector{2,Float64}(random_velocity(1.0, temp_val; dims=2)...) for _ in 1:n_bulk]
#append!(v0_all, [SVector{2,Float64}(0.0, 0.0) for _ in 1:(n_topwall+n_botwall)])


sys = build_emulsion_system(x0_all, n_oil, n_water, box_side, cutoff, v0_all)

dt = 0.001
T = 100
nsteps = Int(T/dt)
simulator = VelocityVerlet(
    dt = dt
)

sim_time = @elapsed simulate!(sys, simulator, nsteps)
println("Simulation completed in $sim_time seconds.")
colors = [if i <= n_oil
            :yellow
        elseif i <= n_oil + n_water
            :blue
        else
            :gray
        end for i in 1:length(x0_all)]

coords_history = sys.loggers.coords.history
# Ensure output dir exists
outdir = joinpath(@__DIR__, "Molly_mp4")
isdir(outdir) || mkpath(outdir) # Create directory if it doesn't exist

# Build filename
fname = "emulsion_molly_positive_velocity_verlet_$(dt)_$(a_oil_water)_$(a_water_water).mp4"
outfile = joinpath(outdir, fname)

visualize(
    coords_history,
    sys.boundary,
    outfile;
    color = colors,
    markersize = 0.8,
    framerate = Int(nsteps/50), # so the simulation will last approx 50 seconds
)

