# If the other file defines the functions at top level:
using StaticArrays
using Molly
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity
using Molly: visualize
using Unitful
using LinearAlgebra: norm
include(joinpath(@__DIR__, "functions.jl"))
# Functions are now available: generate_droplet_centers, generate_oil_particles_in_droplets,
# generate_water_particles, make_rough_wall_particles

# If that file defines a module (e.g. module DropletGen ... end), do:
# include(joinpath(@__DIR__, "Chang_paper_MDS-modified_with_shear_strain.jl"))
# using .DropletGen: generate_droplet_centers, generate_oil_particles_in_droplets,
#     generate_water_particles, make_rough_wall_particles
volume_fraction_oil = 0.3
box_side            = 32.2
cutoff              = 1.0
density_number      = 3.0    # particles per unit area
const a_water_water = 25.0
const a_oil_oil     = 25.0
const a_oil_water   = 80.0

const a_wall_wall   = 25.0
const a_wall_water  = 25.0
const a_wall_oil    = 80.0

volume_oil   = volume_fraction_oil * box_side^2
volume_water = (1.0 - volume_fraction_oil) * box_side^2
n_oil::Int   = ceil(volume_oil * density_number)
n_water::Int = ceil((box_side^2 - volume_oil) * density_number)

n_droplets = 4

n_per_wall = 400
# x_top, x_bot = make_rough_wall_particles(SVector{2,Float64}, box_side, n_per_wall;
#                                             y_offset = 0.0)
# walls = vcat(x_top, x_bot)

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
#box_emulsion = Box([box_side, box_side], cutoff)
#cl_emulsion  = CellList(x0_all, box_emulsion)

n_bulk   = length(x0_emulsion)   # = n_oil + n_water
#n_wall   = length(walls)
# n_total  = n_bulk + n_wall
# n_topwall  = length(x_top)
# n_botwall  = length(x_bot)
# n_total    = length(x0_all)
temp_val = 293.15
v0_all = [SVector{2,Float64}(random_velocity(1.0, temp_val; dims=2)...) for _ in 1:n_bulk]
#append!(v0_all, [SVector{2,Float64}(0.0, 0.0) for _ in 1:(n_topwall+n_botwall)])


# Tag atoms with a type so we can choose a_ij
abstract type BeadType end
struct Oil   <: BeadType end
struct Water <: BeadType end
#struct Wall  <: BeadType end

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

# Custom pairwise interaction with cutoff
struct EmulsionInter{T} <: PairwiseInteraction
    cutoff::T
end
Molly.use_neighbors(::EmulsionInter) = true

# Pairwise scalar force law (depends only on distance)
function Molly.pairwise_force(inter::EmulsionInter, r, params)
    a_ij, repulsive = params
    if r > inter.cutoff
        return 0.0
    elseif repulsive && r < 1e-12
        return 3000 * (1 - r/inter.cutoff)
    else
        return a_ij * (1 - r/inter.cutoff)
    end
end

# Vector force along minimum-image displacement (x periodic only)
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

    # vec_ij is displacement; enforce x-periodic, y direct
    Δx = wrap_x(vec_ij[1], boundary.side_lengths[1])  # periodic in x
    Δy = vec_ij[2]                                # non-periodic in y
    disp = SVector(Δx, Δy)
    r = norm(disp)
    # Choose a_ij from atom "type" (stored in atom metadata)
    ti = atom_i.atom_type  # e.g., Oil/Water/Wall
    tj = atom_j.atom_type
    a_ij = get(A_MAP, (ti, tj), 0.0)
    repulsive = a_ij == a_oil_oil
    params = (a_ij, repulsive)

    fmag = Molly.pairwise_force(inter, r, params)
    return r > 0 ? fmag * disp / r : SVector(0.0, 0.0)
end

function MyCoordinatesLogger(T, n_steps::Integer; dims::Integer=3)
    return Molly.GeneralObservableLogger(
        Molly.coordinates_wrapper,
        Array{SArray{Tuple{dims}, T, 1, dims}, 1},
        n_steps,
    )
end

MyCoordinatesLogger(n_steps::Integer; dims::Integer=3) = MyCoordinatesLogger(Float64, n_steps; dims=dims)
# Build system: assign atom types so EmulsionInter can choose a_ij
# function build_emulsion_system(x0_all::Vector{SVector{2,Float64}},
#                                n_oil::Int, n_water::Int, n_topwall::Int, n_botwall::Int,
#                                box_side::Float64, cutoff::Float64, velocities)
function build_emulsion_system(x0_all::Vector{SVector{2,Float64}},
                               n_oil::Int, n_water::Int,
                               box_side::Float64, cutoff::Float64, velocities)
    boundary = Molly.RectangularBoundary(box_side)
    atoms = Vector{Molly.Atom}(undef, n_oil+n_water)
    
    # oil
    for i in 1:n_oil
        atoms[i] = Molly.Atom(mass=1.0, atom_type=1)
       
    end
    # water
    for i in (n_oil+1):(n_oil+n_water)
        atoms[i] = Molly.Atom(mass=1.0, atom_type=2)
        
    end
    # walls (rest)
    # for i in (n_oil+n_water+1):length(x0_all)
    #     atoms[i] = Molly.Atom(mass=1.0, atom_type=3)
    # end
    
    
    neighbor_finder = Molly.DistanceNeighborFinder(
        eligible=trues(n_bulk, n_bulk),
        n_steps=10,
        dist_cutoff=cutoff,
    )
    
    sys = Molly.System(
        atoms = atoms,
        coords = x0_all[1:n_bulk],
        boundary = boundary,
        velocities = velocities,
        pairwise_inters = (MyPairwiseInter=EmulsionInter(cutoff),),
        neighbor_finder = neighbor_finder,
        loggers = (coords=MyCoordinatesLogger(1,dims=2),),
        energy_units = Unitful.NoUnits,
        force_units = Unitful.NoUnits,
        k = 1.0/temp_val,
    )
    return sys
end


# Example usage (replace your manual forces_emulsion!/fpair with Molly)
# x0_all, n_* prepared earlier
# velocities prepared earlier (bulk random, walls zero)
sys = build_emulsion_system(x0_all, n_oil, n_water, box_side, cutoff, v0_all)

dt = 0.001
nsteps = 10000
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


using GLMakie  # Make sure you have this loaded

coords_history = sys.loggers.coords.history
visualize(
    coords_history,
    sys.boundary,
    "emulsion_molly_positive_velocity_verlet_repulsive.mp4";
    color = colors,
    markersize = 0.8,
    framerate = 50,
)


# visualize(
#     sys.loggers.coords,
#     sys.boundary,
#     filename = "emulsion_molly.mp4",
#     colors = colors,
# )
# Then evolve with Molly’s integrator; your wall motions and x-periodic wrapping
# can stay in md_Verlet_walls, since you already update coords/vels there.
# Inside md_Verlet_walls, replace forces!(f,x) with:
#   f .= Molly.forces(sys)   # after setting sys.coords = x and sys.velocities = v
# Or call Molly.step! if you move fully to Molly’s time integration.