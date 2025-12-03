using StaticArrays
using Molly
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity
using Molly: visualize
using Unitful
using LinearAlgebra: norm
using GLMakie  # Make sure you have this loaded

function wrap_x(x::T, box_side::T) where T
    half_box = box_side / 2
    if x > half_box
        return x - box_side
    elseif x < -half_box
        return x + box_side
    else
        return x
    end
end

const a_water_water = 25.0
const a_oil_oil     = 25.0
const a_oil_water   = 80.0

# Map bead types to interaction strengths
const A_MAP = Dict{Tuple{Int,Int},Float64}(
    (1,1)     => a_oil_oil,
    (2,2) => a_water_water,
    (1,2)   => a_oil_water,
    (2,1)   => a_oil_water,
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



atoms_tiny = [Molly.Atom(mass=1.0, atom_type=1), Molly.Atom(mass=1.0, atom_type=1),
              Molly.Atom(mass=1.0, atom_type=2), Molly.Atom(mass=1.0, atom_type=2),
]    
cutoff_tiny = 3.0
temp_val = 1
p = 0.1
position_tiny = SVector{2,Float64}[
    SVector{2,Float64}(-p, -p),
    SVector{2,Float64}(p, p),
    SVector{2,Float64}(-p, p),
    SVector{2,Float64}(p, -p),
]

vel_tiny = zeros(SVector{2,Float64}, 4)
boundary_tiny = Molly.RectangularBoundary(1.0)
neighbor_finder_tiny = Molly.DistanceNeighborFinder(
    eligible=trues(4, 4),
    n_steps=10,
    dist_cutoff=cutoff_tiny,
)

sys = Molly.System(
    atoms = atoms_tiny,
    coords = position_tiny,
    boundary = boundary_tiny,
    velocities = vel_tiny,
    pairwise_inters = (MyPairwiseInter=EmulsionInter(cutoff_tiny),),
    neighbor_finder = neighbor_finder_tiny,
    loggers = (coords=MyCoordinatesLogger(1,dims=2),),
    energy_units = Unitful.NoUnits,
    force_units = Unitful.NoUnits,
    k = 1.0/temp_val,
)

dt = 0.001
n_steps = 1000
verlet = Verlet(dt=dt)
simulate!(sys, verlet, n_steps);
how_many = length(atoms_tiny)
colors_tiny = [if i <= how_many/2
            :yellow
        elseif i <= how_many
            :blue
        else
            :gray
        end for i in 1:how_many]

visualize(
    sys.loggers.coords.history,
    sys.boundary,
    "emulsion_molly_tiny_positive_velocity_verlet_dt_$dt.mp4";
    color = [:yellow, :orange, :blue, :gray],
    markersize = 0.05,
    framerate = 10,
)