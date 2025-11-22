using StaticArrays

# Defnition of a position vector for each particle
struct Vec2D{T} <: FieldVector{2,T} # A 2D vector with components of type T
    x::T
    y::T
end

#Function that generates a random vector of type VecType within a specified range
function random_vec(::Type{VecType},range) where VecType 
    dim = length(VecType)
    T = eltype(VecType)
    p = VecType(
        range[begin] + rand(T)*(range[end]-range[begin]) for _ in 1:dim
    )
    return p
end

import LinearAlgebra: norm

# Energy function (for two particles) based on the distance between two particles with a cutoff
function energy(x::T,y::T,cutoff) where T
    Δv = y - x # distance vector between two particles
    d = norm(Δv)
    if d > cutoff
        energy = zero(T)
    else
        energy = (d - cutoff)^2
    end
    return energy
end

# Force function (between two particles) based on the distance between two particles with a cutoff
function fₓ(x::T,y::T,cutoff) where T
    Δv = y - x
    d = norm(Δv)
    if d > cutoff
        fₓ = zero(T)
    else
        fₓ = 2*(d - cutoff)*(Δv/d) # the equation is according to the gradient of the energy function
    end
    return fₓ
end

const cutoff = 5. # cutoff distance for interaction

# Function that computes forces on all particles in the system, ! in the name indicates that f is modified in place
function forces!(f::Vector{T}, x::Vector{T}, fₓ::F) where {T,F} # fₓ: function that computes force between two particles
    fill!(f, zero(T))
    n = length(x)
    for i in 1:n-1
        for j in i+1:n
            fᵢ = fₓ(i, j, x[i], x[j])
            f[i] += fᵢ
            f[j] -= fᵢ
        end
    end
    return f
end


x0 = [ random_vec(Vec2D{Float64},(0,100)) for _ in 1:100] # Initial positions of 100 particles in a 2D space
f0 = similar(x0) # Preallocate force vector

# Compute forces on all particles using the defined force function with cutoff
forces!(
    f0,
    x0,
    (i,j,x,y) -> fₓ(x,y,cutoff) # map input of the form (i,j,x,y) to fₓ(x,y,cutoff)
) 

println("Forces on particles:")
for (i, f) in enumerate(f0)
    println("Particle $i: Force = $f")
end

# Molecular Dynamics simulation function
function md(x0::Vector{T}, v0::Vector{T}, mass, dt, nsteps, isave, forces!) where T
    #=
    x0: Initial positions of particles (vector of T)
    v0: Initial velocities of particles (vector of T)
    =#

    x = copy(x0)
    v = copy(v0)
    a = similar(x0)
    f = similar(x0) # Initialize force vector
    trajectory = [ copy(x0) ] # will store the trajectory
    for step in 1:nsteps
        # Compute forces and store in f
        forces!(f,x)
        # Accelerations (@. means element-wise operation)
        @. a = f / mass
        # Update positions
        @. x = x + v*dt + a*dt^2/2
        # Update velocities
        @. v = v + a*dt
        # Save the trajectory at specified intervals
        if mod(step,isave) == 0
            println("Saved trajectory at step: ",step)
            push!(trajectory,copy(x))
        end
    end
    return trajectory
end



# Running the MD simulation with 100 particles
trajectory = md((
    x0 = [random_vec(Vec2D{Float64},(-50,50)) for _ in 1:100 ], 
    v0 = [random_vec(Vec2D{Float64},(-1,1)) for _ in 1:100 ], 
    mass = [ 1.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x, (i,j,p1,p2) -> fₓ(p1,p2,cutoff))
)...)

# Initialize default plot settings
begin
    using Plots
    plot_font = "Computer Modern"
    default(
        fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=:none, grid=false,
        size=(400,400)
    )    
end

# Visualizing the trajectory of particles
anim = @animate for frame in trajectory
    scatter(
        [ p.x for p in frame ],
        [ p.y for p in frame ],
        xlim=(-200,200), ylim=(-200,200),
        title="Molecular Dynamics Simulation",
        xlabel="X Position", ylabel="Y Position",
        markersize=4,
    )
end

# Save as GIF
gif_dir = joinpath(@__DIR__, "gif_martinez_Md")
isdir(gif_dir) || mkpath(gif_dir)
gif_path = joinpath(gif_dir, "trajectory.gif")
gif(anim, gif_path, fps=20)

# Implementing periodic boundary conditions (as the domain is wrapped in a ring)
function wrap(x,side)
    x = rem(x,side) #  like mod(x,side) but works for negative x (returns negative values)
    if x >= side/2 # if distance is more than half the side then the particles are closer through the other side
        x -= side
    elseif x < -side/2 # same logic for negative vector
        x += side
    end
    return x
end

# Modified force function with periodic boundary conditions
function fₓ(x::T,y::T,cutoff,side) where T
    Δv = wrap.(y - x, side) # apply wrap function element-wise
    d = norm(Δv)
    if d > cutoff
        fₓ = zero(T)
    else
        fₓ = 2*(d - cutoff)*(Δv/d)
    end
    return fₓ
end

# Molecular Dynamics simulation function with wrapping of the positions 
function md_wrap(x0::Vector{T}, v0::Vector{T}, mass, dt, nsteps, isave, forces!,side) where T
    #=
    x0: Initial positions of particles (vector of T)
    v0: Initial velocities of particles (vector of T)
    =#

    x = copy(x0)
    v = copy(v0)
    a = similar(x0)
    f = similar(x0) # Initialize force vector
    trajectory = [ copy(x0) ] # will store the trajectory
    for step in 1:nsteps
        # Compute forces and store in f
        forces!(f,x)
        # Accelerations (@. means element-wise operation)
        @. a = f / mass
        # Update positions
        @. x = x + v*dt + a*dt^2/2
        for i in 1:length(x)
            x[i] = wrap.(x[i],side) # ensure positions are wrapped within the box
        end
        # Update velocities
        @. v = v + a*dt
        # Save the trajectory at specified intervals
        if mod(step,isave) == 0
            println("Saved trajectory at step: ",step)
            push!(trajectory,copy(x))
        end
    end
    return trajectory
end

# Running the MD simulation with periodic boundary conditions
const side = 100.0 # Size of the periodic box


trajectory_periodic = md_wrap((
    x0 = [random_vec(Vec2D{Float64},(-50,50)) for _ in 1:100 ], 
    v0 = [random_vec(Vec2D{Float64},(-1,1)) for _ in 1:100 ], 
    mass = [ 10.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 10000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x,(i,j,p1,p2) -> fₓ(p1,p2,cutoff,side)),
    # julia knows to use the modified fₓ with periodic boundaries as it has more arguments than the previous one
    side = side
)...)

# Visualizing the trajectory with periodic boundary conditions
anim_periodic = @animate for frame in trajectory_periodic
    scatter(
        [ p.x for p in frame ],
        [ p.y for p in frame ],
        xlim=(-side/2-30,side/2+30), ylim=(-side/2-30,side/2+30),
        title="MD Simulation with Periodic Boundary Conditions",
        xlabel="X Position", ylabel="Y Position",
        markersize=4,
    )
end

# Save as GIF
gif_path = joinpath(gif_dir, "trajectory_periodic.gif")
gif(anim_periodic, gif_path, fps=20)


# One can easily change the 2D model to 3D 
struct Vec3D{T} <: FieldVector{3,T}
    x::T
    y::T
    z::T
end

trajectory_periodic_3D = md_wrap((
    x0 = [random_vec(Vec3D{Float64},-50:50) for _ in 1:100 ], 
    v0 = [random_vec(Vec3D{Float64},-1:1) for _ in 1:100 ], 
    mass = [ 1.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x,(i,j,p1,p2) -> fₓ(p1,p2,cutoff,side)),
    side = side
)...)

# Visualizing the 3D trajectory with periodic boundary conditions
anim_periodic_3D = @animate for frame in trajectory_periodic_3D
    scatter3d(
        [ p.x for p in frame ],
        [ p.y for p in frame ],
        [ p.z for p in frame ],
        xlim=(-side/2-30,side/2+30), ylim=(-side/2-30,side/2+30), zlim=(-side/2-30,side/2+30),
        title="3D MD Simulation with Periodic Boundary Conditions",
        xlabel="X Position", ylabel="Y Position", zlabel="Z Position",
        markersize=4,
    )
end

# Save as GIF
gif_path = joinpath(gif_dir, "trajectory_periodic_3D.gif")
gif(anim_periodic_3D, gif_path, fps=20)

# Error propagation 
using Measurements

function random_vec(::Type{Vec2D{Measurement{T}}},range,Δ) where T # Measurement{T} is a measurement type that holds a value and its uncertainty
    p = Vec2D(
        range[begin] + rand(T)*(range[end]-range[begin]) ± rand()*Δ,
        range[begin] + rand(T)*(range[end]-range[begin]) ± rand()*Δ
    )
    return p
end

trajectory_2D_error = md_wrap((
    x0 = [random_vec(Vec2D{Measurement{Float64}},(-50,50),1e-5) for _ in 1:100 ], 
    v0 = [random_vec(Vec2D{Measurement{Float64}},(-1,1),1e-5) for _ in 1:100 ],
    mass = [ 1.0 for _ in 1:100 ],
    dt = 0.1,
    nsteps = 100,
    isave = 1,
    forces! = (f,x) -> forces!(f,x, (i,j,p1,p2) -> fₓ(p1,p2,cutoff,side)),
    side = side
)...) 

# Vizualizing the trajectory with uncertainties
anim_2D_error = @animate for frame in trajectory_2D_error
    histogram(
        [ p.x.err for p in frame ],
        xlabel="Uncertainty in x",ylabel="Number of points",
        bins=0:1e-4:20e-4,ylims=[0,50]
    )
end

# Save as GIF
gif_path = joinpath(gif_dir, "trajectory_2D_error.gif")
gif(anim_2D_error, gif_path, fps=20)

# Speeding up the code using cell lists
using CellListMap # package that divides the domain into cells and only computes interactions between particles in neighboring cells
import CellListMap: Box, CellList, UpdateCellList!, map_pairwise!
const n_large = 1000
const box_side = sqrt(n_large / (100/100^2))
x0_large = [ random_vec(Vec2D{Float64},(-box_side/2,box_side/2)) for _ in 1:n_large ]

box = Box([box_side,box_side],cutoff) # define the simulation box with given side length and cutoff

cl = CellList(x0_large,box) # initialize the cell list

# Modified force pair function that works with cell lists
function fpair_cl(x,y,i,j,d2,f,box::Box)
    Δv = y - x
    d = sqrt(d2)
    fₓ = 2*(d - box.cutoff)*(Δv/d)
    f[i] += fₓ
    f[j] -= fₓ
    return f
end

# Function that computes total forces acting on each particle using cell lists
function forces_cl!(f::Vector{T},x,box::Box,cl::CellList,fpair::F) where {T,F}
    fill!(f,zero(T))
    cl = UpdateCellList!(x,box,cl,parallel=false)
    map_pairwise!( #map_pairwise! is a function from CellLists.jl that applies a function to all pairs of particles in neighboring cells
        (x,y,i,j,d2,f) -> fpair(x,y,i,j,d2,f,box),
        f, box, cl, parallel=false
    )
    return f
end

t_naive = @elapsed trajectory_periodic_large = md_wrap((
    x0 = x0_large, 
    v0 = [random_vec(Vec2D{Float64},(-1,1)) for _ in 1:n_large ], 
    mass = [ 10.0 for _ in 1:n_large ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces!(f,x,(i,j,p1,p2) -> fₓ(p1,p2,cutoff,box_side)),
    side = box_side
)...)

t_cell_lists = @elapsed trajectory_cell_lists = md_wrap((
    x0 = x0_large, 
    v0 = [random_vec(Vec2D{Float64},(-1,1)) for _ in 1:n_large ], 
    mass = [ 10.0 for _ in 1:n_large ],
    dt = 0.1,
    nsteps = 1000,
    isave = 10,
    forces! = (f,x) -> forces_cl!(f,x,box,cl,fpair_cl),
    side = box_side
)...)

println("Time taken without cell lists: ",t_naive," seconds")
println("Time taken with cell lists: ",t_cell_lists," seconds")

# Visualizing the trajectory with cell lists
anim_cell_lists = @animate for frame in trajectory_cell_lists
    scatter(
        [ p.x for p in frame ],
        [ p.y for p in frame ],
        xlim=(-box_side/2-30,box_side/2+30), ylim=(-box_side/2-30,box_side/2+30),
        title="MD Simulation with Cell Lists",
        xlabel="X Position", ylabel="Y Position",
        markersize=2,
    )
end

# Save as GIF
gif_path = joinpath(gif_dir, "trajectory_cell_lists.gif")
gif(anim_cell_lists, gif_path, fps=20)

# One can definie the potential energy of the system differently by changing the energy function defined earlier

using FastPow # package for fast exponentiation

# Lennard-Jones potential energy function
function ulj_pair(r2,u,ε,σ)
    @fastpow u += 4*ε*(σ^12/r2^12 - 2*σ^6/r2^6)
    return u
end

# Function that computes 
function ulj(x,ε,σ,box::Box,cl::CellList)
    cl = UpdateCellList!(x,box,cl,parallel=false)
    u = map_pairwise!(
        (x,y,i,j,d2,u) -> ulj_pair(d2,u,ε,σ),
        zero(eltype(σ)), box, cl,
        parallel=false
    )
    return u
end

# Lennard-Jones force function
function flj_pair!(x,y,i,j,r2,f,ε,σ,box_side)
    if r2 < 1# 1e-8
        return f  # skip pathological overlap (then the force is undefined and the simulation loses stability)
    end
    @fastpow begin
        inv_r2 = 1/r2
        sr2 = (σ^2)*inv_r2          # (σ^2 / r^2)
        sr6 = sr2^3                 # (σ^6 / r^6)
        sr12 = sr6^2                # (σ^12 / r^12)
        pref = 24*ε*inv_r2*(2*sr12 - sr6)  # LJ force factor
        ∂u∂x = pref*wrap.(x - y, box_side) # apply periodic boundary conditions
    end
    f[i] -= ∂u∂x
    f[j] += ∂u∂x
    return f
end

# Function that computes forces acting on each particle using Lennard-Jones force 
function flj!(f::Vector{T},x,ε,σ,box,cl,box_side) where T
    cl = UpdateCellList!(x,box,cl,parallel=false)
    fill!(f,zero(T))
    map_pairwise!(
        (x,y,i,j,d2,f) -> flj_pair!(x,y,i,j,d2,f,ε,σ,box_side),
        f, box, cl, 
        parallel=false
    )
    return f
end

# For Neon Gas
const ε = 0.0441795 # kcal/mol
const σ = 2*1.64009 # Å
const n_Ne = 10_000 # number of neon atoms
const box_side_Ne = (10_000/0.1)^(1/3) # box side length for density of 0.1 atoms/Å^3
x0_Ne = [ random_vec(Vec3D{Float64},(-box_side_Ne/2,box_side_Ne/2)) for _ in 1:n_Ne ]
const box_Ne = Box([box_side_Ne,box_side_Ne,box_side_Ne],3σ) # cutoff of 3σ
const cl_Ne = CellList(x0_Ne,box_Ne)
t_Ne = @elapsed trajectory_Ne = md_wrap((
    x0 = x0_Ne, 
    v0 = [random_vec(Vec3D{Float64},(-0.1,0.1)) for _ in 1:n_Ne ], 
    mass = [ 20.18 for _ in 1:n_Ne ], # mass of neon atom in amu
    dt = .001,
    nsteps = 100,
    isave = 1,
    forces! = (f,x) -> flj!(f,x,ε,σ,box_Ne,cl_Ne,box_side_Ne),
    side = box_side_Ne
)...)
println("Time taken for Neon gas simulation: ",t_Ne," seconds")
# Visualizing a slice of the Neon gas trajectory
anim_Ne = @animate for frame in trajectory_Ne
    scatter3d(
        [ p.x for p in frame if abs(p.z) < box_side_Ne/20 ],
        [ p.y for p in frame if abs(p.z) < box_side_Ne/20 ],
        [ p.z for p in frame if abs(p.z) < box_side_Ne/20 ],
        xlim=(-box_side_Ne/2,box_side_Ne/2), ylim=(-box_side_Ne/2,box_side_Ne/2), zlim=(-box_side_Ne/2,box_side_Ne/2),
        title="Neon Gas MD Simulation Slice",
        xlabel="X Position", ylabel="Y Position", zlabel="Z Position",
        markersize=2,
    )
end
# Save as GIF

gif_path = joinpath(gif_dir, "trajectory_Ne_slice.gif")
gif(anim_Ne, gif_path, fps=20)

