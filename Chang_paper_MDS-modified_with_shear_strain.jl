using CellListMap # package that divides the domain into cells and only computes interactions between particles in neighboring cells
import CellListMap: Box, CellList, UpdateCellList!, map_pairwise!
using StaticArrays
import LinearAlgebra: norm


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

#Molecular Dynamics simulator using velocity Verlet algorithm
function md_Verlet(x0::Vector{T}, v0::Vector{T}, mass, dt, box_side, nsteps, isave, forces!) where T
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
            x[i] = wrap.(x[i],box_side) # ensure positions are wrapped within the box
        end
        # Update velocities only at half-time step
        @. v = v + 0.5*a*dt
        # Recompute forces at new positions
        forces!(f,x)
        # Update accelerations
        @. a = f / mass
        # Complete velocity update
        @. v = v + 0.5*a*dt
        # Save the trajectory at specified intervals
        if mod(step,isave) == 0
            println("Saved trajectory at step: ",step)
            push!(trajectory,copy(x))
        end
    end
    return trajectory
end

# function that calculates pairwise forces between oil and water particles in an emulsion
function forces_emulsion!(f::Vector{T},x,box::Box,side,cl::CellList,n_oil,n_water,fpair::F) where {T,F}
    fill!(f,zero(T))
    cl = UpdateCellList!(x,box,cl,parallel=false)
    map_pairwise!(
        (x,y,i,j,d2,f) -> begin
            if i <= n_oil && j <= n_oil
                fpair(i,j,x,y,box.cutoff,side,a_oo,f)
            elseif i > n_oil && j > n_oil
                fpair(i,j,x,y,box.cutoff,side,a_ww,f)
            else
                fpair(i,j,x,y,box.cutoff,side,a_ow,f)
            end
        end,
        f, box, cl,
        parallel=true
    )
    return f
end

# Force calculation between a pair of emulsion particles
function f_emulsion_pair!(i,j,x::T,y::T,cutoff,side,a_ij,f) where T
    Δv = wrap.(y - x, side)
    d = norm(Δv)
    in_repulsive = a_ij == a_oo
    if d > cutoff
        fₓ = zero(T)
    elseif d<1e-12 && in_repulsive
        fₓ = 3000*(1-d/cutoff)*(Δv/d)
    else
        fₓ = a_ij*(1-d/cutoff)*(Δv/d)
    end
    f[i] += fₓ
    f[j] -= fₓ
    return f
end

# Function to plot emulsion interaction forces (to see how they behave with distance)
function femulsion_plot(d,cutoff,a_ij)
    in_repulsive = a_ij == a_oo
    if d > cutoff
        fₓ = 0
    elseif d < 1e-12 && in_repulsive
        fₓ = 3000*(1-d/cutoff)
    else
        fₓ = a_ij*(1-d/cutoff)
    end
    return fₓ
end

# defining repulsion parameters
const cutoff = 1
const a_ww = -25
const a_oo = -25
const a_ow = -80

# Plotting the emulsion forces between particles
using Plots
d_values = 0:0.01:1.5*cutoff
f_oo = [ femulsion_plot(d,cutoff,a_oo) for d in d_values ]
f_ww = [ femulsion_plot(d,cutoff,a_ww) for d in d_values ]
f_ow = [ femulsion_plot(d,cutoff,a_ow) for d in d_values ]
plot(
    d_values, f_oo,
    label="Oil-Oil Interaction",
    xlabel="Distance", ylabel="Force Magnitude",
    title="Emulsion Interaction Forces",
    legend=:topright,
    linewidth=2,
)
plot!(
    d_values, f_ww,
    label="Water-Water Interaction",
    linewidth=2,
)
plot!(
    d_values, f_ow,
    label="Oil-Water Interaction",
    linewidth=2,
)
gif_dir = joinpath(@__DIR__, "gif_chang_MD")
isdir(gif_dir) || mkpath(gif_dir)
savefig(joinpath(gif_dir, "emulsion_forces.png"))

# Defining simulation parameters for the emulsion MD simulation
const box_side = 32.2
const density_number = 3
const nsteps = 10^4
const dt = 0.01

# Place n_droplets centers inside the box, separated by at least ~2R
function generate_droplet_centers(n_droplets::Int,
                                  box_side::T,
                                  R::T;
                                  margin_factor::T = 1.5) where T
    
    centers = Vec2D{T}[]
    min_dist = 2 * R * margin_factor

    x_min = -box_side/2 + R
    x_max =  box_side/2 - R
    y_min = -box_side/2 + R
    y_max =  box_side/2 - R

    max_attempts = 10000
    attempt = 0
    while length(centers) < n_droplets
        c = Vec2D(
            x_min + rand(T)*(x_max - x_min),
            y_min + rand(T)*(y_max - y_min),
        )
        if all(norm(c - c_old) > min_dist for c_old in centers)
            push!(centers, c)
        end

        attempt += 1

        if attempt == max_attempts
            attempt = 0
            n_droplets -= 1
        end
    end

    return centers
end

# Generate oil positions in several identical circular droplets
function generate_multi_droplet(n_oil::Int,
                                    centers::Vector{Vec2D{T}},
                                    droplet_area::T) where T
    n_droplets = length(centers)
    R = sqrt(droplet_area / pi)  # radius of each droplet

    oil = Vec2D{T}[]

    # distribute particle counts as evenly as possible
    base = div(n_oil, n_droplets)
    extra = rem(n_oil, n_droplets)
    n_per = [base + (k <= extra ? 1 : 0) for k in 1:n_droplets]

    for (k, center) in enumerate(centers)
        for i in 1:n_per[k]
            # uniform sampling in a disk: r = R*sqrt(u), theta in [0,2pi]
            r = R * sqrt(rand(T))
            theta = 2 * T(pi) * rand(T)
            push!(oil, Vec2D(
                center.x + r * cos(theta),
                center.y + r * sin(theta),
            ))
        end
    end

    return oil, R
end

# Generate water particles outside all droplets
function generate_outside_droplets(n_water::Int,
                                         centers::Vector{Vec2D{T}},
                                         R::T,
                                         box_side::T) where T
    water = Vec2D{T}[]
    x_min, x_max = -box_side/2, box_side/2
    y_min, y_max = -box_side/2, box_side/2

    while length(water) < n_water
        p = random_vec(Vec2D{T}, (x_min, x_max))
        # keep p only if it is outside every droplet
        if all(norm(p - c) > R for c in centers)
            push!(water, p)
        end
    end

    return water
end

for volume_fraction_oil in [0.2, 0.4, 0.6, 0.8, 0.95]

    volume_oil   = volume_fraction_oil * box_side^2
    volume_water = (1.0 - volume_fraction_oil) * box_side^2
    n_oil::Int   = ceil(volume_oil * density_number)
    n_water::Int = ceil((box_side^2 - volume_oil) * density_number)
    n_total      = n_oil + n_water

    n_droplets = 4
    
    x0_oil, x0_water = if volume_fraction_oil <= 0.5
        droplet_area = volume_oil / n_droplets          # area of ONE droplet
        R_droplet    = sqrt(droplet_area / pi)           # same R used for centers

        # pick droplet centers
        centers = generate_droplet_centers(
            n_droplets,
            box_side,
            R_droplet;
            margin_factor = 1.5,
        )

        # generate oil inside droplets
        x0_oil, R_used = generate_multi_droplet(
            n_oil,
            centers,
            droplet_area,
        )

        # generate water outside droplets
        x0_water = generate_outside_droplets(
            n_water,
            centers,
            R_used,
            box_side,
        )

        x0_oil, x0_water
    else
        droplet_area = volume_water / n_droplets          # area of ONE droplet
        R_droplet    = sqrt(droplet_area / pi)           # same R used for centers

        # pick droplet centers
        centers = generate_droplet_centers(
            n_droplets,
            box_side,
            R_droplet;
            margin_factor = 1.5,
        )

        # generate oil inside droplets
        x0_water, R_used = generate_multi_droplet(
            n_water,
            centers,
            droplet_area,
        )

        # generate water outside droplets
        x0_oil = generate_outside_droplets(
            n_oil,
            centers,
            R_used,
            box_side,
        )

        x0_oil, x0_water
    end

    x0_emulsion = vcat(x0_oil, x0_water)
    box_emulsion = Box([box_side,box_side],cutoff)
    cl_emulsion = CellList(x0_emulsion,box_emulsion)

    t_emulsion = @elapsed trajectory_emulsion = md_Verlet((
        x0 = x0_emulsion, 
        v0 = [random_vec(Vec2D{Float64},(-0.1,0.1)) for _ in 1:n_total ], 
        mass = [ 1.0 for _ in 1:n_total ],
        dt = dt,
        box_side = box_side,
        nsteps = nsteps,
        isave = nsteps/100,
        forces! = (f,x) -> forces_emulsion!(f,x,box_emulsion,box_side,cl_emulsion,n_oil,n_water,f_emulsion_pair!)
    )...)
    println("Time taken for emulsion simulation: ",t_emulsion," seconds")

    # Analyzing emulsion stability 
    println("Number of oil particles at the beginning: ",n_oil,", Number of water particles at the beginning: ",n_water, ", Total particles: ",n_total)
    oil_out = 0
    water_out = 0
    for fram in trajectory_emulsion
        for p in fram[1:n_oil]
            if abs(p.x) > box_side/2 || abs(p.y) > box_side/2
                oil_out += 1
            end
        end
        for p in fram[n_oil+1:end]
            if abs(p.x) > box_side/2 || abs(p.y) > box_side/2
                water_out += 1
            end
        end
    end
    println("Number of oil particles out of bounds: ",oil_out)
    println("Number of water particles out of bounds: ",water_out)
    println("Percent of particles sent to infinity: ", (oil_out + water_out) / n_total * 100, "%")

    # Visualizing the emulsion trajectory
    anim_emulsion = @animate for frame in trajectory_emulsion
        scatter(
            [ p.x for p in frame[1:n_oil] ],
            [ p.y for p in frame[1:n_oil] ],
            xlim=(-box_side/2,box_side/2), ylim=(-box_side/2,box_side/2),
            title="Emulsion MD Simulation",
            xlabel="X Position", ylabel="Y Position",
            markersize=2,
            color=:orange,
        )
        scatter!(
            [ p.x for p in frame[n_oil+1:end] ],
            [ p.y for p in frame[n_oil+1:end] ],
            markersize=2,
            color=:blue,
        )
    end
    # Save as GIF
    gif_path = joinpath(gif_dir, "trajectory_emulsion_vf_$(volume_fraction_oil)_$(nsteps)_$(dt).gif")
    gif(anim_emulsion, gif_path, fps=20)
end