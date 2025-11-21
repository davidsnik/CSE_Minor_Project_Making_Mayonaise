using Plots
using CellListMap # package that divides the domain into cells and only computes interactions between particles in neighboring cells
import CellListMap: Box, CellList, UpdateCellList!, map_pairwise!
using StaticArrays
import LinearAlgebra: norm
using Statistics
import Statistics: mean, std


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

# Definig the directory for all the images and gifs 
gif_dir = joinpath(@__DIR__, "gif_chang_MD")
isdir(gif_dir) || mkpath(gif_dir)

# Defining simulation parameters for the emulsion MD simulation (taken from the paper)
const box_side = 32.2
const density_number = 3
const nsteps = 10^4
const dt = 0.01
const volume_fraction_oil = 0.8

volume_oil = volume_fraction_oil * box_side^2

n_oil::Int = ceil(volume_oil * density_number)
n_water::Int = ceil((box_side^2 - volume_oil) * density_number)
n_total = n_oil + n_water

#initializing positions of oil and water particles randomly within the box
x0_oil = [ random_vec(Vec2D{Float64},(-box_side/2,box_side/2)) for _ in 1:n_oil ] # size 2xn_oil
x0_water = [ random_vec(Vec2D{Float64},(-box_side/2,box_side/2)) for _ in 1:n_water ] # size 2xn_water
x0_emulsion = vcat(x0_oil,x0_water) # size 2x(n_oil+n_water)
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

#= Uncomment if needed but the specific code for that is in the "Chang_paper_MDS.jl" 

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
=#


# for every time frame, assign a cluster ID to every water particle based on proximity
using Clustering
clustered_frames = [] # final size of this will be n_particlesxn_frames (idx for every particle at every frame)
for (frame_index, frame) in enumerate(trajectory_emulsion)
    
    water_positions = frame[n_oil+1:end]
    water_coords = hcat([ [p.x, p.y] for p in water_positions ]) # convert to Nx2 matrix
    # Perform DBSCAN clustering
    clustering_result = dbscan(water_coords, 1.5, min_neighbors=3)
    push!(clustered_frames, clustering_result.assignments)

end

# make a random choice of np water particles and track their cluster ID over time
num_rnd_p = 5
idx = rand(1:n_water, num_rnd_p)
# Plot cluster ID assigned to each particle over time
using StatsPlots
cluster_plot = @animate for (frame_idx, assignments) in enumerate(clustered_frames)
    scatter(
        1:num_rnd_p,
        assignments[idx], # chose random 20 water particles for clarity
        xlabel="Water Particle Index",
        ylabel="Cluster ID",
        title="DBSCAN Clustering of Water Particles at Frame $(frame_idx)",
        ylim=(-1, maximum(assignments)+1),
        markersize=3,
    )
end
gif(cluster_plot, joinpath(gif_dir, "water_particle_clustering.gif"), fps=1)

# Bar plot of cluster sizes over time
cluster_sizes = @animate for (frame_idx, assignments) in enumerate(clustered_frames)
    # Compute sizes per cluster (DBSCAN uses 0 for noise)
    cluster_ids = sort(unique(assignments))
    sizes = [count(==(cid), assignments) for cid in cluster_ids]
    labels = [cid == 0 ? "Noise" : "C$(cid)" for cid in cluster_ids]
    bar(
        labels,
        sizes,
        xlabel = "Cluster",
        ylabel = "Number of water particles",
        title = "DBSCAN Cluster Sizes (Frame $(frame_idx))",
        legend = false,
    )
end
gif(cluster_sizes, joinpath(gif_dir, "cluster_sizes.gif"), fps=1)


# plot cluster id over time for 5 particles (every particle different color)
num_rnd_p = 100
idx = rand(1:n_water, num_rnd_p)
num_frames = length(clustered_frames)

X = [ trajectory_emulsion[t][n_oil + i].x for i in idx, t in 1:num_frames ]  # num_rnd_p × num_frames
Y = [ trajectory_emulsion[t][n_oil + i].y for i in idx, t in 1:num_frames ]  # num_rnd_p × num_frames

println(size(X))  # (5, num_frames)

# If you still want per-frame mean/std to plot with ribbon:
mean_x = vec(Statistics.mean(X; dims=2))  # length = num_frames
mean_y = vec(Statistics.mean(Y; dims=2))
std_x  = vec(Statistics.std(X; dims=2))
std_y  = vec(Statistics.std(Y; dims=2))
println(size(mean_x))  # (num_frames,)
p1 = plot(
    1:num_rnd_p,
    mean_x,
    ribbon=std_x,
    xlabel="Frame Index",
    ylabel="Mean X Position",
    title="Mean X Position of Selected Water Particles Over Time",
    label="Mean X Position",
    legend=:topright,
    label_p1 = permutedims(string.("particle ", idx))  # optional labels
)
p2 = plot(
    1:num_rnd_p,
    mean_y,
    ribbon=std_y,
    xlabel="Frame Index",
    ylabel="Mean Y Position",
    title="Mean Y Position of Selected Water Particles Over Time",
    label="Mean Y Position",
    legend=:topright,
    label_p2 = permutedims(string.("particle ", idx))  # optional labels
)
plot(p1, p2, layout=(2,1))
savefig(joinpath(gif_dir, "water_particle_mean_positions_over_time.png"))

# animate tajectory of particles starting in one (given) cluster
# cluster_ids = [1,4,8,12,15,18,20] 
cluster_ids = [8] 
for cluster_id in cluster_ids
    println("Animating trajectory for cluster ID: ", cluster_id)
    idx_in_cluster = findall(==(cluster_id), clustered_frames[1]) # indices of water particles in the chosen cluster at first frame
    find_initial_particles = true
    anim_cluster = @animate for (frame_idx, assignments) in enumerate(clustered_frames)
        
        water_positions = trajectory_emulsion[frame_idx][n_oil+1:end]
        cluster_positions = [ water_positions[i] for i in idx_in_cluster ]
        scatter(
            [ p.x for p in cluster_positions ],
            [ p.y for p in cluster_positions ],
            xlim=(-box_side/2,box_side/2), ylim=(-box_side/2,box_side/2),
            title="Cluster $(cluster_id) at Frame $(frame_idx)",
            xlabel="X Position", ylabel="Y Position",
            markersize=4,
            color=:blue,
        )
    end
    gif(anim_cluster, joinpath(gif_dir, "cluster_$(cluster_id)_trajectory.gif"), fps=3)
end

# make a gif of a trajectory of water droplets with each cluster colored on different color
possible_colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray, :cyan, :magenta]
anim_emulsion = @animate for (frame_idx, frame) in enumerate(trajectory_emulsion)
    # color water particles based on cluster ID
    plt = scatter(legend=false)
    assignments = clustered_frames[frame_idx]
    for cluster_id in unique(assignments)
        if cluster_id == 0
            cluster_color = :black # noise points
        else
            cluster_color = possible_colors[mod(cluster_id - 1, length(possible_colors)) + 1]
        end
        cluster_positions = [ frame[n_oil + i] for i in 1:n_water if assignments[i] == cluster_id ]
        scatter!(
            [ p.x for p in cluster_positions ],
            [ p.y for p in cluster_positions ],
            markersize=2,
            color=cluster_color,
        )
    end
    plt
end

# Save as GIF
gif_path = joinpath(gif_dir, "trajectory_emulsion_clusters_vf_$(volume_fraction_oil)_$(nsteps)_$(dt).gif")
gif(anim_emulsion, gif_path, fps=5)