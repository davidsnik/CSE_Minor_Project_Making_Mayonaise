using StaticArrays
import CellListMap: Box, CellList, UpdateCellList!, map_pairwise!
using LinearAlgebra: norm
using GLMakie  # Make sure you have this loaded
using Plots: scatter, scatter!, @animate, gif

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

function f_emulsion_pair_shear!(i, j,
                                x::T, y::T,
                                cutoff, side, a_ij,
                                f) where T

    Δ = y - x                  # works for SVector and Vec2D
    Δx = wrap_x(Δ[1], side)    # minimum image only in x
    Δy = wrap_x(Δ[2], side)    # minimum image only in y

    # construct displacement with the SAME type as x,y (T can be Vec2D or SVector)
    Δv = T(Δx, Δy)

    d = norm(Δv)
    in_repulsive = a_ij == a_oil_oil

    if d > cutoff
        fₓ = zero(T)
    elseif d < 1e-12 && in_repulsive
        fₓ = 3000 * (1 - d/cutoff) * (Δv / d)
    else
        fₓ = a_ij * (1 - d/cutoff) * (Δv / d)
    end

    f[i] += fₓ
    
    return f
end
# Defnition of a position vector for each particle
struct Vec2D{T} <: FieldVector{2,T} # A 2D vector with components of type T
    x::T
    y::T
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


function md_Verlet_walls(x0::Vector{T}, v0::Vector{T}, mass,
                         dt, box_side, cutoff, nsteps, isave, 
                         bulk_ids) where T

    # Copy initial state
    x = copy(x0)
    v = copy(v0)
    f = similar(x0)

    for i in bulk_ids
        for j in bulk_ids
            if i!= j
                if i < 3 && j<3
                    a_ij = a_oil_oil
                elseif i >=3 && j>=3
                    a_ij = a_water_water
                else
                    a_ij = a_oil_water
                end
                # compute forces between bulk particles
                f_emulsion_pair_shear!(i, j, x[i], x[j], cutoff, box_side, a_ij, f)
            end
        end
    end

    trajectory       = Vector{Vector{T}}(undef, 0)


    push!(trajectory, copy(x))

    # Main loop
    for step in 1:nsteps
        # --- half-step velocity update (bulk only) ---
        @inbounds for i in bulk_ids
            v[i] += 0.5 * (f[i] / mass[i]) * dt
        end

        # --- full position update for bulk ---
        @inbounds for i in bulk_ids
            x[i] += v[i] * dt
            x[i] = Vec2D(wrap_x(x[i].x, box_side), wrap_x(x[i].y, box_side))
        end



        # --- compute new forces at updated positions ---
        for i in bulk_ids
            for j in bulk_ids
                if i!= j
                    if i < 3 && j<3
                        a_ij = a_oil_oil
                    elseif i >=3 && j>=3
                        a_ij = a_water_water
                    else
                        a_ij = a_oil_water
                    end
                    # compute forces between bulk particles
                    f_emulsion_pair_shear!(i, j, x[i], x[j], cutoff, box_side, a_ij, f)
                end
            end
        end
        

        # --- second half-step velocity update (bulk only) ---
        @inbounds for i in bulk_ids
            v[i] += 0.5 * (f[i] / mass[i]) * dt
        end

        
        if step % isave == 0
            push!(trajectory, copy(x))
           
        end
    end


    return trajectory
end



const a_water_water = 25.0
const a_oil_oil     = 25.0
const a_oil_water   = 80.0

tiny_box_side = 1.0
tiny_cutoff   = 3.0
n_oil = 2
n_water = 0
n_total = n_oil + n_water
x0_oil = [Vec2D(-0.1, -0.1), Vec2D(0.1, 0.1)]
x0_water = [Vec2D(-0.1, 0.1), Vec2D(0.1, -0.1)]
        
x0_emulsion = vcat(x0_oil, x0_water)
        
    

isave = 1
v0_all = [Vec2D(0.0, 0.0) for _ in 1:n_total]
bulk_ids    = 1:(n_oil + n_water)
dt = 0.001
nsteps = 1000
        
mass_all = [1.0 for _ in 1:n_total]
        
t_emulsion = @elapsed trajectory_emulsion = md_Verlet_walls(
    x0_emulsion, v0_all, mass_all,
    dt, tiny_box_side, tiny_cutoff, nsteps, isave, bulk_ids
)

println("Time taken for emulsion simulation: ",t_emulsion," seconds")
     
        
anim_emulsion = @animate for frame in trajectory_emulsion
        # Oil particles
        scatter(
            [p.x for p in frame[1:n_oil]],
            [p.y for p in frame[1:n_oil]],
            xlim = (-tiny_box_side/2, tiny_box_side/2),
            ylim = (-tiny_box_side/2, tiny_box_side/2),
            title = "4 particles MD Simulation",
            xlabel = "X Position",
            ylabel = "Y Position",
            markersize = 2,
            color = [:orange, :blue],
        )

        # Water particles
        scatter!(
            [p.x for p in frame[n_oil+1 : n_oil+n_water]],
            [p.y for p in frame[n_oil+1 : n_oil+n_water]],
            markersize = 2,
            color = :blue,
        )
    
    end
    
gif(anim_emulsion, "emulsion_tiny_positive_velocity_verlet_dt_$dt.mp4", fps = 10)
        


