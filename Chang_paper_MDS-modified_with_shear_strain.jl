using CellListMap # package that divides the domain into cells and only computes interactions between particles in neighboring cells
import CellListMap: Box, CellList, UpdateCellList!, map_pairwise!
using StaticArrays
import LinearAlgebra: norm
using Statistics: mean

output_plots = true

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

# Periodic only in x (no wrapping in y)
@inline function wrap_x(dx::T, side::T) where T
    dx = rem(dx, side)
    if dx >= side/2
        dx -= side
    elseif dx < -side/2
        dx += side
    end
    return dx
end

# Apply x-periodic BC only to a subset of particles (the bulk)
function apply_x_periodic_bulk!(x::Vector{Vec2D{T}}, side::T, bulk_ids) where T
    @inbounds for i in bulk_ids
        xi = x[i]
        x[i] = Vec2D(wrap_x(xi.x, side), xi.y)
    end
end

# Compute shear stress sigma_xy from positions x, velocities v, and forces f
function shear_stress_xy(x::Vector{Vec2D{T}},
                         v::Vector{Vec2D{T}},
                         f::Vector{Vec2D{T}},
                         mass,
                         box_side::T) where T
    A = box_side^2        # 2D "volume" = area

    sigma_kin = 0.0           # kinetic contribution
    sigma_vir = 0.0           # virial (configurational) contribution

    @inbounds for i in eachindex(x)
        # kinetic part: m v_x v_y
        sigma_kin += mass[i] * v[i].x * v[i].y
        # virial part: x_x * F_y
        sigma_vir += x[i].x * f[i].y
    end

    sigma_xy = (sigma_kin + sigma_vir) / A
    return sigma_xy
end

function confine_y_bulk!(x::Vector{Vec2D{T}}, v::Vector{Vec2D{T}},
                         y_min::T, y_max::T, bulk_ids) where T
    @inbounds for i in bulk_ids
        xi = x[i]
        yi = xi.y
        if yi > y_max
            # reflect position
            new_y = 2*y_max - yi
            x[i] = Vec2D(xi.x, new_y)
            # reverse normal velocity (bounce)
            v[i] = Vec2D(v[i].x, -v[i].y)
        elseif yi < y_min
            new_y = 2*y_min - yi
            x[i] = Vec2D(xi.x, new_y)
            v[i] = Vec2D(v[i].x, -v[i].y)
        end
    end
end

function md_Verlet_walls(x0::Vector{T}, v0::Vector{T}, mass,
                         dt, box_side, nsteps, isave, forces!,
                         bulk_ids, topwall_ids, botwall_ids;
                         U_top::Real, U_bot::Real = 0.0) where T

    # Copy initial state
    x = copy(x0)
    v = copy(v0)
    f = similar(x0)

    # Initial forces
    forces!(f, x)

    trajectory       = Vector{Vector{T}}(undef, 0)
    sigma_xy_series  = Float64[]

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
        end

        # confine bulk between the walls (just inside them)
        y_min = -box_side/2 + 0.5*cutoff   # or use same y_offset as wall generation
        y_max =  box_side/2 - 0.5*cutoff
        confine_y_bulk!(x, v, y_min, y_max, bulk_ids)

        # --- apply periodic BC in x only to bulk ---
        apply_x_periodic_bulk!(x, box_side, bulk_ids)

        # --- move walls with prescribed velocities (and wrap in x) ---
        @inbounds for i in topwall_ids
            # prescribe horizontal velocity only
            v[i] = Vec2D(U_top, 0.0)
            xi = x[i]
            x_new_x = wrap_x(xi.x + U_top * dt, box_side)
            # keep the original / jittered y position
            x[i] = Vec2D(x_new_x, xi.y)
        end

        @inbounds for i in botwall_ids
            v[i] = Vec2D(U_bot, 0.0)
            xi = x[i]
            x_new_x = wrap_x(xi.x + U_bot * dt, box_side)
            x[i] = Vec2D(x_new_x, xi.y)
        end

        # --- compute new forces at updated positions ---
        forces!(f, x)

        # --- second half-step velocity update (bulk only) ---
        @inbounds for i in bulk_ids
            v[i] += 0.5 * (f[i] / mass[i]) * dt
        end

        # --- compute shear stress (you can restrict to bulk if you want) ---
        sigma_xy = shear_stress_xy(x, v, f, mass, box_side)

        if step % isave == 0
            push!(trajectory, copy(x))
            push!(sigma_xy_series, sigma_xy)
        end
    end

    # Time-average over last 30% of saved samples
    if !isempty(sigma_xy_series)
        i0 = ceil(Int, 0.7 * length(sigma_xy_series))
        return trajectory, mean(sigma_xy_series[i0:end])
    else
        return trajectory, NaN
    end
end

# function that calculates pairwise forces between oil and water particles in an emulsion
function forces_emulsion!(f::Vector{T}, x,
                          box::Box, side, cl::CellList,
                          n_oil, n_water, n_wall,
                          fpair::F) where {T,F}
    fill!(f,zero(T))
    cl = UpdateCellList!(x,box,cl,parallel=false)
    map_pairwise!(
        (x,y,i,j,d2,f) -> begin
            # index ranges
            i_oil   = i <= n_oil
            j_oil   = j <= n_oil
            i_water = (n_oil < i <= n_oil + n_water)
            j_water = (n_oil < j <= n_oil + n_water)
            # walls are the rest
            i_wall  = !(i_oil || i_water)
            j_wall  = !(j_oil || j_water)

            if i_oil && j_oil
                a = a_oil_oil
            elseif i_water && j_water
                a = a_water_water
            elseif i_wall && j_wall
                a = a_wall_wall
            elseif (i_oil && j_water) || (i_water && j_oil)
                a = a_oil_water               # oil–water
            elseif (i_oil && j_wall) || (i_wall && j_oil)
                a = a_wall_oil         # oil–wall
            else # (i_water && j_wall) || (i_wall && j_water)
                a = a_wall_water       # water–wall
            end

            fpair(i, j, x, y, box.cutoff, side, a, f)
        end,
        f, box, cl,
        parallel=true
    )
    return f
end

function f_emulsion_pair_shear!(i, j,
                                x::T, y::T,
                                cutoff, side, a_ij,
                                f) where T

    Δ = y - x                  # works for SVector and Vec2D
    Δx = wrap_x(Δ[1], side)    # minimum image only in x
    Δy = Δ[2]                  # direct distance in y

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
    f[j] -= fₓ
    return f
end

# Function to plot emulsion interaction forces (to see how they behave with distance)
function femulsion_plot(d,cutoff,a_ij)
    in_repulsive = a_ij == a_oil_oil
    if d > cutoff
        fₓ = 0
    elseif d < 1e-12 && in_repulsive
        fₓ = 3000*(1-d/cutoff)
    else
        fₓ = a_ij*(1-d/cutoff)
    end
    return fₓ
end
# Place n_droplets centers inside the box, separated by at least ~2R
function generate_droplet_centers(n_droplets::Int,
                                  box_side::T,
                                  R::T;
                                  margin_factor::T = 1.5) where T
    
    centers = Vec2D{T}[]
    min_dist = 2 * R * margin_factor

    x_min = -box_side/2 + R
    x_max =  box_side/2 - R
    y_min = -box_side/2 + (R + cutoff)
    y_max =  box_side/2 - (R + cutoff)

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
                                   box_side::T,
                                   walls::Vector{Vec2D{T}} = Vec2D{T}[];
                                   wall_buffer::T = 0.5*cutoff) where T
    water = Vec2D{T}[]
    x_min, x_max = -box_side/2, box_side/2
    # keep away from walls in y by a small buffer as well
    y_min, y_max = -box_side/2 + wall_buffer, box_side/2 - wall_buffer
    wall_buffer2 = wall_buffer^2

    max_attempts = 2_000_000
    attempts = 0

    while length(water) < n_water && attempts < max_attempts
        attempts += 1

        # sample x uniformly
        px = x_min + rand(T)*(x_max - x_min)
        # sample y in interior
        py = y_min + rand(T)*(y_max - y_min)
        p = Vec2D(px, py)

        # outside every droplet?
        inside_any_droplet = any(norm(p - c) <= R for c in centers)

        # not too close to any wall bead?
        close_to_wall = false
        @inbounds for w in walls
            dx = wrap_x(p.x - w.x, box_side)
            dy = p.y - w.y
            if dx*dx + dy*dy < wall_buffer2
                close_to_wall = true
                break
            end
        end

        if !inside_any_droplet && !close_to_wall
            push!(water, p)
        end

        # optional: lightweight progress print every so often
        if attempts % 200_000 == 0
            println("Placed $(length(water)) / $(n_water) water after $attempts attempts")
        end
    end

    if length(water) < n_water
        println("WARNING: only placed $(length(water)) of $(n_water) water particles. ",
                "Relax constraints (wall_buffer, droplet radius, density) or reduce n_water.")
    end

    return water
end


function make_rough_wall_particles(
        VecType::Type{Vec2D{T}},
        box_side::T,
        n_per_wall::Int;
        y_offset::T = 0.0,
        y_amp::T = 0.5*cutoff,   # vertical roughness amplitude
    ) where T

    dx = box_side / n_per_wall
    x_start = -box_side/2 + dx/2

    x_top_vec = VecType[]
    x_bot_vec = VecType[]

    # Roughness parameters
    jitter_amp = 0.3 * dx       # horizontal jitter in x

    for k in 0:(n_per_wall-1)
        xk = x_start + k*dx

        # Top wall: random x and y slightly below +box_side/2
        x_top_pos = xk + (2*rand(T) - one(T)) * jitter_amp
        x_top_pos = wrap_x(x_top_pos, box_side)
        y_top_pos = box_side/2 - y_offset - rand(T)*y_amp

        # Bottom wall: random x and y slightly above -box_side/2
        x_bot_pos = xk + (2*rand(T) - one(T)) * jitter_amp
        x_bot_pos = wrap_x(x_bot_pos, box_side)
        y_bot_pos = -box_side/2 + y_offset + rand(T)*y_amp

        push!(x_top_vec, VecType(x_top_pos, y_top_pos))
        push!(x_bot_vec, VecType(x_bot_pos, y_bot_pos))
    end

    return x_top_vec, x_bot_vec
end

function wall_velocity_from_shear(applied_shear, nsteps, dt, box_side; cutoff_local = cutoff)
    H = box_side - cutoff_local      # effective gap between walls
    T = nsteps * dt                  # total simulation time
    return applied_shear * H / T
end


# TODO these values have to be tweak
# defining repulsion parameters
cutoff = 1.0
const a_water_water = -25.0
const a_oil_oil     = -25.0
const a_oil_water   = -80.0

const a_wall_wall   = -25.0
const a_wall_water  = -25.0
const a_wall_oil    = -80.0

# Plotting the emulsion forces between particles
using Plots
d_values = 0:0.01:1.5*cutoff
f_oo = [ femulsion_plot(d,cutoff,a_oil_oil) for d in d_values ]
f_ww = [ femulsion_plot(d,cutoff,a_water_water) for d in d_values ]
f_ow = [ femulsion_plot(d,cutoff,a_oil_water) for d in d_values ]

if output_plots
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
    savefig(joinpath(gif_dir, "emulsion_forces_wall.png"))
end


# Defining simulation parameters for the emulsion MD simulation
box_side = 32.2
density_number = 3.0
nsteps = 10^4
dt = 0.001

for volume_fraction_oil in [0.3]
    for applied_shear in [0]
        volume_oil   = volume_fraction_oil * box_side^2
        volume_water = (1.0 - volume_fraction_oil) * box_side^2
        n_oil::Int   = ceil(volume_oil * density_number)
        n_water::Int = ceil((box_side^2 - volume_oil) * density_number)
    
        n_droplets = 4
    
        n_per_wall = 400
        x_top, x_bot = make_rough_wall_particles(Vec2D{Float64}, box_side, n_per_wall;
                                                 y_offset = 0.0)
        walls = vcat(x_top, x_bot)
    
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
                walls,             # <- avoid walls too
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
                walls,             # <- avoid walls too
            )
    
            x0_oil, x0_water
        end
    
        x0_emulsion = vcat(x0_oil, x0_water)
        x0_all      = vcat(x0_emulsion, walls)
    
        box_emulsion = Box([box_side, box_side], cutoff)
        cl_emulsion  = CellList(x0_all, box_emulsion)
    
        n_bulk   = length(x0_emulsion)   # = n_oil + n_water
        n_wall   = length(walls)
        n_total  = n_bulk + n_wall
        n_topwall  = length(x_top)
        n_botwall  = length(x_bot)
        n_total    = length(x0_all)
    
        bulk_ids    = 1:n_bulk
        topwall_ids = (n_bulk+1):(n_bulk+n_topwall)
        botwall_ids = (n_bulk+n_topwall+1):(n_total)
    
        isave = 100
    
        v0_all = [random_vec(Vec2D{Float64},(-0.1,0.1)) for _ in 1:n_bulk]
        append!(v0_all, [Vec2D(0.0, 0.0) for _ in 1:(n_topwall+n_botwall)])
    
        mass_all = [1.0 for _ in 1:n_total]
        U_wall   = wall_velocity_from_shear(applied_shear, nsteps, dt, box_side)
    
        t_emulsion = @elapsed trajectory_emulsion, sigma_xy = md_Verlet_walls(
            x0_all, v0_all, mass_all,
            dt, box_side, nsteps, isave,
            (f,x) -> forces_emulsion!(f, x, box_emulsion, box_side,
                              cl_emulsion, n_oil, n_water, n_wall,
                              f_emulsion_pair_shear!),
            bulk_ids, topwall_ids, botwall_ids;
            U_top = U_wall, U_bot = 0.0,
        )
    
        println("Time taken for emulsion simulation: ",t_emulsion," seconds")
        println("Sigma_xy: ", sigma_xy)
    
        # Analyzing emulsion stability 
        println("Number of oil particles at the beginning: ",n_oil,", Number of water particles at the beginning: ",n_water, ", Total particles: ",n_total)
        println("Number of oil particles at the beginning: ", n_oil,
            ", Number of water particles at the beginning: ", n_water,
            ", Total particles: ", n_total)
    
        oil_oob = falses(n_oil)
        water_oob = falses(n_water)
    
        for fram in trajectory_emulsion
            # oil
            for (k,p) in enumerate(fram[1:n_oil])
                if abs(p.x) > box_side/2 || abs(p.y) > box_side/2
                    oil_oob[k] = true
                end
            end
            # water ONLY (exclude walls)
            for (k,p) in enumerate(fram[n_oil+1 : n_oil+n_water])
                if abs(p.x) > box_side/2 || abs(p.y) > box_side/2
                    water_oob[k] = true
                end
            end
        end
    
        n_oil_oob   = count(oil_oob)
        n_water_oob = count(water_oob)
    
        println("Unique oil particles ever out of bounds: ", n_oil_oob)
        println("Unique water particles ever out of bounds: ", n_water_oob)
        println("Percent of particles ever out of bounds: ",
                (n_oil_oob + n_water_oob) / n_total * 100, "%")
    
        if output_plots
            anim_emulsion = @animate for frame in trajectory_emulsion
                # Oil particles
                scatter(
                    [p.x for p in frame[1:n_oil]],
                    [p.y for p in frame[1:n_oil]],
                    xlim = (-box_side/2, box_side/2),
                    ylim = (-box_side/2, box_side/2),
                    title = "Emulsion MD Simulation, vf = $(volume_fraction_oil)",
                    xlabel = "X Position",
                    ylabel = "Y Position",
                    markersize = 2,
                    color = :orange,
                )
    
                # Water particles
                scatter!(
                    [p.x for p in frame[n_oil+1 : n_oil+n_water]],
                    [p.y for p in frame[n_oil+1 : n_oil+n_water]],
                    markersize = 2,
                    color = :blue,
                )
    
                # Wall particles (last chunk)
                scatter!(
                    [p.x for p in frame[n_oil+n_water+1 : end]],
                    [p.y for p in frame[n_oil+n_water+1 : end]],
                    markersize = 3,
                    color = :red,
                )
            end
    
            gif_path = joinpath(gif_dir, "trajectory_emulsion_vf_$(volume_fraction_oil)_$(nsteps)_$(dt)_wall.gif")
            gif(anim_emulsion, gif_path, fps = isave/dt)
        end
    end
end
