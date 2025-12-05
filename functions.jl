# Place n_droplets centers inside the box, separated by at least ~2R
function generate_droplet_centers(n_droplets::Int,
                                  box_side::T,
                                  R::T;
                                  margin_factor::T = 1.5) where T
    
    centers = SVector{2,T}[]
    min_dist = 2 * R * margin_factor

    x_min = -box_side/2 + R
    x_max =  box_side/2 - R
    y_min = -box_side/2 + (R + cutoff)
    y_max =  box_side/2 - (R + cutoff)

    max_attempts = 10000
    attempt = 0
    while length(centers) < n_droplets
        c = SVector{2,T}(
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
                                    centers::Vector{SVector{2,T}},
                                    droplet_area::T) where T
    n_droplets = length(centers)
    R = sqrt(droplet_area / pi)  # radius of each droplet

    oil = SVector{2,T}[]

    # distribute particle counts as evenly as possible
    base = div(n_oil, n_droplets)
    extra = rem(n_oil, n_droplets)
    n_per = [base + (k <= extra ? 1 : 0) for k in 1:n_droplets]

    for (k, center) in enumerate(centers)
        for i in 1:n_per[k]
            # uniform sampling in a disk: r = R*sqrt(u), theta in [0,2pi]
            r = R * sqrt(rand(T))
            theta = 2 * T(pi) * rand(T)
            push!(oil, SVector{2,T}(
                center[1] + r * cos(theta),
                center[2] + r * sin(theta),
            ))
        end
    end

    return oil, R
end

# Generate water particles outside all droplets
# function generate_outside_droplets(n_water::Int,
#                                    centers::Vector{SVector{2,T}},
#                                    R::T,
#                                    box_side::T,
#                                    walls::Vector{SVector{2,T}} = SVector{2,T}[];
#                                    wall_buffer::T = 0*cutoff) where T
function generate_outside_droplets(n_water::Int,
                                   centers::Vector{SVector{2,T}},
                                   R::T,
                                   box_side::T) where T
    wall_buffer = 0
    water = SVector{2,T}[]
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
        p = SVector{2,T}(px, py)

        # outside every droplet?
        inside_any_droplet = any(norm(p - c) <= R for c in centers)

        # not too close to any wall bead?
        close_to_wall = false
        # @inbounds for w in walls
        #     dx = wrap_x(p.x - w.x, box_side)
        #     dy = p.y - w.y
        #     if dx*dx + dy*dy < wall_buffer2
        #         close_to_wall = true
        #         break
        #     end
        # end

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
        VecType::Type{SVector{2,T}},
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