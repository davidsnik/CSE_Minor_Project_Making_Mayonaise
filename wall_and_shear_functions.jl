function make_rough_wall_particles(
        VecType::Type{SVector{2,T}},
        box_side::T,
        n_per_wall::Int;
        y_offset::T = 0.0,
        y_amp::T = 0.5*cutoff,   # vertical roughness amplitude
    ) where T

    dx = box_side / n_per_wall
    x_start = dx/2

    x_top_vec = VecType[]
    x_bot_vec = VecType[]

    # Roughness parameters
    jitter_amp = 0.3 * dx       # horizontal jitter in x

    for k in 0:(n_per_wall-1)
        xk = x_start + k*dx

        # Top wall: random x and y slightly below box_side
        x_top_pos = xk + (2*rand(T) - one(T)) * jitter_amp
        x_top_pos = wrap_x(x_top_pos, box_side)
        y_top_pos = box_side - y_offset - rand(T)*y_amp

        # Bottom wall: random x and y slightly above 0
        x_bot_pos = xk + (2*rand(T) - one(T)) * jitter_amp
        x_bot_pos = wrap_x(x_bot_pos, box_side)
        y_bot_pos = y_offset + rand(T)*y_amp

        push!(x_top_vec, VecType(x_top_pos, y_top_pos))
        push!(x_bot_vec, VecType(x_bot_pos, y_bot_pos))
    end

    return x_top_vec, x_bot_vec
end

# --------------------------- Shear profiles --------------------------- #
struct ShearProfile{F,G}
    gamma::F        # gamma(t): shear strain at time t
    gamma_rate::G   # gamma_dot(t): shear rate at time t
end

ShearProfile(gamma::F, gamma_rate::G) where {F<:Function,G<:Function} = ShearProfile{F,G}(gamma, gamma_rate)

# Build a shear profile. Supply any gamma_fn you like, or pick a preset `kind`.
# The shear rate is always computed numerically from gamma_fn for consistency.
function make_shear_profile(; gamma_fn::Union{Nothing,Function}=nothing,
                            kind::Symbol = :linear_ramp,
                            gamma_final::Real = 0.5,
                            ramp_time::Real = 1.0,
                            gamma_rate::Real = 0.0,
                            amplitude::Real = 0.5,
                            frequency::Real = 1.0,
                            phase::Real = 0.0,
                            offset::Real = 0.0,
                            gamma0::Real = 0.0)
    if gamma_fn === nothing
        if kind === :linear_ramp
            slope = ramp_time == 0 ? zero(gamma_final) : (gamma_final - gamma0) / ramp_time
            gamma_fn = t -> t <= ramp_time ? gamma0 + slope * t : gamma_final
        elseif kind === :constant_rate
            gamma_fn = t -> gamma0 + gamma_rate * t
        elseif kind === :sinusoidal
            omega = 2pi * frequency
            gamma_fn = t -> offset + amplitude * sin(omega * t + phase)
        else
            error("Unknown shear profile kind: $kind")
        end
    end

    # Always approximate the derivative numerically from gamma_fn.
    rate_fn = t -> begin
        h = 1e-6
        (gamma_fn(t + h) - gamma_fn(t - h)) / (2h)
    end

    return ShearProfile(gamma_fn, rate_fn)
end

@inline shear_state(profile::ShearProfile, t) = (profile.gamma(t), profile.gamma_rate(t))

@inline function wall_sign(side)
    side === :top && return one(Int)
    side === :bottom && return -one(Int)
    throw(ArgumentError("side must be :top or :bottom, got $side"))
end

@inline function wall_displacement_from_shear(gamma::T, gap::T, side::Symbol) where T
    return wall_sign(side) * gamma * gap / 2
end

@inline function wall_speed_from_shear_rate(gamma_rate::T, gap::T, side::Symbol) where T
    return wall_sign(side) * gamma_rate * gap / 2
end

function average_wall_gap(top_wall::Vector{SVector{2,T}}, bot_wall::Vector{SVector{2,T}}) where T
    isempty(top_wall) && error("top_wall is empty")
    isempty(bot_wall) && error("bot_wall is empty")

    y_top_sum = zero(T)
    for w in top_wall
        y_top_sum += w[2]
    end
    y_bot_sum = zero(T)
    for w in bot_wall
        y_bot_sum += w[2]
    end

    return y_top_sum / length(top_wall) - y_bot_sum / length(bot_wall)
end

function wall_velocity_from_shear(applied_shear, nsteps, dt, box_side; cutoff_local = cutoff)
    H = box_side - cutoff_local      # effective gap between walls
    T = nsteps * dt                  # total simulation time
    return applied_shear * H / T
end


# Apply tangential drag near the walls to reduce slip. Drag is stronger closer to the wall.
function apply_wall_drag!(sys::Molly.System,
                          n_bulk::Int,
                          wall_y_top::Float64,
                          wall_y_bot::Float64,
                          shear::ShearProfile,
                          t::Real,
                          gap::Real,
                          box_side::Real;
                          dt::Real,
                          drag_coeff::Float64 = 1.5,
                          rate_ref::Float64 = 1.0)
    gamma, gamma_rate = shear_state(shear, t)
    y_max = wall_y_top
    y_min = wall_y_bot
    v_top = wall_speed_from_shear_rate(gamma_rate, gap, :top)
    v_bot = wall_speed_from_shear_rate(gamma_rate, gap, :bottom)

    @inbounds for i in 1:n_bulk
        pos = sys.coords[i]
        vel = sys.velocities[i]

        dist_top = max(0.0, y_max - pos[2])
        dist_bot = max(0.0, pos[2] - y_min)

        # Linear weights across the gap to cover the whole bulk and cancel at midplane.
        w_top = max(0.0, 1.0 - dist_top / gap)
        w_bot = max(0.0, 1.0 - dist_bot / gap)

        w_sum = w_top + w_bot
        if w_sum == 0
            continue
        end

        v_target = (w_top * v_top + w_bot * v_bot) / w_sum
        rate_scale = abs(gamma_rate) / max(rate_ref, eps(rate_ref))
        drag = rate_scale * (vel[1] - v_target) * w_sum
        vx = vel[1] - drag_coeff * drag * dt
        sys.velocities[i] = SVector(vx, vel[2])
    end
end

# Update wall bead positions and velocities to follow the current shear state.
function enforce_wall_motion!(sys::Molly.System,
                              wall_indices::AbstractVector{Int},
                              wall_bases::AbstractVector{SVector{2,Float64}},
                              wall_sides::AbstractVector{Symbol},
                              shear::ShearProfile,
                              t::Real,
                              gap::Real,
                              box_side::Real)
    gamma, gamma_rate = shear_state(shear, t)

    @inbounds for (idx, base, side) in zip(wall_indices, wall_bases, wall_sides)
        disp = wall_displacement_from_shear(gamma, gap, side)
        speed = wall_speed_from_shear_rate(gamma_rate, gap, side)
        x = wrap_x(base[1] + disp, box_side)
        sys.coords[idx] = SVector(x, base[2])
        sys.velocities[idx] = SVector(speed, 0.0)
    end
end