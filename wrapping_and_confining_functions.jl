# Keep bulk particles inside the gap in y by reflecting if they cross the walls.
function confine_bulk_y!(sys::Molly.System, n_bulk::Int, y_min::Float64, y_max::Float64)
    @inbounds for i in 1:n_bulk
        pos = sys.coords[i]
        vel = sys.velocities[i]
        if pos[2] > y_max
            new_y = 2y_max - pos[2]
            sys.coords[i] = SVector(pos[1], new_y)
            sys.velocities[i] = SVector(vel[1], -vel[2])
        elseif pos[2] < y_min
            new_y = 2y_min - pos[2]
            sys.coords[i] = SVector(pos[1], new_y)
            sys.velocities[i] = SVector(vel[1], -vel[2])
        end
    end
end

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
