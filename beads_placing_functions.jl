# Place n_droplets centers inside the box, separated by at least ~2R
function generate_droplet_centers(n_droplets::Int,
                                  box_side::T,
                                  R::T;
                                  margin_factor::T = 1.5) where T
    
    centers = SVector{2,T}[]
    min_dist = 2 * R * margin_factor

    x_min = R
    x_max = box_side - R
    y_min = R + cutoff
    y_max = box_side - (R + cutoff)

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
            println("Could only place $(length(centers)) droplets. Reducing target to $(n_droplets).")
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

# Generate oil positions in several identical circular droplets
function generate_multi_droplet_and_matrix(n_oil::Int, n_water::Int,
                                    centers::Vector{SVector{2,T}},
                                    droplet_area::T, surface_concentration::T) where T
    n_droplets = length(centers)
    
    R = sqrt(droplet_area / pi)  # radius of each droplet

    oil = SVector{2,T}[]

    # distribute particle counts as evenly as possible
    base = div(n_oil, n_droplets)
    extra = rem(n_oil, n_droplets)
    n_per = [base + (k <= extra ? 1 : 0) for k in 1:n_droplets]
   
    # per-droplet blocks (Bool)
    droplets_matrices = [falses(n_per[k], n_per[k]) for k in 1:n_droplets]

    # assemble a dense block-diagonal matrix
    droplets_matrix = trues(n_oil+n_water, n_oil+n_water)
    offset = 0
    for M in droplets_matrices
        m = size(M, 1)
        droplets_matrix[offset+1:offset+m, offset+1:offset+m] .= M
        offset += m
    end

    # Store convex hulls for each droplet
    convex_hulls = Vector{Vector{Int}}()  # indices within each droplet
    convex_hull_coords = Vector{Vector{SVector{2,T}}}()  # actual coordinates
    hulls = Vector{Any}()
    
    for (k, center) in enumerate(centers)
        droplet_points = SVector{2,T}[]
        
        nr_beads_boundary = Int(ceil(n_per[k]*surface_concentration))
       
        angle_increment = 2 * pi / nr_beads_boundary
        for i in 1:n_per[k]
            if i < nr_beads_boundary + 1
                # dense bounary
                r = R
                theta = (i-1) * angle_increment
                
            else
                # uniform sampling in a disk: r = R*sqrt(u), theta in [0,2pi]
                r = R * sqrt(rand(T))
                theta = 2 * T(pi) * rand(T)
            end
            p = SVector{2,T}(
                center[1] + r * cos(theta),
                center[2] + r * sin(theta),
            )
            push!(oil, p)
            push!(droplet_points, p)
        end
        
        # Compute convex hull using GeometryOps
        # Convert to tuples for GeometryOps
        points_tuples = [(p[1], p[2]) for p in droplet_points]
        hull_polygon = convex_hull(points_tuples)

        push!(hulls, hull_polygon)

       
        
        # Extract coordinates from the polygon
        hull_coords = GeoInterface.coordinates(hull_polygon)[1]  # Get exterior ring
        
        # Extract hull indices by matching coordinates
        hull_indices = Int[]
        hull_svecs = SVector{2,T}[]
        
        for hull_pt in hull_coords[1:end-1]  # Last point repeats the first in closed polygons
            for (idx, pt) in enumerate(droplet_points)
                if abs(pt[1] - hull_pt[1]) < 1e-10 && abs(pt[2] - hull_pt[2]) < 1e-10
                    push!(hull_indices, idx)
                    push!(hull_svecs, pt)
                    break
                end
            end
        end
        
        push!(convex_hulls, hull_indices)
        push!(convex_hull_coords, hull_svecs)
    end
    

    return oil, R, droplets_matrix, n_per, hulls, convex_hulls, convex_hull_coords
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
    wall_buffer = 2cutoff
    interface_buffer = 0.5cutoff
    water = SVector{2,T}[]
    x_min, x_max = zero(T), box_side
    # keep away from walls in y by a small buffer as well
    y_min, y_max = wall_buffer, box_side - wall_buffer
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

        # outside every droplet with a small buffer so water isn't seeded on the interface
        inside_any_droplet = any(norm(p - c) <= R + interface_buffer for c in centers)

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
