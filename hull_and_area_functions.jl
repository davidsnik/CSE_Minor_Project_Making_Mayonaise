# Soft wall around each droplet hull: pushes beads inward if outside/near boundary
function apply_soft_hull_wall!(sys::Molly.System,
                               droplet_ranges::Vector{UnitRange{Int}},
                               box_side::Float64;
                               hulls_idx,
                               dt::Float64,
                               k_wall::Float64 = 120.0,
                               buffer::Float64 = 0.5*cutoff)
    area2(verts) = begin
        a = 0.0; n = length(verts)
        @inbounds for i in 1:n
            j = (i == n) ? 1 : i+1
            a += verts[i][1]*verts[j][2] - verts[j][1]*verts[i][2]
        end
        a
    end
  
    
    global_hulls_idx = local_idx_global(hulls_idx, [length(dr) for dr in droplet_ranges])
    @inbounds for (k, dr) in enumerate(droplet_ranges)
      
        pts = sys.coords[dr]
        length(pts) < 3 && continue

        # Current mean-x for local rewrap; use to rewrap both beads and cached hull
        xref = Statistics.mean(p[1] for p in pts)
        rewrap_x(x) = xref + wrap_x(x - xref, box_side)

        cached_hulls = [ pts[i] for i  in hulls_idx[k] ]
        verts = [SVector(rewrap_x(v[1]), v[2]) for v in cached_hulls]
        length(verts) < 3 && continue

        # Ensure CCW orientation
        if area2(verts) < 0.0
            reverse!(verts)
        end

        # Edge outward normals
        m = length(verts)
        normals = Vector{SVector{2,Float64}}(undef, m)
        @inbounds for i in 1:m
            j = (i == m) ? 1 : i+1
            e = verts[j] - verts[i]
            n = SVector(e[2], -e[1])
            len = norm(n)
            normals[i] = len > 0 ? n/len : SVector(0.0, 0.0)
        end

        @inbounds for gi in dr
            if gi in global_hulls_idx[k]
                continue
            end
            p = sys.coords[gi]
            
            p_rw = SVector(rewrap_x(p[1]), p[2])
        

            # Half-space max to find most-violated face
            sdist = -Inf
            n_sel = SVector(0.0, 0.0)
            

            for i in 1:m
                val = dot(normals[i], p_rw - verts[i])
                if val > sdist
                    sdist = val
                    n_sel = normals[i]
                end
            end

            v = sys.velocities[gi]
            HULL_REFLECT_BUFFER = 0.2 * cutoff
            HULL_REFLECT_DAMP   = 0.9

            if sdist >= 0.0 || sdist > -HULL_REFLECT_BUFFER
                
                v_n = dot(v, n_sel)
                v_t = v - v_n * n_sel
                p_rw_new = p_rw - (sdist + buffer) * n_sel # push back inside with buffer
                #sys.coords[gi] = SVector(wrap_x(p_rw_new[1], box_side), p_rw_new[2])
                sys.velocities[gi] = abs(v_n) * -n_sel +  v_t
            end

            
        end
    end
end
function local_idx_global(hull_local_idx, particles_per_droplet)
    offset = 0
    global_hulls_idx = Vector{Vector{Int}}()
    for (k, hull_idx) in enumerate(hull_local_idx)
        # Convert local indices (within droplet) to global indices (within x0_oil)
        global_indices = hull_idx .+ offset

        push!(global_hulls_idx, global_indices)
        offset += particles_per_droplet[k]
    end
    return global_hulls_idx
end

# Inter-droplet repulsion: push hull beads of different droplets apart to prevent penetration.
function apply_inter_droplet_repulsion!(
    sys::Molly.System,
    surface_particles_idx::Vector{Vector{Int}},
    box_side::Float64;
    dt::Float64,
    k_rep::Float64,
    r_cut::Float64 = 0.3,
)
    n_drops = length(surface_particles_idx)
    n_drops < 2 && return

    @inbounds for i in 1:(n_drops - 1)
        ring_i = surface_particles_idx[i]
        for j in (i + 1):n_drops
            ring_j = surface_particles_idx[j]
            # Check all pairs of hull beads between droplet i and j
            for gi in ring_i
                pi = sys.coords[gi]
                for gj in ring_j
                    pj = sys.coords[gj]
                    dx = wrap_x(pj[1] - pi[1], box_side)
                    dy = pj[2] - pi[2]
                    r = sqrt(dx^2 + dy^2)
                    r < 1e-12 && continue
                    r > r_cut && continue

                    # Linear repulsion: force magnitude increases as beads get closer
                    fmag = k_rep * (1.0 - r / r_cut) * dt
                    n_hat = SVector(dx, dy) / r
                    # Debug check for unit vector; keep lightweight
                    abs(norm(n_hat) - 1) < 1e-6 || @warn "n_hat not unit" norm(n_hat)

                    # Push both beads apart along the line connecting them
                    sys.velocities[gi] = sys.velocities[gi] - fmag * n_hat
                    sys.velocities[gj] = sys.velocities[gj] + fmag * n_hat
                end
            end
        end
    end
end

# Maintain droplet area by adding outward velocity to hull beads when area shrinks.
function apply_area_constraint!(
    sys::Molly.System,
    surface_particles_idx::Vector{Vector{Int}},
    target_areas::Vector{Float64},
    k_area::Float64,
    box_side::Float64;
    dt::Float64,
)
    k_area <= 0 && return

    area2(verts) = begin
        a = 0.0; n = length(verts)
        @inbounds for i in 1:n
            j = (i == n) ? 1 : i+1
            a += verts[i][1]*verts[j][2] - verts[j][1]*verts[i][2]
        end
        a
    end

    @inbounds for (k, ring) in enumerate(surface_particles_idx)
        length(ring) < 3 && continue
        A0 = target_areas[k]
        A0 <= 0 && continue

        # Local wrap around the hull centroid to avoid seam issues.
        pts = sys.coords[ring]
        xref = Statistics.mean(p[1] for p in pts)
        rewrap_x(x) = xref + wrap_x(x - xref, box_side)
        verts = [SVector(rewrap_x(p[1]), p[2]) for p in pts]

        A = 0.5 * abs(area2(verts))
        A >= A0 && continue

        deficit = (A0 - A) / A0
        push_mag = k_area * deficit * dt

        # Use radial outward direction from centroid of current verts
        cx = Statistics.mean(v[1] for v in verts)
        cy = Statistics.mean(v[2] for v in verts)
        centroid = SVector(cx, cy)

        for (idx_local, gi) in enumerate(ring)
            dir = verts[idx_local] - centroid
            norm_dir = norm(dir)
            norm_dir == 0 && continue
            n_hat = dir / norm_dir
            sys.velocities[gi] = sys.velocities[gi] + push_mag * n_hat
        end
    end
end