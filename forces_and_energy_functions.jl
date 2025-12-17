# Custom pairwise interaction with cutoff
struct EmulsionInter{T} <: PairwiseInteraction
    cutoff::T
end
Molly.use_neighbors(::EmulsionInter) = true # use neighbor lists; x-wrapping handled in force

# minimum-image helper for displacements
@inline wrap_delta(d::Real, L::Real) = d - L * round(d / L)

@inline function emulsion_force_magnitude(a_ij::Real, r::Real, cutoff::Real)
    if r >= cutoff
        return 0.0
    end
    return a_ij * (1 - r/cutoff)
end

@inline function emulsion_potential(a_ij::Real, r::Real, cutoff::Real)
    if r >= cutoff
        return 0.0
    end
    return a_ij * (0.5*cutoff - r + r^2/(2*cutoff))
end

@inline function emulsion_force_potential(a_ij::Real, r::Real, cutoff::Real)
    fmag = emulsion_force_magnitude(a_ij, r, cutoff)
    U = emulsion_potential(a_ij, r, cutoff)
    return fmag, U
end

# Pairwise scalar force law (depends only on scalar distance between particles "r")
function Molly.pairwise_force(inter::EmulsionInter, r, params)
    a_ij, cut_off, repulsive = params
    fmag, _ = emulsion_force_potential(a_ij, r, inter.cutoff)
    return fmag
end

# Vector force along minimum distance vector between particles i and j
function Molly.force(inter::EmulsionInter,
                     vec_ij,
                     atom_i::Molly.Atom,
                     atom_j::Molly.Atom,
                     force_units,
                     special,
                     coord_i,
                     coord_j,
                     boundary::Molly.RectangularBoundary,
                     velocity_i,
                     velocity_j,
                     step_n)

    # Recompute displacement to wrap only in x; do not wrap in y to avoid asymmetric image artifacts.
    dx = wrap_x(coord_j[1] - coord_i[1], boundary.side_lengths[1])
    dy_raw = coord_j[2] - coord_i[2]
    dy = periodic_y_mode ? wrap_delta(dy_raw, boundary.side_lengths[2]) : dy_raw
    disp = SVector(dx, dy)
    r = norm(disp)
    # Guard against r = 0 to avoid division by zero
    if r <= eps(eltype(disp))
        return zero(disp)
    end

    # Choose indexes corresponding to atom types 1-> oil, 2-> water 3-> wall
    ti = atom_i.atom_type  
    tj = atom_j.atom_type
    
    a_ij = get(A_MAP, (ti, tj), 0.0) # assign repulsion coefficient based on the atom types

    # Apply slightly higher cutoff for the surface interactions
    if ti == 4 || tj == 4
        coff = inter.cutoff
    else
        coff = inter.cutoff/1.5
    end
    repulsive = a_ij == a_oil_oil # specify if there should be strong surfactant like repulsion betwen two particles 
    params = (a_ij, coff, repulsive)

    fmag, _ = emulsion_force_potential(a_ij, r, coff) # get scalar force magnitude
    return fmag * disp / r 
end

# Compute proxy temperature from kinetic energy (k_B = 1).
@inline function current_temperature(sys::Molly.System, n_bulk::Int)
    ke = 0.0
    @inbounds for i in 1:n_bulk
        ke += 0.5 * sys.atoms[i].mass * sum(abs2, sys.velocities[i])
    end
    # 2 degrees of freedom per particle (2D), k_B = 1
    return 2 * ke / (2 * n_bulk)
end

# Total energy (kinetic + potential) using the same interaction as the force.
function emulsion_energy(sys::Molly.System, cutoff::Float64, box_side::Float64; n_bulk_only::Union{Nothing,Int}=nothing)
    N = n_bulk_only === nothing ? length(sys.coords) : n_bulk_only

    E_kin = 0.0
    @inbounds for i in 1:N
        E_kin += 0.5 * sys.atoms[i].mass * sum(abs2, sys.velocities[i])
    end

    E_pot = 0.0
    @inbounds for i in 1:N-1
        ti = sys.atoms[i].atom_type
        xi = sys.coords[i]
        for j in i+1:N
            tj = sys.atoms[j].atom_type
            a = get(A_MAP, (ti, tj), 0.0)
            dx = wrap_x(sys.coords[j][1] - xi[1], box_side)
            dy_raw = sys.coords[j][2] - xi[2]
            dy = periodic_y_mode ? wrap_delta(dy_raw, box_side) : dy_raw
            r = sqrt(dx*dx + dy*dy)
            _, U = emulsion_force_potential(a, r, cutoff)
            E_pot += U
        end
    end

    return E_kin + E_pot, E_kin, E_pot
end
