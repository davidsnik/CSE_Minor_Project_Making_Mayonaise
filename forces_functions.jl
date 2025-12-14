# Custom pairwise interaction with cutoff
struct EmulsionInter{T} <: PairwiseInteraction
    cutoff::T
end
Molly.use_neighbors(::EmulsionInter) = true # use neighbor lists; x-wrapping handled in force

# Pairwise scalar force law (depends only on scalar distance between particles "r")
function Molly.pairwise_force(inter::EmulsionInter, r, params)
    a_ij, cut_off, repulsive = params
    if r > cut_off
        return 0.0
    # huge repulsive force is temporarily disabled
    # elseif repulsive && r < 1e-12
    #     return 3000 * (1 - r/inter.cutoff)
    else
        return a_ij * (1 - r/cut_off)
    end
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
    dy = coord_j[2] - coord_i[2]
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

    if ti == 4 || tj == 4
        coff = inter.cutoff
    else
        coff = inter.cutoff/1.5
    end
    repulsive = a_ij == a_oil_oil # specify if there should be strong surfactant like repulsion betwen two particles 
    params = (a_ij, coff, repulsive)

    fmag = Molly.pairwise_force(inter, r, params) # get scalar force magnitude
    return fmag * disp / r 
end