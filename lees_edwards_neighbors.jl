# Custom neighbor finder that respects Lees–Edwards boundary conditions

module LeesEdwardsNeighbors

using Molly
using StaticArrays
using Base.Threads
import Main: shear_state
import Main: wrap_x

export LeesEdwardsNeighborFinder, make_le_neighbor_finder

struct LeesEdwardsNeighborFinder
    cutoff::Float64        # neighbor cutoff distance
    n_steps::Int           # rebuild interval (steps)
    box_side::Float64      # box length (assumed square)
    shear_profile          # object used by `shear_state(profile, t)`
    dt::Float64            # timestep to convert step_n -> time
    bin_size::Float64      # spatial bin size (≥ cutoff); default = cutoff
end

"""
    make_le_neighbor_finder(; cutoff, n_steps=10, box_side, shear_profile, dt)

Convenience constructor with keyword arguments.
"""
function make_le_neighbor_finder(; cutoff::Float64, n_steps::Int=10, box_side::Float64, shear_profile, dt::Float64, bin_size::Union{Nothing,Float64}=nothing)
    bs = isnothing(bin_size) ? cutoff : max(cutoff, bin_size)
    return LeesEdwardsNeighborFinder(cutoff, n_steps, box_side, shear_profile, dt, bs)
end

# Local LE-aware separation using user's wrap_x and shear_state helpers
@inline function le_separation(xi::SVector{2,Float64}, xj::SVector{2,Float64}, box_side::Float64, γ::Float64)
    dx_raw = xj[1] - xi[1]
    dy_raw = xj[2] - xi[2]
    dy = wrap_x(dy_raw, box_side)
    dx = wrap_x(dx_raw - γ * dy, box_side)
    return SVector(dx, dy)
end

"""
    Molly.find_neighbors(sys, nf::LeesEdwardsNeighborFinder, current_neighbors=nothing,
                         step_n::Integer=0, force_recompute::Bool=false, n_threads::Integer=Threads.nthreads())

Build a neighbor list using Lees–Edwards minimum-image displacement with cutoff `nf.cutoff`.
Recomputes every `nf.n_steps` unless `force_recompute=true`.
"""
function Molly.find_neighbors(
    sys::Molly.System,
    nf::LeesEdwardsNeighborFinder,
    current_neighbors::Union{Nothing, Molly.NeighborList}=nothing,
    step_n::Integer=0,
    force_recompute::Bool=false;
    n_threads::Integer=Threads.nthreads(),
)
    if !force_recompute && step_n % nf.n_steps != 0 && current_neighbors !== nothing
        return current_neighbors
    end

    neighbors = Molly.NeighborList()
    coords = sys.coords
    N = length(coords)
    t = step_n * nf.dt
    γ, _ = shear_state(nf.shear_profile, t)
    cutoff2 = nf.cutoff^2
    L = nf.box_side
    bs = max(nf.cutoff, nf.bin_size)
    nx = max(1, Int(floor(L / bs)))
    ny = nx

    # If bins collapse to 1, fall back to O(N^2)
    if nx == 1 || ny == 1
        @inbounds for i in 1:N-1
            xi = coords[i]
            for j in i+1:N
                dr = le_separation(xi, coords[j], L, γ)
                if dr[1]*dr[1] + dr[2]*dr[2] <= cutoff2
                    push!(neighbors, (i, j, false))
                end
            end
        end
        return neighbors
    end

    # Transform to sheared coordinates: x' = x - γ y, y' = y, wrapped to [0,L)
    xs = Vector{SVector{2,Float64}}(undef, N)
    @inbounds for i in 1:N
        p = coords[i]
        xprime = mod(p[1] - γ * p[2], L)
        yprime = mod(p[2], L)
        xs[i] = SVector(xprime, yprime)
    end

    # Build bins
    bins = [Int[] for _ in 1:(nx*ny)]
    @inline function bin_index(xp::Float64, yp::Float64)
        ix = 1 + Int(floor(xp / bs))
        iy = 1 + Int(floor(yp / bs))
        ix = clamp(ix, 1, nx)
        iy = clamp(iy, 1, ny)
        return (iy - 1) * nx + ix
    end

    @inbounds for i in 1:N
        xp, yp = xs[i][1], xs[i][2]
        b = bin_index(xp, yp)
        push!(bins[b], i)
    end

    # Neighbor search across 3x3 neighboring bins around each particle's bin
    @inline function wrap_bin_idx(k::Int, maxk::Int)
        k < 1 && return maxk
        k > maxk && return 1
        return k
    end

    @inbounds for i in 1:N
        xp, yp = xs[i][1], xs[i][2]
        ix = 1 + Int(floor(xp / bs)); ix = clamp(ix, 1, nx)
        iy = 1 + Int(floor(yp / bs)); iy = clamp(iy, 1, ny)
        for dix in -1:1
            jx = wrap_bin_idx(ix + dix, nx)
            for diy in -1:1
                jy = wrap_bin_idx(iy + diy, ny)
                b = (jy - 1) * nx + jx
                for j in bins[b]
                    j <= i && continue
                    dr = le_separation(coords[i], coords[j], L, γ)
                    if dr[1]*dr[1] + dr[2]*dr[2] <= cutoff2
                        push!(neighbors, (i, j, false))
                    end
                end
            end
        end
    end

    return neighbors
end

end # module
