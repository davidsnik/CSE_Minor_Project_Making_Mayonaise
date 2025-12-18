# Custom logger to record coordinates without units (taken from Molly source code)
function MyCoordinatesLogger(T, n_steps::Integer; dims::Integer=2)
    return Molly.GeneralObservableLogger(
        Molly.coordinates_wrapper,
        Array{SArray{Tuple{dims}, T, 1, dims}, 1},
        n_steps,
    )
end
MyCoordinatesLogger(n_steps::Integer; dims::Integer=2) = MyCoordinatesLogger(Float64, n_steps; dims=dims)

# Fallback manual trajectory collector with subsampling by save_every
# Saves the initial frame, then every save_every steps, and always the final frame.
function run_and_collect!(sys::Molly.System, simulator, nsteps::Integer;
                          save_every::Integer=1,
                          pre_step!::Union{Nothing,Function}=nothing,
                          post_step!::Union{Nothing,Function}=nothing)
    save_every = max(1, save_every)
    hist = Vector{Vector{SVector{2,Float64}}}()

    if pre_step! !== nothing
        pre_step!(0)
    end
    push!(hist, copy(sys.coords))

    # Progress bar for simulation
    local p = nothing
    if HAS_PROGRESSMETER[]
        p = ProgressMeter.Progress(nsteps; desc="Simulating", dt=0.2)
    end

    for s in 1:nsteps
        if pre_step! !== nothing
            pre_step!(s)
        end
        simulate!(sys, simulator, 1)
        
        if post_step! != nothing
           
            post_step!(s)
        end
        if s % save_every == 0 || s == nsteps
            push!(hist, copy(sys.coords))
        end
        if p !== nothing
            ProgressMeter.next!(p)
        end
    end
    return hist
end

function build_emulsion_system(x0_bulk::Vector{SVector{2,Float64}},
                               x0_walls::Vector{SVector{2,Float64}},
                               n_oil::Int, n_water::Int,
                               box_side::Float64, cutoff::Float64,
                               bulk_velocities,
                               wall_velocities,
                               surface_idx,
                               global_hulls_idx,
                               bonds_type,
                               nsteps::Integer; periodic_y::Bool=false)

    n_bulk_local = n_oil + n_water
    n_walls = length(x0_walls)
    n_total = n_bulk_local + n_walls
    wall_range = n_walls == 0 ? (0:-1) : ((n_bulk_local+1):n_total)

    # Molly.RectangularBoundary does not accept per-dimension periodic flags; use the
    # standard constructor (periodic in all dimensions) and handle x wrapping manually
    # for wall motion via wrap_x.
    # Periodic in x; standard span in y (we ignore y wrapping in force calculation).
    y_span = box_side
    boundary = Molly.RectangularBoundary(SVector{2,Float64}(box_side, y_span))
    
    atoms = Vector{Molly.Atom}(undef, n_total)
    for i in 1:n_oil
        if i in surface_idx[1]
            atoms[i] = Molly.Atom(mass=4.0, atom_type=4)  # surface oil particles
        else
            atoms[i] = Molly.Atom(mass=4.0, atom_type=1)  # bulk oil particles
        end
        
    end
    for i in (n_oil+1):(n_oil+n_water)
        atoms[i] = Molly.Atom(mass=1.0, atom_type=2)
    end
    for i in (n_bulk_local+1):n_total
        atoms[i] = Molly.Atom(mass=1.0, atom_type=3)
    end

    coords_all = Vector{SVector{2,Float64}}(undef, n_total)
    coords_all[1:n_bulk_local] = x0_bulk
    if n_walls > 0
        coords_all[wall_range] = x0_walls
    end

    velocities = Vector{SVector{2,Float64}}(undef, n_total)
    velocities[1:n_bulk_local] = bulk_velocities
    if n_walls > 0
        wall_velocities = isempty(wall_velocities) ? [SVector{2,Float64}(0.0, 0.0) for _ in 1:n_walls] : wall_velocities
        velocities[wall_range] = wall_velocities
    end

    # eligible = falses(n_total, n_total)
    # eligible[1:n_bulk_local, 1:n_bulk_local] = cluster_matrix
    eligible = trues(n_total, n_total)
    # Disable self-interactions
    @inbounds for i in 1:n_total
        eligible[i,i] = false
    end
    # Disable interactions within each droplet hull
    for k in length(global_hulls_idx)
        hull = global_hulls_idx[k]
        for i in hull, j in hull
            eligible[i,j] = false
        end
    end
   
    if n_walls > 0
        eligible[wall_range, wall_range] .= false  # keep rough wall beads rigid relative to each other
    end

    cellListMap_finder = Molly.CellListMapNeighborFinder(
        eligible=eligible,
        dist_cutoff=cutoff,
        x0=coords_all,
        unit_cell = boundary,
        n_steps = 5, # update neighbors more often so moving walls stay accurate
        dims = 2,
    )

    # Build Molly system that will be solved by a simulator
    sys = Molly.System(
        atoms = atoms,
        coords = coords_all,
        boundary = boundary,
        velocities = velocities,
        pairwise_inters = (MyPairwiseInter=EmulsionInter(cutoff),),
        specific_inter_lists = bonds_type,
        neighbor_finder = cellListMap_finder,
        loggers = (coords=MyCoordinatesLogger(nsteps, dims=2),),
        energy_units = Unitful.NoUnits,
        force_units = Unitful.NoUnits
    )
    return sys, wall_range
end
