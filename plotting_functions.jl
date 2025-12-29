function visualize_with_progress(
    coords_history::Vector{Vector{SVector{2,Float64}}},
    boundary::Molly.RectangularBoundary,
    outfile::String;
    color,
    markersize::Real=1.0,
    framerate::Integer=30,
    droplet_ranges::Vector{UnitRange{Int}},
    box_side::Float64,
    hulls_idx::Union{Nothing, Vector{Vector{Int}}} = nothing,
    droplet_colors::Vector{RGB{Float64}},
)
    frames = length(coords_history)

    isempty(coords_history) && error("No coordinates recorded; nothing to visualize.")

    # Axis limits from full trajectory (as before)
    xmin = Inf; xmax = -Inf; ymin = Inf; ymax = -Inf
    for frame in coords_history
        for p in frame
            x = p[1]; y = p[2]
            if isfinite(x) && isfinite(y)
                xmin = min(xmin, x); xmax = max(xmax, x)
                ymin = min(ymin, y); ymax = max(ymax, y)
            end
        end
    end
    if !isfinite(xmin) || !isfinite(ymin)
        error("Coordinates contain no finite values; cannot visualize trajectory.")
    end
    span_x = xmax - xmin
    span_y = ymax - ymin
    pad = 0.1 * max(max(span_x, span_y), 1.0)

    fig = GLMakie.Figure(size=(800,800))
    ax = GLMakie.Axis(fig[1,1];
        limits = (xmin - pad, xmax + pad, ymin - pad, ymax + pad),
        aspect = GLMakie.DataAspect(),
    )

    # Scatter setup (as before)
    first_frame = coords_history[1]
    N = length(first_frame)
    xs = GLMakie.Observable([first_frame[i][1] for i in 1:N])
    ys = GLMakie.Observable([first_frame[i][2] for i in 1:N])
    msizes = [i <= n_oil ? 6.0 : i <= n_oil + n_water ? 4.0 : 3.5 for i in 1:N]
    mcolors = copy(color)
    # Enlarge and color the first bead of each droplet as a red marker
    for dr in droplet_ranges
        isempty(dr) && continue
        first_idx = first(dr)
        if first_idx <= N
            msizes[first_idx] = 10.0
            mcolors[first_idx] = RGB(1.0, 0.0, 0.0)
        end
    end
    GLMakie.scatter!(ax, xs, ys; color=mcolors, markersize=msizes)


    n_drops = length(droplet_ranges)

    plot_hulls = hulls_idx !== nothing

    hull_pts_obs =[GLMakie.Observable(GLMakie.Point2f[]) for _ in 1:n_drops] 

    if plot_hulls
        @assert length(hulls_idx) == n_drops "hulls_idx must match droplet_ranges"
        for k in 1:n_drops
            GLMakie.lines!(ax, hull_pts_obs[k];
                color = droplet_colors[k],
                linewidth = 2
            )
        end
    end

    # Progress for rendering
    local progress = nothing
    if HAS_PROGRESSMETER[]
        progress = ProgressMeter.Progress(frames; desc="Rendering", dt=0.2)
    end

    # Helper: compute hull ring coords for a droplet within a frame
    compute_hull_coords = function(frame::Vector{SVector{2,Float64}}, dr::UnitRange{Int}, hull_indx::Vector{Int})
        pts = frame[dr]
        length(pts) < 3 && return GLMakie.Point2f[]
        # Local x wrap around mean
        xref = Statistics.mean(p[1] for p in pts)
        rewrap_x(x) = xref + wrap_x(x - xref, box_side)
        pts_rw = [(rewrap_x(p[1]), p[2]) for p in pts]

        # Build polyline with NaN breaks when crossing the box in x to avoid long wrap lines
        ps = GLMakie.Point2f[]
        nidx = length(hull_indx)
        for k in 1:nidx
            i = hull_indx[k]
            j = hull_indx[mod1(k+1, nidx)]  # next vertex (wrap around)
            p1 = pts_rw[i]; p2 = pts_rw[j]
            push!(ps, GLMakie.Point2f(p1[1], p1[2]))
            dx = p2[1] - p1[1]
            if abs(dx) > box_side/2
                # insert break to avoid drawing across periodic seam
                push!(ps, GLMakie.Point2f(NaN, NaN))
            end
        end
        # Close polygon: add first point again unless last segment already NaN-broken
        if isempty(ps) || !(isnan(ps[end][1]) || isnan(ps[end][2]))
            first_idx = hull_indx[1]
            push!(ps, GLMakie.Point2f(pts_rw[first_idx][1], pts_rw[first_idx][2]))
        end
        return ps
    end

    GLMakie.record(fig, outfile, 1:frames; framerate=framerate) do i
        ci = coords_history[i]
        @inbounds for k in 1:N
            xs[][k] = ci[k][1]
            ys[][k] = ci[k][2]
        end
        GLMakie.notify(xs); GLMakie.notify(ys)

        if plot_hulls
            for (idx, dr) in enumerate(droplet_ranges)
                pts_h = compute_hull_coords(ci, dr, hulls_idx[idx])
                hull_pts_obs[idx][] = pts_h
                GLMakie.notify(hull_pts_obs[idx])
            end
        end
        if progress !== nothing
            ProgressMeter.next!(progress)
        end
    end
end


# Plot the droplet arrangement for verification (with the convex hulls)

function plot_hull_points(x0_oil::Vector{SVector{2,T}}, 
                          hulls_idx::Vector{Vector{Int}}, 
                          n_per::Vector{Int};
                          show_all_oil::Bool = true,
                          box_side::Union{Nothing,Float64} = nothing) where T
    
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1]; 
              title="Convex Hull Points", 
              xlabel="x", 
              ylabel="y", 
              aspect=DataAspect())
    
    # Optionally plot all oil particles in background
    if show_all_oil
        scatter!(ax, [p[1] for p in x0_oil], [p[2] for p in x0_oil]; 
                 color=:lightgray, markersize=3, label="All Oil")
    end
    
    # Plot hull points for each droplet
    offset = 0
    for (k, hull_local_idx) in enumerate(hulls_idx)
        # Convert local indices (within droplet) to global indices (within x0_oil)
        global_indices = hull_local_idx .+ offset
        
        # Extract hull points
        hull_points = [x0_oil[i] for i in global_indices]
        hull_xs = [p[1] for p in hull_points]
        hull_ys = [p[2] for p in hull_points]
        
        # Plot hull points with unique color per droplet
        scatter!(ax, hull_xs, hull_ys; 
                 markersize=8, 
                 label=(k==1 ? "Hull Points" : nothing))
        
        # Optionally draw lines connecting hull points
        # Close the polygon by adding first point at end
        push!(hull_xs, hull_xs[1])
        push!(hull_ys, hull_ys[1])
        lines!(ax, hull_xs, hull_ys; 
               color=:red, 
               linewidth=2,
               label=(k==1 ? "Hull Outline" : nothing))
        
        offset += n_per[k]
    end
    
    # Set axis limits if box_side is provided
    if box_side !== nothing
        xlims!(ax, 0, box_side)
        ylims!(ax, 0, box_side)
    end
    
    axislegend(ax; position=:lt)
    
    return fig
end

function visualize_soft_spheres_with_progress(
    coords_history::Vector{Vector{SVector{2,Float64}}},
    boundary::Molly.RectangularBoundary,
    outfile::String;
    framerate::Integer=30,
    box_side::Float64,
    droplet_radius::Float64,
    n_droplets::Int,
    wall_radius::Float64 = 1.0,
    n_walls::Int = 0,
)
    frames = length(coords_history)

    isempty(coords_history) && error("No coordinates recorded; nothing to visualize.")

    # Axis limits from full trajectory (as before)
    xmin = Inf; xmax = -Inf; ymin = Inf; ymax = -Inf
    for frame in coords_history
        for p in frame
            x = p[1]; y = p[2]
            if isfinite(x) && isfinite(y)
                xmin = min(xmin, x); xmax = max(xmax, x)
                ymin = min(ymin, y); ymax = max(ymax, y)
            end
        end
    end
    if !isfinite(xmin) || !isfinite(ymin)
        error("Coordinates contain no finite values; cannot visualize trajectory.")
    end
    span_x = xmax - xmin
    span_y = ymax - ymin
    pad = 0.1 * max(max(span_x, span_y), 1.0)

    fig = GLMakie.Figure(size=(800,800))
    ax = GLMakie.Axis(fig[1,1];
        limits = (xmin - pad, xmax + pad, ymin - pad, ymax + pad),
        aspect = GLMakie.DataAspect(),
    )

    # Scatter setup (as before)
    first_frame = coords_history[1]
    N = length(first_frame)
    xs = GLMakie.Observable([first_frame[i][1] for i in 1:N])
    ys = GLMakie.Observable([first_frame[i][2] for i in 1:N])
    msizes = [i<= n_droplets ? 2*droplet_radius : 2*wall_radius for i in 1:N]
    color = [i<= n_droplets ? RGB(0.2, 0.6, 1.0) : RGB(0.5, 0.5, 0.5) for i in 1:N]
    
    GLMakie.scatter!(ax, xs, ys; markerspace = :data, markersize=msizes, color=color)

    # Progress for rendering
    local progress = nothing
    if HAS_PROGRESSMETER[]
        progress = ProgressMeter.Progress(frames; desc="Rendering", dt=0.2)
    end

    

    GLMakie.record(fig, outfile, 1:frames; framerate=framerate) do i
        ci = coords_history[i]
        @inbounds for k in 1:N
            xs[][k] = ci[k][1]
            ys[][k] = ci[k][2]
        end
        GLMakie.notify(xs); GLMakie.notify(ys)

        if progress !== nothing
            ProgressMeter.next!(progress)
        end
    end
end

function visualize_soft_spheres_with__artificial_wall(
    coords_history::Vector{Vector{SVector{2,Float64}}},
    boundary::Molly.RectangularBoundary,
    outfile::String;
    framerate::Integer=30,
    box_side::Float64,
    droplet_radius::Float64,
    n_droplets::Int,
    show_walls::Bool=true,
)
    frames = length(coords_history)

    isempty(coords_history) && error("No coordinates recorded; nothing to visualize.")

    # Axis limits from full trajectory (as before)
    xmin = Inf; xmax = -Inf; ymin = Inf; ymax = -Inf
    for frame in coords_history
        for p in frame
            x = p[1]; y = p[2]
            if isfinite(x) && isfinite(y)
                xmin = min(xmin, x); xmax = max(xmax, x)
                ymin = min(ymin, y); ymax = max(ymax, y)
            end
        end
    end
    if !isfinite(xmin) || !isfinite(ymin)
        error("Coordinates contain no finite values; cannot visualize trajectory.")
    end
    

    span_x = xmax - xmin
    span_y = ymax - ymin
    pad = 0.1 * max(max(span_x, span_y), 1.0)

    fig = GLMakie.Figure(size=(800,800))
    ax = GLMakie.Axis(fig[1,1];
        limits = (xmin - pad, xmax + pad, ymin - pad, ymax + pad),
        aspect = GLMakie.DataAspect(),
    )


    if show_walls
        # Visual cue for walls at y=0 and y=box_side
        GLMakie.lines!(ax, [xmin, xmax], [box_side, box_side]; color=:gray, linewidth=10)
        GLMakie.lines!(ax, [xmin, xmax], [0, 0]; color=:gray, linewidth=10)
    end

    # Scatter setup (as before)
    first_frame = coords_history[1]
    N = length(first_frame)
    xs = GLMakie.Observable([first_frame[i][1] for i in 1:N])
    ys = GLMakie.Observable([first_frame[i][2] for i in 1:N])
    msizes = [2*droplet_radius for i in 1:N]
    color = [RGB(0.2, 0.6, 1.0) for i in 1:N]
    
    GLMakie.scatter!(ax, xs, ys; markerspace = :data, markersize=msizes, color=color)

    # Progress for rendering
    local progress = nothing
    if HAS_PROGRESSMETER[]
        progress = ProgressMeter.Progress(frames; desc="Rendering", dt=0.2)
    end

    

    GLMakie.record(fig, outfile, 1:frames; framerate=framerate) do i
        ci = coords_history[i]
        @inbounds for k in 1:N
            xs[][k] = ci[k][1]
            ys[][k] = ci[k][2]
        end
        GLMakie.notify(xs); GLMakie.notify(ys)

        if progress !== nothing
            ProgressMeter.next!(progress)
        end
    end
end
