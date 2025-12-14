function detect_clusters_from_file( #cluster detection andrej
    filename::String;
    box_side::Float64,
    cutoff::Float64,
    factor::Float64 = 1.2
)
    data = readdlm(filename)
    N = size(data, 1)

    coords = [(data[i,1], data[i,2]) for i in 1:N]
    r2 = (factor * cutoff)^2

    visited = falses(N)
    clusters = Vector{Vector{Int}}()

    for i in 1:N
        visited[i] && continue

        cluster = Int[]
        stack = [i]
        visited[i] = true

        while !isempty(stack)
            p = pop!(stack)
            push!(cluster, p)

            xi, yi = coords[p]

            for j in 1:N
                visited[j] && continue

                xj, yj = coords[j]
                dx = wrap_x(xj - xi, box_side)
                dy = yj - yi

                if dx*dx + dy*dy < r2
                    visited[j] = true
                    push!(stack, j)
                end
            end
        end

        push!(clusters, cluster)
    end

    return clusters
end

function analyze_clusters_simple(clusters, cutoff::Float64)
    bead_radius = cutoff / 2
    bead_area   = π * bead_radius^2

    n_clusters = length(clusters)

    beads_per_cluster = [length(c) for c in clusters]

    radii = [
        sqrt((length(c) * bead_area) / π)
        for c in clusters
    ]

    avg_radius = Statistics.mean(radii)

    return n_clusters, beads_per_cluster, avg_radius
end