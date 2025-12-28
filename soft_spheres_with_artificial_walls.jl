
using Random
using StaticArrays
using Molly
using GLMakie  # Make sure you have this loaded
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity, visualize, Langevin
using Unitful
using DelimitedFiles
using Statistics
using LinearAlgebra: norm, dot
using GeometryOps: convex_hull
using Colors: RGB
using GeoInterface

# Optional progress bars
const HAS_PROGRESSMETER = Ref(false)
try
    @eval import ProgressMeter
    HAS_PROGRESSMETER[] = true
catch
    @warn "ProgressMeter.jl not available; progress bars will be disabled."
end

#region -------- Include function files --------
#include(joinpath(@__DIR__, "functions.jl"))

# After each include there is a list of functions made available.
include(joinpath(@__DIR__, "wrapping_and_confining_functions.jl"))
# wrap_x, wrap_delta, periodic_confinement_y!,
include(joinpath(@__DIR__, "beads_placing_functions.jl"))
# generate_droplet_centers, generate_multi_droplet, generate_multi_droplet_and_matrix, generate_outside_droplets
include(joinpath(@__DIR__, "forces_and_energy_functions.jl"))
# wrap_delta, emulsion_force_magnitude, emulsion_potential, Molly.pairwise_force, Molly.force, emulsion_energy, current_temperature
include(joinpath(@__DIR__, "wall_and_shear_functions.jl"))  
# make_rough_wall_particles, make_shear_profile, shear_state, wall_sign, wall_displacement_from_shear 
# wall_speed_from_shear_rate, average_wall_gap,wall_velocity_from_shear, apply_wall_drag!, enforce_wall_motion!
include(joinpath(@__DIR__, "system_and_simulation_functions.jl"))
# MyCoordinatesLogger, run_and_collect!, build_emulsion_system
include(joinpath(@__DIR__, "plotting_functions.jl"))
# visualize_with_progress, plot_hull_points
include(joinpath(@__DIR__, "cluster_functions.jl"))
# detect_clusters_from_file, analyze_clusters_simple, visualize_soft_spheres_with_progress

#endregion


# Setting random seed for reproducibility
Random.seed!(1234)

#region ------- Simulation Parameters -----
# -------- General system parameters --------
volume_fraction_oil = 0.3                  
box_side            = 5.0
temperature         = 1.0                   # reduced temp to limit initial speeds

# ------------- Droplet settings ------------
#n_droplets = 20               # number of droplets is now based on volume fraction and droplet size
droplet_radius     = 0.1
droplet_mass       = 1.0
energy_strength    = 1.0
cutoff             = 2.5*droplet_radius    # use a modest LJ/soft-sphere cutoff ~2.5 sigma
minimal_dist_frac  = 1                  # minimal distance between initial droplet centers as fraction of 2*R
skipping_neighbors_threshold = 0.3  # if droplet diameter is larger than this, skip neighbor list in Molly system
# -------------- Wall settings --------------
# Rough wall configuration; motion is set later by the shear profile.
enable_walls = true             # set false to disable walls and use periodic y instead
periodic_y_mode = !enable_walls
wall_mass      = 10*droplet_mass
wall_radius    = 1.0 * droplet_radius     # give walls a sensible size to avoid singular kicks
wall_energy_strength = 0.5 * energy_strength  # moderate wall interaction

# Time-dependent shear strain gamma(t). Sinusoidal for viscoelastic analysis.
gamma_amplitude = 0.05
shear_freq = 0.5                    # cycles per unit time
gamma_fn = t -> gamma_amplitude * sin(2*pi*shear_freq*t)
# Optional sweep over multiple shear frequencies (leave empty to disable).
enable_freq_sweep = true          # set true to run frequency sweep instead of single-run simulation
sweep_freqs = [0.5, 1.0]      # cycles per unit time
cycles_per_freq = 3                 # simulate this many cycles for each swept frequency
discard_cycles = 1                  # ignore this many initial cycles when fitting G′/G″
max_cycles_per_freq = 5             # safety cap on cycles per frequency
stability_percent = 10.0             # declare G′/G″ stable if change is below this percent

# ------------ Simulation settings ----------
enable_energy_logging = true     # set false to skip energy calc/logging for speed
enable_langevin = false          # thermostat toggle
gamma_langevin = 0.5             # friction coefficient for Langevin thermostat
dt = 1e-5
T = 1.0                              # CHANGE TIME HERE
nsteps = Int(round(T/dt))
# Desired output fps; we will subsample to approximate this while keeping duration = T
desired_fps = 60
realistic_time = false          # if true, the video plays in real time and desired fps is disregarded; if false, output fps matches desired fps 
output_folder = "soft_spheres_mp4" # folder to save visualization output (mp4, temp plot, energy plot)
enable_shear = true             # apply affine shear each step based on gamma_fn
compute_modulus_history = true # compute shear modulus for saved frames after the run
#endregion


#region ------- System Initialization -----
volume_oil   = volume_fraction_oil * box_side^2
volume_water = (1.0 - volume_fraction_oil) * box_side^2
droplet_area = min(pi*droplet_radius^2, volume_oil)         # area of ONE droplet
if droplet_area == volume_oil
    println("Warning: Only enough oil for one droplet.")
end
n_droplets::Int = floor(Int, volume_oil / droplet_area)
println("Number of droplets: ", n_droplets)
println("Total oil area: ", n_droplets * droplet_area)
println("Real volume fraction of oil: ", (n_droplets * droplet_area) / box_side^2)


droplets = [Molly.Atom(mass=droplet_mass, σ=2*droplet_radius, ϵ=energy_strength, atom_type=1) for _ in 1:n_droplets]
# Boundary for placement; Molly's RectangularBoundary here is periodic flag-less. Non-periodic y is handled by walls.
boundary = Molly.RectangularBoundary(SVector{2,Float64}(box_side, box_side))
# Place atoms away from walls: y in [2r, box_side-2r], x in [2r, box_side-2r]
function place_away_from_walls(n, box_side, r, min_dist)
    coords = SVector{2,Float64}[]
    max_attempts = 100000
    attempts = 0
    while length(coords) < n && attempts < max_attempts
        attempts += 1
        x = 2r + rand()*(box_side - 4r)
        y = 2r + rand()*(box_side - 4r)
        p = SVector{2,Float64}(x, y)
        if all(norm(p - q) > min_dist for q in coords)
            push!(coords, p)
        end
    end
    if length(coords) < n
        error("Could not place atoms without overlap after $max_attempts attempts")
    end
    return coords
end
init_coord = place_away_from_walls(n_droplets, box_side, droplet_radius, 2.0*droplet_radius*minimal_dist_frac)
n_bulk   = n_droplets
v0_bulk = [random_velocity(droplet_mass, temperature; dims=2) for _ in 1:n_droplets]

pairwise_inter = (Molly.SoftSphere(use_neighbors=false),)
#endregion

#region -------- Initialize wall interactions --------
if enable_walls
    wall_up = SoftsphereWall(
        box_side,      # Wall at y=box_side
        wall_radius,          
        wall_energy_strength, 
        max(wall_radius * 3, cutoff)     # Cutoff
    )
    wall_down = SoftsphereWall(
        0.0,      # Wall at y=0
        wall_radius,          
        wall_energy_strength, 
        max(wall_radius * 3, cutoff)     # Cutoff
    )
    
    walls = (wall_up, wall_down,)

else
    walls = ()
end

#endregion

#region ------- Plot the force function -----
rs = range(0.1*droplet_radius, 4*droplet_radius, length=500) # Distance range 
force_magnitudes = Float64[]
forces_diff_str = Vector{Float64}[]
energy_strs= [energy_strength,0.1, 1.0, 10.0, 100.0]  # Different energy strengths to compare

for s in energy_strs
    for r in rs
        atom1 = Atom(mass=1.0, σ=2*droplet_radius, ϵ=s)
        atom2 = Atom(mass=1.0, σ=2*droplet_radius, ϵ=s)
        
        # Molly's force function expects a displacement vector (dr)
        dr = SVector(r, 0.0) 
        
        # Call the Molly force function directly
        # We pass 1.0 as force_units to get raw numbers back
        f_vec = force(pairwise_inter[1], dr, atom1, atom2, 1.0)
        
        # Store the magnitude (norm) of the force vector
        push!(force_magnitudes, norm(f_vec))
    end
    push!(forces_diff_str, copy(force_magnitudes))
    empty!(force_magnitudes)
end

# 5. Plot using GLMakie
fig = Figure(size=(800, 500))
ax = Axis(fig[1, 1], 
    title = "Soft Sphere Force Function (via Molly.jl)",
    xlabel = "Distance (r)", 
    ylabel = "Force Magnitude (F)",
    yscale = log10 # Log scale is helpful because the 1/r^13 rise is very steep
)
 colors = [:red, :green, :blue, :orange, :purple]
for (i, s) in enumerate(energy_strs)
    lines!(ax, rs, forces_diff_str[i], label="ϵ = $s", color=colors[i])
end
vlines!(ax, [droplet_radius], color=:black, linestyle=:dash, label="Sigma (Diameter)")

# Add a limit to avoid the graph looking empty due to the infinite rise at r=0
ylims!(ax, 1e-3, 1e4) 
axislegend(ax)
outdir = joinpath(@__DIR__, output_folder)
isdir(outdir) || mkpath(outdir) # Create directory if it doesn't exist
save(joinpath(outdir, "soft_sphere_force_plot.png"), fig)
println("Force plot saved as 'soft_sphere_force_plot.png'.")
#endregion
#region -------- Shear profile setup --------
shear_profile = make_shear_profile(gamma_fn = gamma_fn)

# Precompute a reference shear rate over the simulation span to normalize drag strength.
rate_samples = [abs(shear_profile.gamma_rate(t)) for t in range(0, stop=T, length=101)]
gamma_rate_ref = maximum(rate_samples)
gamma_rate_ref = gamma_rate_ref <= eps(Float64) ? 1.0 : gamma_rate_ref

#endregion
# Apply incremental affine shear (small step) to coords at time t, accumulating total gamma to keep it bounded.
function apply_affine_shear!(sys::Molly.System, shear_profile, t::Real, dt::Real, box_side::Real, gamma_accum::Base.RefValue{Float64})
    gamma, gamma_rate = shear_state(shear_profile, t)
    gamma_accum[] = clamp(gamma, -0.3, 0.3)
    @inbounds for i in eachindex(sys.coords)
        p = sys.coords[i]
        dx = gamma_rate * dt * p[2]   # incremental displacement
        x_new = mod(p[1] + dx, box_side)
        sys.coords[i] = SVector(x_new, p[2])
    end
end

# Instantaneous shear stress σ_xy = (kinetic + virial) / area for soft spheres.
function shear_stress_soft(sys::Molly.System, box_side::Float64)
    A = box_side^2
    coords = sys.coords
    velocities = sys.velocities
    atoms = sys.atoms
    N = length(coords)

    sigma_kin = 0.0
    @inbounds for i in 1:N
        sigma_kin += atoms[i].mass * velocities[i][1] * velocities[i][2]
    end

    sigma_vir = 0.0
    @inbounds for i in 1:N-1
        xi = coords[i]
        for j in i+1:N
            dx = wrap_x(coords[j][1] - xi[1], box_side)
            dy = coords[j][2] - xi[2]               # no y-wrapping for artificial walls
            dr = SVector(dx, dy)
            r = norm(dr)
            if r <= eps(Float64)
                continue
            end
            fij = force(sys.pairwise_inters[1], dr, atoms[i], atoms[j], 1.0)
            sigma_vir += dr[1] * fij[2]
        end
    end

    return (sigma_kin + sigma_vir) / A
end

# Compute storage/loss moduli from sinusoidal shear via stress response
function compute_moduli_over_history(coords_history, velocities_history, atoms, box_side, cutoff, pairwise_inter, walls; gamma_fn, save_every, dt)
    sigma_hist = Float64[]
    time_hist = Float64[]
    nframes = length(coords_history)
    N = length(coords_history[1])
    masses = [atoms[i].mass for i in 1:N]
    sys_tmp = build_soft_emulsion_system_with_artificial_walls(
        copy(coords_history[1]),
        atoms,
        box_side,
        cutoff,
        copy(velocities_history[1]),
        pairwise_inter,
        walls,
        1;
        n_threashold = skipping_neighbors_threshold,
    )
    for (idx, frame) in enumerate(coords_history)
        sys_tmp.coords .= frame
        sys_tmp.velocities .= velocities_history[idx]
        # compute stress for this frame
        sigma_xy = shear_stress_soft(sys_tmp, masses, cutoff, box_side)
        push!(sigma_hist, sigma_xy)
        push!(time_hist, (idx-1) * save_every * dt)
    end
    # Fit to sigma(t) = sigma0*sin(ωt) + sigma1*cos(ωt)
    ω = 2*pi*shear_freq
    s_sin = sum(sigma_hist .* sin.(ω .* time_hist))
    s_cos = sum(sigma_hist .* cos.(ω .* time_hist))
    norm_factor = sum(sin.(ω .* time_hist).^2)
    sigma0 = s_sin / norm_factor
    sigma1 = s_cos / norm_factor
    gamma0 = gamma_amplitude
    Gprime = sigma0 / gamma0
    Gdoubleprime = sigma1 / gamma0
    return Gprime, Gdoubleprime, time_hist, sigma_hist
end
#region ------- Build the system and run simulation -----

sys = build_soft_emulsion_system_with_artificial_walls(
    init_coord,
    droplets,
    box_side,
    cutoff,
    v0_bulk,
    pairwise_inter,
        walls,
        nsteps;
        n_threashold = skipping_neighbors_threshold,
)
#endregion
#region -------- Energy and temperature logging setup --------
energy_history = Float64[]
time_history = Float64[]
temperature_history = Float64[]
temperature_time = Float64[]
sigma_history = Float64[]
sigma_time = Float64[]
if enable_energy_logging
    # initial energies (t=0)
    E0 = total_energy(sys)
    push!(energy_history, E0)
    push!(time_history, 0.0)
end
temp0 = current_temperature(sys, n_bulk)  # proxy temperature (k_B=1)
push!(temperature_history, temp0)
push!(temperature_time, 0.0)
# initial stress
push!(sigma_history, shear_stress_soft(sys, box_side))
push!(sigma_time, 0.0)
#endregion
#region -------- Setup pre_step! and post_step! functions --------

# disabled pre wall as the wall is just a force right now 
# pre_wall! = isempty(wall_indices) ? nothing : (step_idx -> begin
#     t = step_idx == 0 ? 0.0 : (step_idx - 1) * dt
#     enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
# end)

post_wall! = enable_walls ? nothing :
   (step_idx -> begin
    t = step_idx * dt
    #enforce_wall_motion!(sys, wall_indices, wall_bases, wall_sides, shear_profile, t, wall_gap, box_side)
    #apply_wall_drag!(sys, n_bulk, wall_y_top_ref, wall_y_bot_ref, shear_profile, t, wall_gap, box_side,
    #                  dt=dt, rate_ref=gamma_rate_ref)
    #confine_bulk_y!(sys, n_bulk, wall_y_bot_ref + cutoff, wall_y_top_ref - cutoff)
    

    if enable_energy_logging && (step_idx % save_every == 0 || step_idx == nsteps)
        Etot = total_energy(sys)
        push!(energy_history, Etot)
        push!(time_history, step_idx * dt)
    end
    if step_idx % save_every == 0 || step_idx == nsteps
        temp_inst = current_temperature(sys, n_bulk)
        push!(temperature_history, temp_inst)
        push!(temperature_time, step_idx * dt)
    end
    end)

# Simple hard confinement in y to keep particles inside [0, box_side]
function confine_y!(sys::Molly.System, y_min::Float64, y_max::Float64)
    @inbounds for i in eachindex(sys.coords)
        p = sys.coords[i]; v = sys.velocities[i]
        if p[2] < y_min
            sys.coords[i] = SVector(p[1], y_min + (y_min - p[2]))
            sys.velocities[i] = SVector(v[1], -v[2])
        elseif p[2] > y_max
            sys.coords[i] = SVector(p[1], y_max - (p[2] - y_max))
            sys.velocities[i] = SVector(v[1], -v[2])
        end
    end
end

# Periodic-y hook (used when walls are disabled).
post_periodic! = periodic_y_mode ? (step_idx -> begin
    apply_periodic_y!(sys, n_bulk, box_side)
    
    if enable_energy_logging && (step_idx % save_every == 0 || step_idx == nsteps)
        Etot = total_energy(sys)
        push!(energy_history, Etot)
        push!(time_history, step_idx * dt)
    end
    if step_idx % save_every == 0 || step_idx == nsteps
        temp_inst = current_temperature(sys, n_bulk)
        push!(temperature_history, temp_inst)
        push!(temperature_time, step_idx * dt)
    end
end) : nothing

#pre_hook  = periodic_y_mode ? nothing       : pre_wall!
post_hook = periodic_y_mode ? post_periodic! : post_wall!
# Always confine after each step to keep beads in the box.
post_confine! = (step_idx -> confine_y!(sys, wall_radius, box_side - wall_radius))

# Logging hook for energy/temperature
post_log! = (step_idx -> begin
    if enable_energy_logging && (step_idx % save_every == 0 || step_idx == nsteps)
        Etot = total_energy(sys)
        push!(energy_history, Etot)
        push!(time_history, step_idx * dt)
    end
    if step_idx % save_every == 0 || step_idx == nsteps
        temp_inst = current_temperature(sys, n_bulk)
        push!(temperature_history, temp_inst)
        push!(temperature_time, step_idx * dt)
        sigma_xy = shear_stress_soft(sys, box_side)
        push!(sigma_history, sigma_xy)
        push!(sigma_time, step_idx * dt)
    end
end)
#endregion
#region -------- Run the simulation and collect coordinates -------- 
simulator = enable_langevin ?
    Langevin(dt = dt, temperature = temperature, friction = gamma_langevin) :
    VelocityVerlet(dt = dt)

# Compute save_every from desired_fps; if not enough steps to reach target fps, save every frame
# frames_target = desired_fps * T; save_every ~ nsteps / frames_target
frames_target = max(1, Int(round(desired_fps * T)))
save_every = max(1, Int(floor(nsteps / frames_target)))

if enable_freq_sweep
    Gp_sweep = Float64[]
    Gpp_sweep = Float64[]
    freq_used = Float64[]
    p_freqs = HAS_PROGRESSMETER[] ? ProgressMeter.Progress(length(sweep_freqs); desc="Freq sweep", dt=0.2) : nothing

    for f in sweep_freqs
        local_gamma_fn = t -> gamma_amplitude * sin(2*pi*f*t)
        local_profile = make_shear_profile(gamma_fn = local_gamma_fn)
        sys_freq = build_soft_emulsion_system_with_artificial_walls(
            copy(init_coord),
            droplets,
            box_side,
            cutoff,
            copy(v0_bulk),
            pairwise_inter,
            walls,
            nsteps;
            n_threashold = skipping_neighbors_threshold,
        )
        sigma_hist_local = Float64[]
        time_hist_local = Float64[]
        gamma_accum_local = Ref(0.0)
        total_steps = 0
        cycles_done = 0.0
        stable = false
        last_Gp = nothing
        last_Gpp = nothing
        total_cycles_target = min(max_cycles_per_freq, discard_cycles + cycles_per_freq)
        p_run = HAS_PROGRESSMETER[] ? ProgressMeter.Progress(max(1, Int(round((total_cycles_target/f)/dt))); desc="freq=$(round(f,digits=3))", dt=0.1) : nothing

        segment_cycles = 1  # check stability every cycle for early stop
        while !stable && cycles_done < max_cycles_per_freq
            seg_time = segment_cycles / f
            nsteps_seg = max(1, Int(round(seg_time / dt)))
            save_every_freq = max(1, Int(floor(nsteps_seg / max(1, Int(round(desired_fps * seg_time))))))
            for s in 1:nsteps_seg
                simulate!(sys_freq, simulator, 1)
                t_now = (total_steps + s) * dt
                if enable_shear
                    apply_affine_shear!(sys_freq, local_profile, t_now, dt, box_side, gamma_accum_local)
                end
                confine_y!(sys_freq, wall_radius, box_side - wall_radius)
                if s % save_every_freq == 0 || s == nsteps_seg
                    push!(sigma_hist_local, shear_stress_soft(sys_freq, box_side))
                    push!(time_hist_local, t_now)
                end
                p_run === nothing || ProgressMeter.next!(p_run)
            end
            total_steps += nsteps_seg
            cycles_done = total_steps * dt * f

            # check stability after accumulating some cycles
            if cycles_done >= max(discard_cycles + 1, 1)
                ω = 2*pi*f
                t_min = discard_cycles / f
                s_sin = 0.0; s_cos = 0.0; s_sin2 = 0.0; s_cos2 = 0.0
                for (σ, t) in zip(sigma_hist_local, time_hist_local)
                    t < t_min && continue
                    s_val = sin(ω * t); c_val = cos(ω * t)
                    s_sin += σ * s_val
                    s_cos += σ * c_val
                    s_sin2 += s_val * s_val
                    s_cos2 += c_val * c_val
                end
                if gamma_amplitude > eps(Float64) && (s_sin2 > eps(Float64) || s_cos2 > eps(Float64))
                    Gp = s_sin / (gamma_amplitude * max(s_sin2, eps(Float64)))
                    Gpp = s_cos / (gamma_amplitude * max(s_cos2, eps(Float64)))
                    if last_Gp !== nothing && last_Gpp !== nothing
                        rel_gp  = abs(last_Gp  - Gp)  / max(abs(Gp),  1e-9)
                        rel_gpp = abs(last_Gpp - Gpp) / max(abs(Gpp), 1e-9)
                        tol = stability_percent / 100
                        if rel_gp < tol && rel_gpp < tol
                            stable = true
                        end
                    end
                    last_Gp = Gp
                    last_Gpp = Gpp
                end
            end
        end

        if last_Gp !== nothing && last_Gpp !== nothing
            push!(Gp_sweep, last_Gp)
            push!(Gpp_sweep, last_Gpp)
            push!(freq_used, f)
        end
        p_freqs === nothing || ProgressMeter.next!(p_freqs)
    end

    if !isempty(freq_used)
        figSweep = GLMakie.Figure(size=(700,400))
        axSweep = GLMakie.Axis(figSweep[1,1]; xlabel="shear frequency", ylabel="modulus")
        GLMakie.lines!(axSweep, freq_used, Gp_sweep, color=:blue, label="G'")
        GLMakie.lines!(axSweep, freq_used, Gpp_sweep, color=:red, label="G''")
        GLMakie.axislegend(axSweep)
        sweep_outfile = joinpath(outdir, "storage_loss_moduli_freq_sweep.png")
        GLMakie.save(sweep_outfile, figSweep)
        println("Saved swept-frequency moduli to ", sweep_outfile)
    else
        println("No valid frequency points for sweep (check gamma_amplitude or data).")
    end

else
    sim_time = @elapsed begin
        gamma_accum = Ref(0.0)
        coords_history = run_and_collect!(
            sys,
            simulator,
            nsteps;
            save_every=save_every,
            post_step! = (s -> begin
                if enable_shear
                    apply_affine_shear!(sys, shear_profile, s*dt, dt, box_side, gamma_accum)
                end
                post_confine!(s)
                post_hook === nothing || post_hook(s)
                post_log!(s)
            end),
        )
    end
    println("Simulation completed in $sim_time seconds.")
    if isempty(coords_history) || isempty(coords_history[1])
        error("No particle coordinates recorded; check system initialization.")
    end
    #endregion
    #region ------- Visualize the results -----
    if realistic_time
        frames = length(coords_history)
        fps = max(1, Int(round(frames / T)))
    else
        fps = desired_fps 
    end
    frames = length(coords_history)
    visualize_soft_spheres_with__artificial_wall(
        coords_history,
        boundary,
        joinpath(outdir, "sim_soft_spheres_aw_vf$(round(volume_fraction_oil, digits=2))_d$(round(droplet_radius, digits=2)).mp4");
        box_side=box_side,
        framerate=fps,
        droplet_radius=droplet_radius,
        n_droplets=n_droplets,
    )

    # Compute and plot shear modulus over saved frames if requested
    if compute_modulus_history
        G_hist = Float64[]
        G_time = Float64[]
        # simple finite-difference modulus using current frame
        shear_modulus_energy_curvature = function(base_coords; h=1e-2)
            energy_at_gamma = function(gamma)
                coords_sheared = [SVector(mod(p[1] + gamma*p[2], box_side), p[2]) for p in base_coords]
                zero_vels = [SVector{2,Float64}(0.0,0.0) for _ in coords_sheared]
                sys_tmp = build_soft_emulsion_system_with_artificial_walls(
                    coords_sheared,
                    droplets,
                    box_side,
                    cutoff,
                    zero_vels,
                    pairwise_inter,
                    walls,
                    1;
                    n_threashold = skipping_neighbors_threshold,
                )
                return total_energy(sys_tmp)
            end
            E0 = energy_at_gamma(0.0)
            Eh = energy_at_gamma(h)
            Emh = energy_at_gamma(-h)
            d2E = (Eh - 2E0 + Emh) / (h*h)
            return d2E / (box_side^2)
        end
        for (idx, frame) in enumerate(coords_history)
            Gval = shear_modulus_energy_curvature(frame)
            push!(G_hist, Gval)
            push!(G_time, (idx-1) * save_every * dt)
        end
        figG = GLMakie.Figure(size=(600,400))
        axG = GLMakie.Axis(figG[1,1]; xlabel="time", ylabel="shear modulus")
        GLMakie.lines!(axG, G_time, G_hist, color=:purple)
        G_outfile = joinpath(outdir, "shear_modulus_vs_time.png")
        GLMakie.save(G_outfile, figG)
        println("Saved shear modulus plot to ", G_outfile)
    end

    # Plot energy vs time
    if enable_energy_logging && !isempty(energy_history)
        figE = GLMakie.Figure(size=(600,400))
        axE = GLMakie.Axis(figE[1,1]; xlabel="time", ylabel="total energy")
        GLMakie.lines!(axE, time_history, energy_history, color=:black)
        energy_outfile = joinpath(outdir, "energy_vs_time.png")
        GLMakie.save(energy_outfile, figE)
        println("Saved energy plot to ", energy_outfile)
    end

    # Plot storage (G') and loss (G'') moduli from sinusoidal shear response
    if compute_modulus_history && enable_shear && !isempty(sigma_history)
        if gamma_amplitude <= eps(Float64)
            println("Gamma amplitude is zero; cannot compute storage/loss moduli.")
        else
            let ω = 2*pi*shear_freq
                local s_sin = 0.0
                local s_cos = 0.0
                local s_sin2 = 0.0
                local s_cos2 = 0.0
                Gprime_hist = Float64[]
                Gloss_hist = Float64[]
                Gmod_time = Float64[]
                for (σ, t) in zip(sigma_history, sigma_time)
                    s_val = sin(ω * t)
                    c_val = cos(ω * t)
                    s_sin += σ * s_val
                    s_cos += σ * c_val
                    s_sin2 += s_val * s_val
                    s_cos2 += c_val * c_val
                    push!(Gprime_hist, s_sin / (gamma_amplitude * max(s_sin2, eps(Float64))))
                    push!(Gloss_hist,  s_cos / (gamma_amplitude * max(s_cos2, eps(Float64))))
                    push!(Gmod_time, t)
                end
                figGL = GLMakie.Figure(size=(700,400))
                axGL = GLMakie.Axis(figGL[1,1]; xlabel="time", ylabel="modulus")
                GLMakie.lines!(axGL, Gmod_time, Gprime_hist, color=:blue, label="G'")
                GLMakie.lines!(axGL, Gmod_time, Gloss_hist,  color=:red, label="G''")
                GLMakie.axislegend(axGL)
                mod_outfile = joinpath(outdir, "storage_loss_moduli_vs_time.png")
                GLMakie.save(mod_outfile, figGL)
                println("Saved storage/loss moduli plot to ", mod_outfile)
            end
        end
    end

    # Plot temperature vs time
    if !isempty(temperature_history)
        figT = GLMakie.Figure(size=(600,400))
        axT = GLMakie.Axis(figT[1,1]; xlabel="time", ylabel="kinetic temperature")
        GLMakie.lines!(axT, temperature_time, temperature_history, color=:red)
        temp_outfile = joinpath(outdir, "temperature_vs_time.png")
        GLMakie.save(temp_outfile, figT)
        println("Saved temperature plot to ", temp_outfile)
    end
end
#endregion
