using Unitful
using Random
using StaticArrays
using Molly
using Statistics
using DelimitedFiles
using GLMakie
import LinearAlgebra: norm


const SEED = 1234

# -----------------------------
# Potential choice
# -----------------------------



# Geometry / particle parameters
const R     = 0.025           # geometric disk radius 0.25,0.1,0.05, 0.025
const EPS   = 1.0
const MASS  = 1.0
const SIGMA = 2.0 * R       # particle diameter
const RC_SOFT = 1.0 * SIGMA
const EPS_SOFT = EPS / 4

# Compression of box path
const PHI_START     = 0.10
const PHI_END       = 0.94
const PHI_FINE_FROM = 0.80
const DPHI_COARSE   = 0.01
const DPHI_FINE     = 0.005

# Initial box size (defines N via PHI_START)
const L0 = 8.0

# Dynamics (NVT)
const DT        = 2e-4
const TARGET_T  = 1.0   # reduced units
const FRICTION  = 5.0

# Quasi-static relaxation + averaging at each phi
const EQUIL_STEPS  = 100_000 #increasing does not change much
const SAMPLE_STEPS = 100_000 #increasing does not change much
const SAMPLE_STRIDE = 10    # record every N steps

# Initial placement (avoid pathological overlaps)
const MIN_DIST_FRAC = 0.85  # min_dist ≈ MIN_DIST_FRAC * SIGMA

const OUTDIR = joinpath(@__DIR__, "osmotic_out")

# -----------------------------
# Helpers
# -----------------------------
@inline phi_from_N_L(N::Int, R::Float64, L::Float64) = (N * pi * R^2) / (L^2)

function build_phi_targets()
    phis = unique(sort(vcat(
        collect(PHI_START:DPHI_COARSE:min(PHI_FINE_FROM, PHI_END)),
        collect(max(PHI_FINE_FROM, PHI_START):DPHI_FINE:PHI_END)
    )))
    return phis
end

function make_neighbor_finder(coords, boundary, cutoff, N)
    return Molly.CellListMapNeighborFinder(
        eligible    = trues(N, N),
        dist_cutoff = cutoff,
        x0          = coords,
        unit_cell   = boundary,
        n_steps     = 1, 
        dims        = 2,
    )
end

"Scale coords for isotropic box scaling L_old -> L_new (periodic)."
function scale_box!(sys::Molly.System, L_old::Float64, L_new::Float64)
    s = L_new / L_old
    @inbounds for i in eachindex(sys.coords)
        x = sys.coords[i]
        sys.coords[i] = SVector(mod(x[1] * s, L_new), mod(x[2] * s, L_new))
    end
    return nothing
end

"(reduced units with k_B = 1)."
function assign_velocities!(sys::Molly.System, T::Float64; rng=Random.default_rng())
    # For reduced units: <v_x^2> = T/m
    σv = sqrt(T / MASS)
    @inbounds for i in eachindex(sys.velocities)
        sys.velocities[i] = SVector(σv * randn(rng), σv * randn(rng))
    end
    return nothing
end

"2D temperature from kinetic energy with k_B=1: in 2D, K = N*T."
@inline function temperature_2d(sys::Molly.System)
    K = Molly.kinetic_energy(sys)
    N = length(sys.atoms)
    return K / N
end

# Observables for AverageObservableLogger 
# AverageObservableLogger calls: obs(sys, buffers, neighbors, step_n; kwargs...)

"I initlaly used this, but apparently it only works for 3D. So I replaced it with obs_trP below."
function obs_pressure_trace(sys::Molly.System, buffers, neighbors, step_n::Int;
                            n_threads::Integer=Threads.nthreads(), kwargs...)
    # scalar_pressure(..., buffers=...) -> returns TRACE(pressure tensor)
    return Molly.scalar_pressure(sys, neighbors, step_n, buffers; n_threads=n_threads)
end

"I initlaly used this, but apparently it only works for 3D. So I replaced it with obs_trP below."
function obs_virial_trace(sys::Molly.System, buffers, neighbors, step_n::Int;
                          n_threads::Integer=Threads.nthreads(), kwargs...)
    # scalar_virial(...) -> returns TRACE(virial tensor)
    return Molly.scalar_virial(sys, neighbors, step_n; n_threads=n_threads)
end

function obs_kinetic(sys::Molly.System, buffers, neighbors, step_n::Int; kwargs...)
    return Molly.kinetic_energy(sys)
end

"Trace of the pressure tensor in 2D."
function obs_trP(sys::Molly.System, buffers, neighbors, step_n::Int;
                 n_threads::Integer=Threads.nthreads(), kwargs...)
    P = Molly.pressure(sys, neighbors, step_n, buffers; n_threads=n_threads)
    return P[1,1] + P[2,2]
end

function main()
    isdir(OUTDIR) || mkpath(OUTDIR)
    Random.seed!(SEED)

    phis = build_phi_targets()
    @info "Thermal (osmotic-like) compression sweep: $(length(phis)) points."

    # Choose N from PHI_START, L0, R
    droplet_area = pi * R^2
    N = floor(Int, (PHI_START * L0^2) / droplet_area)
    N < 2 && error("N too small. Increase L0 or PHI_START.")
    @info "Initial N = $N (from PHI_START=$PHI_START, L0=$L0)."

atoms = [Molly.Atom(mass=MASS, σ=SIGMA, ϵ=EPS_SOFT) for _ in 1:N]
    boundary = Molly.RectangularBoundary(SVector{2,Float64}(L0, L0))

    coords = Molly.place_atoms(
        N, boundary;
        min_dist = MIN_DIST_FRAC * SIGMA,
        max_attempts = 1_000_000
    )

    vels = [SVector{2,Float64}(0.0, 0.0) for _ in 1:N]

    pairwise_inter=(Molly.SoftSphere(cutoff=Molly.DistanceCutoff(RC_SOFT), use_neighbors=true),) 
    nf = make_neighbor_finder(coords, boundary, RC_SOFT, N)


    # Build system (no loggers by default; we attach them only during sampling)
    sys = Molly.System(
        atoms = atoms,
        coords = coords,
        velocities = vels,
        boundary = boundary,
        pairwise_inters = pairwise_inter,
        neighbor_finder = nf,
        energy_units = Unitful.NoUnits,
        force_units  = Unitful.NoUnits,
    )

    # Initialize velocities and thermostat
    assign_velocities!(sys, TARGET_T; rng=Random.default_rng())
    sim = Molly.Langevin(dt=DT, temperature=TARGET_T, friction=FRICTION)

    # Outputs
    phi_vals   = Float64[]
    p_vals     = Float64[]     # total scalar pressure (2D)
    pvir_vals  = Float64[]     # virial contribution to scalar pressure
    pkin_vals  = Float64[]     # kinetic contribution to scalar pressure
    T_vals     = Float64[]
    L_vals     = Float64[]

    L = L0

    for (k, phi_target) in enumerate(phis)
        # Set box size for target packing fraction
        L_new = sqrt(N * pi * R^2 / phi_target)

        # Compress
        scale_box!(sys, L, L_new)
        boundary_new = Molly.RectangularBoundary(SVector{2,Float64}(L_new, L_new))
        nf_new = make_neighbor_finder(sys.coords, boundary_new, RC_SOFT, N)

        # Rebuild system with new boundary + neighbor finder (keep atoms/inters)
        sys = Molly.System(
            atoms = sys.atoms,
            coords = sys.coords,
            velocities = sys.velocities,
            boundary = boundary_new,
            pairwise_inters = sys.pairwise_inters,
            neighbor_finder = nf_new,
            energy_units = Unitful.NoUnits,
            force_units  = Unitful.NoUnits,
        )
        L = L_new

        # Equilibrate at this phi
        simulate!(sys, sim, EQUIL_STEPS)
        phi_current = phi_from_N_L(N, R, L)
        # Attach averaging loggers for sampling
        #p_logger  = Molly.AverageObservableLogger(obs_pressure_trace, Float64, SAMPLE_STRIDE) #I initlaly used this, but apparently it only works for 3D. So I replaced it with obs_trP below.
        #w_logger  = Molly.AverageObservableLogger(obs_virial_trace,   Float64, SAMPLE_STRIDE) #I initlaly used this, but apparently it only works for 3D. So I replaced it with obs_trP below.
        K_logger  = Molly.AverageObservableLogger(obs_kinetic,        Float64, SAMPLE_STRIDE)
        trP_logger = Molly.AverageObservableLogger(obs_trP, Float64, SAMPLE_STRIDE)
        sys = Molly.System(
            atoms = sys.atoms,
            coords = sys.coords,
            velocities = sys.velocities,
            boundary = sys.boundary,
            pairwise_inters = sys.pairwise_inters,
            neighbor_finder = sys.neighbor_finder,
            loggers = (K=K_logger,trP=trP_logger),
            energy_units = Unitful.NoUnits,
            force_units  = Unitful.NoUnits,
        )

        # Sample with running averages; skip step 0 logging so each phi is indpeendent
        simulate!(sys, sim, SAMPLE_STEPS; run_loggers=:skipzero)

        # Extract averages (
        #p_trace_mean, p_trace_sem = values(sys.loggers.p_trace; std=true)
        #w_trace_mean, w_trace_sem = values(sys.loggers.w_trace; std=true)
        trP_mean, trP_sem = values(sys.loggers.trP; std=true)
        K_mean,   K_sem   = values(sys.loggers.K;   std=true)
        A = L^2
        phi_current = phi_from_N_L(N, R, L)
        ρ = N / A
        T = K_mean / N                     # 2D reduced: K = N*T
        neigh = Molly.find_neighbors(sys)

    "This used pressre function that only works for 3D, so I replaced it with trP below. But the overall shape is similar."
    # trP = Molly.scalar_pressure(sys, neigh)   # trace(P)
    # p = trP / 2.0                         # 2D scalar pressure

    # K = Molly.kinetic_energy(sys)             # scalar kinetic energy = tr(K)
    # p_kin = K / A                             # 2D: p_kin = tr(K)/A

    # p_vir = p - p_kin                     # consistent by construction
    # "suggestion to fix pvir being bigger than ptot (this one works aswell)"
    # p=p_trace_mean / 2.0
    # p_kin=K_mean / A
    # p_vir=p - p_kin

    A = L^2
    p     = trP_mean / (2.0)
    p_kin = K_mean   / A
    p_vir = p - p_kin

        push!(phi_vals,  phi_current)
        push!(p_vals,    p)
        push!(pvir_vals, p_vir)
        push!(pkin_vals, p_kin)
        push!(T_vals,    T)
        push!(L_vals,    L)

        if k % 10 == 0 || k == 1 || k == length(phis)
            @info "k=$k/$(length(phis))  phi=$(round(phi_current,digits=4))  p=$(p)"
        end

        # Drop loggers before next compression (so memory stayss clean)
        sys = Molly.System(
            atoms = sys.atoms,
            coords = sys.coords,
            velocities = sys.velocities,
            boundary = sys.boundary,
            pairwise_inters = sys.pairwise_inters,
            neighbor_finder = sys.neighbor_finder,
            energy_units = Unitful.NoUnits,
            force_units  = Unitful.NoUnits,
        )
    end

    # Save CSV
    csv_path = joinpath(OUTDIR, "phi_pressure_Ris$R.csv")
    header = ["phi" "p_total" "p_vir"  "L"]
    data = hcat(phi_vals, p_vals, pvir_vals,  L_vals)
    writedlm(csv_path, vcat(header, data), ',')
    @info "Saved: $csv_path"

    # Plot p vs phi
    fig1 = Figure(size=(900, 450))
    ax1  = Axis(fig1[1,1], xlabel="Packing fraction ϕ", ylabel="Total pressure p",
                title="2D softspher: pressure vs packing fraction")
    lines!(ax1, phi_vals, p_vals)
    scatter!(ax1, phi_vals, p_vals, markersize=4)
    save(joinpath(OUTDIR, "p_vs_phi.png"), fig1)

    # Plot p vs phi (semi-log)
    fig2 = Figure(size=(900, 450))
    ax2  = Axis(
        fig2[1, 1],
        xlabel = "Volume fraction ϕ",
        ylabel = "Total pressure p",
        title  = "SoftSphere: Semi-log pressure vs volume fraction",
        yscale = log10
    )

    mask = p_vals .> 0
    phi_plot = phi_vals[mask]
    p_plot   = p_vals[mask]

    lines!(ax2, phi_plot, p_plot)
    scatter!(ax2, phi_plot, p_plot, markersize=4)

    save(joinpath(OUTDIR, "p_total_vs_phi_semilog.png"), fig2)
end
main()
