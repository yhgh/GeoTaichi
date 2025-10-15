# GeoTaichi API Surface (example coverage)

This document enumerates the public-facing APIs that GeoTaichi exposes via `from geotaichi import *`, based on how they are exercised across the repository's example suites. Each entry lists the observed call patterns so that downstream ports can preserve compatibility.

## Runtime setup

- `init(dim=3, arch="gpu", cpu_max_num_threads=0, offline_cache=True, debug=False, default_fp="float64", default_ip="int32", device_memory_GB=None, device_memory_fraction=None, kernel_profiler=False, log=True)` configures the Taichi runtime (CPU/GPU backend, precision, memory budgeting, profiler, logging).【F:geotaichi/__init__.py†L72-L148】
  - Examples call it with defaults (`init()`), GPU memory reservations (`init(device_memory_GB=4)`), 2-D solvers (`init(dim=2, device_memory_GB=2)`), or profiler toggles (`init(kernel_profiler=True)`).【F:example/mpm/Barrier/barrier.py†L3-L13】【F:example/mpm/ExternalOBJ/bunny.py†L3-L13】【F:example/mpm/ColumnCollapse/DPmaterial2DImplicit.py†L3-L16】【F:example/dem/ParticleSliding/particle_sliding.py†L3-L29】

## Solver constructors

- `MPM()` instantiates the material point method solver used throughout MPM examples.【F:example/mpm/Barrier/barrier.py†L5-L126】
- `DEM()` instantiates the discrete element solver; variants like `lsdem = DEM()` are used for level-set DEM workflows.【F:example/dem/TriaxialTest/conso.py†L5-L69】【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L5-L33】
- `DEMPM()` provides the coupled DEM–MPM framework exposing nested `dem` and `mpm` solvers for hybrid problems.【F:example/dempm/SphereImpact/plane_strain.py†L5-L184】

## MPM solver surface

- `set_configuration(...)` accepts solver-wide options such as domain extents, damping, gravity, mapping scheme (`USF`, `USL`, etc.), shape functions (`GIMP`, `QuadBSpline`), solver type (explicit/implicit), and geometry flags like `is_2DAxisy`.【F:example/mpm/Barrier/barrier.py†L7-L13】【F:example/mpm/CPT/Pile2DAxisy_dem.py†L7-L14】【F:example/mpm/ColumnCollapse/DPmaterial2DImplicit.py†L7-L15】
- `set_solver({...})` consumes a dictionary with `Timestep`, `SimulationTime`, `SaveInterval`, and optional `SavePath`/`OutputData` folders.【F:example/mpm/Barrier/barrier.py†L15-L19】【F:example/mpm/ExternalOBJ/bunny.py†L15-L20】
- `memory_allocate({...})` reserves per-problem capacities (material/particle counts, constraint pools, verlet factors). Nested dictionaries describe constraint maxima.【F:example/mpm/Barrier/barrier.py†L21-L28】【F:example/mpm/ExternalOBJ/bunny.py†L22-L31】
- `add_contact(...)` registers contact handling (`MPMContact`, `GeoContact`, or `DEMContact`) plus friction/penalty parameters.【F:example/mpm/Barrier/barrier.py†L30-L41】【F:example/mpm/CPT/Pile2DAxisy_dem.py†L32-L49】
- `add_material(model=..., material={...})` loads constitutive models (elastic, Drucker–Prager, Mohr–Coulomb variants) via keyed property dictionaries.【F:example/mpm/Barrier/barrier.py†L32-L49】【F:example/mpm/CPT/Pile2DAxisy_dem.py†L35-L49】
- `add_element(element={...})` defines grid/element topology (e.g., `R8N3D`, `Q4N2D`) and size vectors.【F:example/mpm/Barrier/barrier.py†L51-L54】【F:example/mpm/ColumnCollapse/DPmaterial2DImplicit.py†L42-L45】
- `add_region(...)` names geometric regions (2D/3D rectangles, axisymmetric slices) supplying bounding boxes, local axes, or padding for particle seeding.【F:example/mpm/Barrier/barrier.py†L56-L70】【F:example/mpm/CPT/Pile2DAxisy_dem.py†L56-L71】
- `add_body(body={"Template": ...})` populates particle bodies from region templates, specifying particle-per-cell density, body/material IDs, initial velocities, stress/traction, and velocity constraints.【F:example/mpm/Barrier/barrier.py†L72-L94】【F:example/mpm/CPT/Pile2DAxisy_dem.py†L73-L88】
- `add_body_from_file(body={...})` imports particle clouds from OBJ/TXT assets with template metadata and offsets.【F:example/mpm/ExternalOBJ/bunny.py†L47-L56】
- `add_polygons(body={...})` injects polygonal intruders (e.g., piles) with vertex files and prescribed velocities.【F:example/mpm/CPT/Pile2DAxisy_dem.py†L90-L115】
- `add_boundary_condition(boundary=[...])` applies velocity/displacement/reflection constraints over segments defined by start/end points, normals, or custom attributes like `NLevel`.【F:example/mpm/Barrier/barrier.py†L96-L124】【F:example/mpm/ColumnCollapse/DPmaterial2DImplicit.py†L67-L81】
- `add_virtual_stress_field(field={...})` superimposes confining pressures or virtual forces before stepping.【F:example/mpm/ElementTest/AxialLoadingElastic.py†L62-L72】
- `set_implicit_solver_parameters(...)` toggles implicit options such as `quasi_static` when using implicit MPM.【F:example/mpm/ColumnCollapse/DPmaterial2DImplicit.py†L7-L21】
- `select_save_data(...)` configures output channels (grid fields, particle dumps). Accepts flags like `grid=True` or default `()` for particle-only output.【F:example/mpm/Barrier/barrier.py†L126-L130】【F:example/mpm/ElementTest/DrainedMCC.py†L215-L223】
- `run(...)` advances the simulation. Examples pass booleans or callables (`gravity_field=True` or lambda-defined spatial variation).【F:example/mpm/Barrier/barrier.py†L126-L130】【F:example/mpm/ColumnCollapse/DPmaterial2DImplicit.py†L83-L87】
- `postprocessing(...)` writes VTU/background grids; optional `start_file`, `end_file`, or `write_background_grid` arguments control output batches.【F:example/mpm/Barrier/barrier.py†L128-L130】【F:example/mpm/CPT/Pile2DAxisy_dem.py†L118-L122】
- `update_particle_properties(...)` edits body-level state mid-run (e.g., impose strain rate by changing velocity).【F:example/mpm/ElementTest/TriaxialCompressionMC.py†L158-L170】
- `modify_parameters(...)` adjusts solver settings (time horizon, save cadence) before restarting `run()`.【F:example/mpm/ElementTest/TriaxialCompressionMC.py†L158-L170】

## DEM solver surface

- `set_configuration(...)` establishes DEM options: domain, boundary handling, gravity, integrator (`VelocityVerlet`, `SymplecticEuler`), search structures (`LinkedCell`), and specialized schemes like `LSDEM` with visualization flags.【F:example/dem/TriaxialTest/conso.py†L7-L24】【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L5-L24】【F:example/dem/ParticleSliding/particle_sliding.py†L7-L29】
- `set_solver({...})` matches the MPM pattern for timestep/stop time/save path dictionaries.【F:example/dem/TriaxialTest/conso.py†L26-L31】【F:example/dem/ParticleSliding/particle_sliding.py†L25-L29】
- `memory_allocate({...})` provisions DEM resources: particle/sphere/clump counts, servo/facet slots, level-set grids, coordination numbers, Verlet parameters, etc.【F:example/dem/TriaxialTest/conso.py†L13-L24】【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L13-L25】
- `add_attribute(materialID=..., attribute={...})` sets density and local damping for DEM materials.【F:example/dem/TriaxialTest/conso.py†L33-L38】【F:example/dem/GranularPackings/sphere/sphere_packing.py†L28-L46】
- `add_template(template={...})` registers level-set or rigid templates, referencing SDF objects, mesh files, or procedural shapes with metadata (surface resolution, writeback toggles).【F:example/dem/RotatingDrums/sphere_packing.py†L48-L58】【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L41-L82】
- `create_body(body={...})` instantiates rigid bodies from templates with initial states, offsets, scale factors, or body orientations.【F:example/dem/ParticleSliding/cylinder_sliding.py†L46-L58】【F:example/dempm/SphereImpact/plane_strain.py†L66-L80】
- `add_body_from_file(body={...})` loads particle assemblies (sphere packs, clumps, bounding volumes) from text/VTU data, including writer hints and initial kinematics.【F:example/dem/GranularPackings/sphere/sphere_packing.py†L37-L71】
- `add_region(...)` and `add_body(...)` support procedural packing by defining generation regions and template quotas.【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L83-L143】
- `add_wall(body={...})` installs planes, facets, patches, or digital elevation models with normals, materials, transforms, and optional mesh assets.【F:example/dem/GranularPackings/sphere/sphere_packing.py†L93-L120】【F:example/dem/DebrisFlow/sphere/debris_flow.py†L60-L95】
- `static_wall()` freezes previously added walls for terrain-style boundaries.【F:example/dem/DebrisFlow/sphere/debris_flow.py†L60-L99】
- `choose_contact_model(...)` selects particle-particle and particle-wall interaction laws (Linear, Hertz-Mindlin, Energy Conserving, Fluid Particle, etc.).【F:example/dem/TriaxialTest/conso.py†L40-L56】【F:example/dem/ParticleSliding/cylinder_sliding.py†L74-L80】
- `add_property(...)` provides contact parameters per material pair (stiffness, friction, damping).【F:example/dem/TriaxialTest/conso.py†L44-L62】【F:example/dem/GranularPackings/sphere/sphere_packing.py†L73-L92】
- `select_save_data(...)` toggles which DEM assets (particles, walls, contacts, grid/bounding surfaces) are exported per run.【F:example/dem/TriaxialTest/conso.py†L65-L107】【F:example/dem/RotatingDrums/static_drum.py†L120-L124】
- `run(...)` advances the simulation; callbacks can be passed for servo control or custom logging.【F:example/dem/TriaxialTest/conso.py†L103-L107】【F:example/dem/RotatingDrums/static_drum.py†L120-L124】
- `postprocessing(...)` writes DEM outputs to VTU/VTK or other post pipelines.【F:example/dem/TriaxialTest/conso.py†L103-L107】【F:example/dem/ParticleSliding/particle_sliding.py†L95-L108】
- `servo_switch(status="...")` toggles servo wall control during triaxial tests.【F:example/dem/TriaxialTest/conso.py†L103-L107】【F:example/dem/TriaxialTest/undrained.py†L63-L79】
- `read_restart(...)` reloads previously saved particle/wall/contact states for staged analyses.【F:example/dem/TriaxialTest/conso.py†L65-L107】
- `update_particle_properties(...)` edits per-particle attributes (e.g., assign `groupID` via region function).【F:example/dem/RotatingDrums/rotating_drum.py†L117-L124】
- `delete_particles(function=...)` removes particles matching a predicate to sculpt packings.【F:example/dem/RotatingDrums/static_drum.py†L117-L124】
- `modify_parameters(...)` changes solver duration/save rate mid-simulation before restarting `run()`.【F:example/dem/ParticleSliding/particle_sliding.py†L95-L107】

## DEM–MPM coupling surface (`DEMPM`)

- `set_configuration(...)` configures the coupling domain, scheme (`DEM-MPM`, `MPDEM`), and interaction toggles for particle/wall exchange.【F:example/dempm/SphereImpact/plane_strain.py†L8-L31】【F:example/dempm/GranularImpact/granular_impact.py†L6-L36】
- `set_solver({...})` defines the shared timestep horizon and output cadence for the coupled run.【F:example/dempm/SphereImpact/plane_strain.py†L26-L31】
- `memory_allocate({...})` reserves coordination buffers for coupling (body/wall coordination numbers, compaction ratio) in addition to calling the nested `dem`/`mpm` allocators.【F:example/dempm/SphereImpact/plane_strain.py†L33-L56】
- `choose_contact_model(...)` and `add_property(...)` tie DEM and MPM materials together with coupling stiffness, damping, and friction values.【F:example/dempm/SphereImpact/plane_strain.py†L81-L178】【F:example/dempm/DebrisFlow/debris_flow.py†L69-L127】
- `add_body(...)` performs overlap checks or additional coupling setup before stepping.【F:example/dempm/BoxSinking/box.py†L181-L188】
- `run()` marches the coupled system; nested solvers expose their own `postprocessing()` for DEM and MPM outputs.【F:example/dempm/SphereImpact/plane_strain.py†L165-L184】

The nested `dempm.dem` and `dempm.mpm` objects reuse the full DEM/MPM APIs listed above for configuring materials, templates, bodies, and boundary conditions.【F:example/dempm/SphereImpact/plane_strain.py†L13-L163】

## Geometry and signed-distance utilities

Examples rely on several SDF helpers exported via `geotaichi` to describe rigid bodies and level-set templates:

- `polyhedron(file=...).grids(...)` loads STL/OBJ meshes and samples them into level-set grids, optionally resetting transforms.【F:example/dem/ParticleSliding/cylinder_sliding.py†L40-L58】【F:example/dem/GranularPackings/polyLevelSet/packing_generate.py†L41-L74】
- `sphere(radius).grids(...)` seeds spherical templates for packings.【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L41-L82】
- `capped_cylinder(...)` combined with boolean operations (`orient`, bitwise OR) builds composite shapes for templates.【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L51-L89】
- `torus(R, r).grids(...)` provides donut-shaped rigid templates.【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L65-L107】
- `polysuperellipsoid(...)` and `polysuperquadrics(...)` produce smooth superquadric solids used for particle packs.【F:example/dem/RotatingDrums/sphere_packing.py†L48-L71】【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L71-L82】

These SDF objects expose helper methods like `.grids(space=..., extent=...)`, `.reset(False)`, and `.orient([...])` in the examples, and are typically passed into `add_template()` definitions for level-set DEM workflows.【F:example/dem/ParticleSliding/cylinder_sliding.py†L40-L58】【F:example/dem/GranularPackings/multiLSShape/mixture_generate.py†L41-L107】

