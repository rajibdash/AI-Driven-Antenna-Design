# AI-Driven RF device and Antenna Design

A software designer's guide to translating antenna requirements into reproducible electromagnetic (EM) simulations, optimization experiments, and evidence-backed hardware proposals.
Again, as next step here we can think about **AI-Driven RF Production Test Intelligence** which will explore how AI and machine learning can transform large-scale RF production test 
data into an actionable engineering insights. By correlating RF performance metrics with more factors, it aims to detect anomalies, support RCA, flag quality risks, and identify 
opportunities to optimize test coverage and time - helping software engineers to resolve issues faster and get higher quality products to market. A lot of research works were already 
conducted in different ways for this approach and plenty of good quality of materials are already available to understand, learn and apply for further advanced approach. 

## Additional thoughts with this guide !!!
There are more possible areas of investigation include, but are not limited to below: 
* (1) A full Production Data Understanding and Feature Engineering Analyze **large-scale RF production test datasets**. Also, include __Hybrid beamforming (digital/analog/both) design__ related production test dataset.
* (2) Identify key RF performance indicators(KPIs for good antenna design and combine/connect with Radio) and manufacturing parameters.
* (3) Develop new methods for data cleansing, normalization, and feature extraction.
* (4) AI-Based Anomaly Detection Investigate machine learning approaches for detecting abnormal test behavior.
* (5) Identify outlier devices and unusual production trends. Compare conventional statistical methods with modern AI techniques or methods.
* (6) Root Cause Analysis Support Correlate RF test results with manufacturing information, component lots, test stations, environmental conditions, and other available factors.
* (7) Explore how AI can assist engineers in narrowing down potential root causes of fialures. To study and investigate how to reduce return rate from sites as well via AI based approach.
* (8) Develop explainable analysis methods to improve engineering confidence level.
* (9) Production Quality Risk Prediction Investigate predictive models for identifying potential quality risks before products leave manufacturing.
* (10) Study what are early indicators of future failures or yield degradation. Evaluate model performance using historical production data.
* (11) Test Coverage Optimization Analyze relationships between different production test steps.
* (12) Identify redundant measurements and opportunities to streamline testing.
* (13) Evaluate potential reductions in test time while maintaining product quality and fault coverage.
* (14) Data Visualization and Engineering Insights Develop intuitive visualizations and dashboards for production data exploration.
* (15) Present AI-generated insights in a way that supports engineering decision-making. Investigate techniques for explainable AI within a production environment.
* (16) A Proof-of-Concept Implementation: Build and evaluate a proof-of-concept AI solution using real production data. Demonstrate the ability to detect anomalies, provide engineering insights, and support production optimization activities. Also assess or figure out the deployment opportunities and future industrial applications.

**Repository status:** This is a research and implementation guide, not an implemented agent, solver integration, trained model, or validated antenna. The architectures, configurations, 
project plans, and acceptance thresholds below are proposals unless explicitly attributed to an external source. No simulation results or performance improvements are claimed for this 
repository here.

**Central principle:** An LLM can interpret a specification and coordinate tools; it cannot establish that an antenna works by describing it. Numerical performance needs qualified EM 
simulation, and hardware performance needs measurement. There is no universally "perfect" agent or antenna: objectives, physical constraints, available evidence, and compute budgets 
determine the appropriate design.

## 1. What is established, and what is proposed?

| Evidence | What it supports | What it does not establish |
| --- | --- | --- |
| NASA's [Automated Antenna Design with Evolutionary Algorithms](https://ntrs.nasa.gov/api/citations/20060024675/downloads/20060024675.pdf) | Published precedent for automated, requirements-driven antenna search using evolutionary algorithms, including fabricated designs. | That an LLM autonomously interpreted those requirements, or that evolutionary optimization guarantees a global optimum. |
| [openEMS documentation](https://docs.openems.de/) and its [Python patch-antenna example](https://github.com/thliebig/openEMS/blob/master/python/Tutorials/Simple_Patch_Antenna.py) | A real open-source FDTD solver, programmable geometry/materials, port calculations, and near-field-to-far-field post-processing. | That a successful process exit, a low reflection coefficient, or a plausible plot proves hardware readiness. |
| [PyAEDT](https://aedt.docs.pyansys.com/version/stable/) and the [Ansys Antenna Toolkit](https://aedt.antenna.toolkit.docs.pyansys.com/version/stable/index.html) | Python automation of AEDT workflows and antenna-design tooling. Their upstream documentation describes software and licensing requirements. | That these tools are inherently autonomous AI agents, or that installing a Python package supplies an HFSS solver license. |
| [BoTorch](https://botorch.org/) and [Optuna](https://optuna.org/) | Existing libraries for optimization; BoTorch provides probabilistic-model/acquisition-function building blocks, and Optuna includes multi-objective examples. | That either library understands antenna physics, chooses valid materials, or supplies an EM solver. |
| The agent architecture in this README | A proposed way to connect requirement interpretation, validated tools, optimization, and engineering review. | A tested end-to-end implementation or evidence that multiple agents outperform a simpler workflow. |

The sources above establish narrower capabilities, not a complete autonomous product. Do not equate evolutionary algorithms, Bayesian optimization, surrogate ML, and LLM agents: they solve different parts of the problem.

## 2. How an agent should interpret requirements

Treat interpretation as **compiling a human specification into a typed engineering contract**, not generating geometry immediately.

1. **Extract with provenance.** Map each statement to a requirement ID, original text, source document/revision, units, and proposed verification method. Separate user requirements, cited material properties, and agent assumptions.
2. **Resolve ambiguity.** Ask what "small," "high gain," "wideband," or "efficient" means numerically. Identify missing operating bands, available volume, platform/ground plane, polarization, impedance, materials, feed arrangement, and manufacturing process. Unknowns remain unknown; do not silently select FR-4, a substrate thickness, or a gain target.
3. **Normalize and validate.** Convert GHz/Hz and mm/m explicitly. Validate positive dimensions, ordered frequency ranges, material-model validity, compatible objectives, and geometry bounds. Reject unknown configuration fields and unsupported solver capabilities.
4. **Separate constraints from preferences.** A maximum board size may be mandatory; minimizing area may be a preference. Explain tradeoffs and report infeasibility rather than quietly relaxing a constraint.
5. **Choose a bounded design family.** Recommend a patch, wire, monopole, or array only with a documented rationale. Restrict the initial search to a parameterized, reviewed template. Analytical formulas provide starting estimates, not certified final dimensions.
6. **Obtain approval.** Have an RF engineer approve the normalized specification, assumptions, verification rules, and compute budget before an optimization campaign.

An agent should retrieve information from approved solver documentation, material datasheets, and engineering references, retaining the exact source revision. Retrieval and structured output reduce some failure modes; neither guarantees factual correctness. A schema-valid configuration can still describe bad physics.

### Example configuration contract

**Illustrative YAML, not a runnable interface implemented here.** Values are synthetic learning targets, not a claim that a patch with these bounds can meet them. The 2.40-2.50 GHz interval is a study band, not permission to transmit.

```yaml
schema_version: 1
project_id: patch-study
design:
  family: rectangular_microstrip_patch
  coordinate_system: right_handed_xyz
  length_unit: mm
  ground_plane: finite_board_backside
  max_board_size_mm: [80.0, 80.0]
  patch_x_mm_bounds: [20.0, 45.0]
  patch_y_mm_bounds: [20.0, 45.0]
  feed_x_mm_bounds: [-10.0, -2.0]
  feed_y_mm: 0.0
  feed_model: z_directed_lumped_port
  substrate:
    model: synthetic_constant_dielectric
    thickness_mm: 1.524
    relative_permittivity: 3.38
    loss_tangent: 0.002
    property_source: illustrative_not_a_manufacturer_datasheet
  conductor:
    model: perfect_electric_conductor
requirements:
  - id: RF-001
    metric: s11_db
    operator: "<="
    limit: -10.0
    band_hz: [2400000000, 2500000000]
    aggregation: maximum_over_band
    frequency_step_hz: 1000000
    reference_impedance_ohm: 50.0
optimization:
  objectives: [minimize_patch_area_mm2]
  max_solver_attempts: 60
  max_parallel_jobs: 2
  max_job_wall_time_minutes: 30
  max_campaign_wall_time_hours: 8
  seed: 42
verification:
  mesh_refinement_levels: 3
  max_resonance_shift_fraction: 0.01
  max_band_s11_change_db: 0.5
  independent_final_rerun: true
approval:
  simulation_campaign: required
  fabrication: required
```

The adapter must also record its exact mesh policy, boundary/air-box settings, excitation, stopping criteria, port definition, and software versions in the resolved run configuration. It must enforce conductor/feed containment, not just independent dimension bounds. Specify how the synthetic loss tangent is represented in the chosen solver; a constant conductivity and a constant loss tangent are not interchangeable over a wide band.

This example verifies matching only. It is incomplete for hardware release: add application-specific radiation, efficiency, polarization, platform, material provenance, tolerances, and regulatory requirements before fabrication. Explicitly approve any PEC or simplified-feed approximation.

### Metrics must have unambiguous definitions

For a one-port antenna with the specified reference impedance:

```text
S11_dB(f) = 20 * log10(abs(S11(f)))
return_loss_dB(f) = -S11_dB(f)
RF-001 passes on the sampled grid if max(S11_dB(f)) <= -10
```

S11 at -10 dB corresponds to 10% reflected power; it does **not** imply 90% radiation efficiency. Accepted power can be dissipated. Do not confuse positive return loss with signed S11, directivity with gain, or gain with realized gain.

Define bandwidth as a contiguous interval meeting a specified threshold, not merely the distance between two favorable samples. A sampled sweep can miss narrow violations; refine around resonances and threshold crossings. Specify gain direction/coverage, frequency, polarization basis, efficiency definition, and angular sampling when adding those requirements.

## 3. Reference architecture for a software implementation

Start with one orchestrator and deterministic services. Use multiple agents only if an evaluation demonstrates an advantage over that baseline.

```text
Requirement text + approved documents
                  |
      Interpreter -> clarification / human approval
                  |
         Validated immutable specification
                  |
      Candidate proposer / numerical optimizer
                  |
       Geometry and policy validation
                  |
      Solver adapter -> isolated simulation worker
                  |
      Result parser -> numerical-quality checks
                  |
      Constraint evaluator -> experiment store
                  |
      Next candidate OR stop / infeasibility report
                  |
      Refined-mesh verification -> engineer approval
                  |
      Fabrication -> measurement -> discrepancy analysis
```

| Component | Responsibility and boundary |
| --- | --- |
| Requirement interpreter | Produce typed fields, citations, unresolved questions, and assumptions; cannot approve its own changes to requirements. |
| Candidate proposer | Choose parameters within the approved family and domain; may use a numerical optimizer rather than an LLM. |
| Solver adapter | Turn validated parameters into geometry, materials, mesh, ports, boundaries, and a solver job. Prefer reviewed templates over arbitrary generated scripts. |
| Job controller | Enforce time/memory/concurrency/license budgets, persist job IDs, support cancellation, and resume without duplicating paid work. |
| Numerical verifier | Check solver completion, convergence, valid frequency coverage, finite outputs, and physically meaningful results. |
| Constraint evaluator | Calculate requirements from stored numerical data. The LLM must not invent or override pass/fail values. |
| Reporter and reviewer | Link every performance claim to a run and artifact; distinguish prediction, simulation, and measurement. Humans approve fabrication and deployment. |

Expose narrow tools such as `validate_spec`, `propose_candidate`, `submit_simulation`, `get_job_status`, `read_metrics`, and `request_approval`. These are **proposed API names**, not existing repository functions. Tool schemas should constrain units, ranges, permitted template IDs, and artifact paths.

Useful job states are `queued`, `running`, `succeeded`, `failed`, `cancelled`, and `timed_out`; engineering feasibility is a separate result. A failed simulation is not an infeasible antenna, and neither is a successful design. Preserve failure logs; retry only classified transient failures within the campaign budget. Count retries against the attempt limit.

Use an idempotency key derived from the resolved configuration, template/code revision, solver version, and fidelity settings. Persist external job IDs before retry/resume. Do not reuse a cache entry after changing materials, boundaries, mesh, or post-processing.

For durable orchestration and approval checkpoints, consider [LangGraph's overview](https://docs.langchain.com/oss/python/langgraph/overview) and [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts). A plain Python state machine is sufficient for an initial prototype; an agent framework is not a physics validator.

## 4. Tools and a practical first simulation

| Need | Starting point | Important qualification |
| --- | --- | --- |
| Open-source full-wave simulation | [openEMS documentation](https://docs.openems.de/) and [upstream installation/build guidance](https://github.com/thliebig/openEMS#building) | Requires the native solver, CSXCAD, and compatible Python bindings; do not assume a generic `pip install` supplies the complete stack. |
| Commercial solver automation | [PyAEDT documentation](https://aedt.docs.pyansys.com/version/stable/) | Check Python/AEDT compatibility and licensing. |
| Parameterized antenna tooling | [Ansys Antenna Toolkit documentation](https://aedt.antenna.toolkit.docs.pyansys.com/version/stable/index.html) | Check supported antenna models and AEDT requirements; not a replacement for engineering review. |
| RF network post-processing | [scikit-rf documentation](https://scikit-rf.readthedocs.io/en/latest/) | Useful for S-parameters and measurement workflows; not a full-wave antenna solver. |
| Accessible optimization baseline | [Optuna documentation](https://optuna.readthedocs.io/en/stable/) | Supply the objective, constraints, solver adapter, persistence, and budget controls yourself. |
| Probabilistic surrogate-assisted search | [BoTorch documentation](https://botorch.org/docs/introduction) | Requires appropriate modeling and validation; uncertainty estimates are not guarantees. |

### First milestone: reproduce an upstream patch example

Use the official [Python tutorial index](https://docs.openems.de/en/latest/python/openEMS/Tutorials/index.html), [patch tutorial](https://docs.openems.de/en/latest/python/openEMS/Tutorials/Simple_Patch_Antenna.html), and [corresponding source](https://github.com/thliebig/openEMS/blob/master/python/Tutorials/Simple_Patch_Antenna.py).

1. Install openEMS and CSXCAD using the upstream instructions for your OS. Verify both imports in the intended environment.
2. Obtain the upstream tutorial at a recorded commit or release. Read it before running and use a dedicated working directory.
3. Replace its fixed temporary output location with a unique run directory before concurrent execution. Initially reproduce the tutorial's own settings, not the illustrative configuration above.
4. Run it locally, inspect geometry and solver logs, and retain numerical S11/impedance data and the far-field output when produced.
5. Repeat with mesh refinement and record differences before allowing an agent to modify parameters.

With the native libraries and Python bindings correctly installed, the initial import check is:

```bash
python -c "import CSXCAD; import openEMS; print('Bindings import successfully')"
```

This checks imports only, not a successful simulation. The upstream example calculates S11 and input impedance and conditionally evaluates a far field at an identified resonance. Its plotted pattern is labeled **directivity**; do not relabel it realized gain. It is an educational example, not a hardware qualification procedure.

**Nothing in this repository currently runs these steps automatically.** Add a solver adapter only after reproducing this baseline outside an agent loop.

## 5. Simulation projects, in increasing difficulty

These are proposed experiments, not completed projects. Evaluation counts are planning examples; measure runtime and memory before committing a budget.

### Project A: parameter sweep without an LLM

**Goal:** Understand sensitivity and establish a reproducible baseline.

Reproduce the upstream patch, then parameterize its resonant dimension and feed position. Try a 5-by-5 sweep within reviewed bounds. Keep substrate, board, boundary conditions, and excitation fixed; generate each mesh with the same documented policy.

**Deliverable:** A table of all 25 attempted candidates, failures, geometry parameters, frequency-dependent S11, resonance, and runtime, with plots backed by raw data. Re-run the best feasible candidate on a finer mesh. Do not promise full-band matching or efficiency from a resonance plot.

### Project B: constrained optimization and a surrogate

**Goal:** Reduce patch area subject to an explicit matching constraint.

Use Project A data for initial exploration. Compare random search with Optuna, or a constrained Bayesian optimization implementation using BoTorch. Reserve some of the total solver budget for independent final verification. Model the worst-band constraint or frequency-dependent response, rather than only resonance frequency.

If training a surrogate, start with a small-data baseline such as a Gaussian process rather than assuming deep learning or reinforcement learning is needed. Split data by geometry/design group to avoid putting almost identical sweep points in training and test sets. Evaluate error near constraint boundaries and reject out-of-domain proposals. Always verify finalists with the full-wave solver.

**Deliverable:** Feasible solutions or an explicit "none found within budget" report, surrogate test errors if used, and objective-versus-solver-attempt curves. Compare repeated seeded campaigns at equal fidelity and total attempt budgets, not one favorable run.

### Project C: requirement-to-configuration agent

**Goal:** Demonstrate useful autonomy without giving the LLM control over physics.

Create a small labeled set of requirement statements: complete specifications, missing units, contradictory limits, unsupported materials, and vague terms. Have the agent produce the contract or request clarification. After approval, connect it to the same adapter used in Projects A/B.

**Deliverable:** Requirement-extraction accuracy, clarification behavior, rejected invalid tool calls, traceable reports, and solver-budget adherence. Compare against hand-authored configurations. Do not use prose quality as a substitute for correctness.

### Project D: tolerance-aware design

**Goal:** Study whether a nominally feasible design is robust.

Obtain material and fabrication tolerances from actual sources, then perturb dimensions, substrate properties, and feed placement. Use a recorded sampling strategy. Without defensible probability distributions, report sensitivity/corner results, not a claimed manufacturing yield.

**Deliverable:** Nominal versus perturbed metrics, failure cases, and uncertainty limitations. If estimating yield by Monte Carlo, include sample count and a confidence interval; a small sample does not establish high reliability.

### Project E: platform-aware and measured validation

**Goal:** Investigate the simulation-to-hardware gap.

Add a finite platform, connector/feed model, and enclosure where relevant. After RF review, fabricate a candidate. Compare calibrated VNA measurements at a documented reference plane against simulated S11; use suitable radiation measurements to assess gain/pattern/efficiency.

**Deliverable:** Geometry/BOM revision, calibration/de-embedding details, raw measured data, comparison plots, and a discrepancy report. VNA matching alone cannot validate radiation performance. Fabrication and any transmission require appropriate permissions and applicable compliance review.

## 6. Optimization strategy and numerical acceptance

Prefer the simplest method that meets the compute budget:

| Situation | Reasonable initial approach | Watch for |
| --- | --- | --- |
| Few parameters, cheap evaluations | Grid or random search | Coarse grids can miss narrow feasible regions. |
| Expensive, bounded continuous search | Gaussian-process surrogate and Bayesian optimization | Poor scaling/model fit, discontinuities, and overconfident extrapolation. |
| Mixed/discrete topology choices | A suitable evolutionary or mixed-variable optimizer | Large evaluation budgets and invalid geometries. |
| Several competing goals | Constrained multi-objective search and a Pareto set | No single "best" design without stakeholder preferences. |
| Repeated sequential decisions with a justified training budget | Consider reinforcement learning only after simpler baselines | Training cost and transfer to new requirements; RL is not a prerequisite for autonomy. |

Physics-based final acceptance should include:

- **Solver validity:** Correct geometry/materials/ports, solver completion and stopping/convergence checks, finite outputs, and coverage of the entire required frequency/angle domain.
- **Discretization sensitivity:** Repeat with successive mesh refinements and inspect metric stability. The example's 1% resonance-shift and 0.5 dB S11-change limits are proposed learning thresholds, not standards. Define the resonance-selection rule and compare aligned frequency grids; no detectable resonance means the resonance check is unresolved, not passed.
- **Boundary sensitivity:** Check air-box/absorbing-boundary placement and port/reference-plane assumptions. A stable mesh cannot correct a wrong physical model.
- **Physical consistency:** Inspect applicable passivity and power-balance checks, radiation versus dissipated power, and the distinction between directivity, gain, and realized gain. Account for numerical tolerances and solver normalization.
- **Independent verification:** Re-run the selected design in a clean run directory using the approved specification. For higher assurance, consider another solver or a benchmark; hardware release still requires suitable measurements.

Stop on exhausted budgets, persistent numerical failures, or lack of feasible candidates. Report the reason and unresolved requirements. Finding no solution is valid evidence; inventing a passing result is not.

## 7. MLOps and agent operations

There are **two separate model lifecycles**: the EM surrogate predicts numerical responses; the LLM interprets requirements and selects tools. Version and evaluate them separately, and link them through a shared experiment ID.

### Minimum traceability for every candidate

Store the original and normalized requirement revisions, approved assumptions, geometry parameters, material sources, solver input, mesh/boundary/port settings, solver and dependency versions, hardware/thread settings, seed, execution status, logs, raw responses, derived metrics, and artifact hashes.

For agent runs, also record the provider/model identifier or available snapshot, prompt/template version, retrieved source revisions, tool inputs/outputs, approvals, latency, and cost. Protect proprietary geometries and requirements; do not send them to an external model without authorization. Record observable decisions, not private reasoning traces.

Use Git for source, configuration, and small manifests. Use versioned artifact storage for large simulation files, with an explicit retention policy. Record the hashes and locations of retained numerical evidence; a screenshot is not an adequate result artifact.

| Layer | Suggested tool/reference | What the implementation must add |
| --- | --- | --- |
| Dataset and pipeline versions | [DVC documentation](https://dvc.org/doc) and [pipeline introduction](https://dvc.org/doc/start/data-management/data-pipelines) | Declare code/config/data dependencies; configure and back up remote storage. |
| Experiment tracking | [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) | Log EM-specific settings, failed attempts, raw-data artifacts, and provenance. |
| Surrogate registration | [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/) | Attach dataset/split IDs, supported design domain, validation results, limitations, and promotion approval. |
| Agent observability | [MLflow tracing](https://mlflow.org/docs/latest/genai/tracing/) and [evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/) | Define requirement-extraction, tool-use, evidence, and budget metrics; built-in text evaluation is not EM verification. |
| Workflow recovery | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), or a persisted state machine | Idempotent solver submission, job reconciliation, cancellation, and approval enforcement. |

These are building blocks, not a requirement to deploy every tool. A single-user prototype can start with structured manifests and local artifacts; add shared services when collaboration or operations justify them.

### Proposed delivery pipeline

```text
spec/schema checks -> adapter/unit checks -> small solver smoke test
    -> versioned simulation dataset -> train surrogate
    -> held-out evaluation + constraint-boundary checks
    -> register candidate model -> human promotion approval
    -> shadow evaluation -> approved use -> monitor / rollback
```

Keep cheap checks in ordinary CI. Run solver-dependent tests only on workers with the required installations/resources/licenses. Store secrets outside Git. Pin environments and container images where supported; containers do not supply commercial licenses or guarantee bit-for-bit reproducibility across solver/hardware/thread configurations.

Promote a surrogate only against predetermined error and false-feasibility limits for its documented domain. Track residuals between predictions and new full-wave results. New topology families, materials, frequencies, or solver settings can invalidate that domain and should trigger review/revalidation, not automatic trust.

Monitor job failures/timeouts, compute spend, feasible-design rate, surrogate residuals, agent schema/tool errors, and unsupported claims. Roll back model, prompt, and adapter versions independently. Do not autonomously promote a retrained surrogate or rewrite approved requirements.

### Agent evaluation and operational safeguards

Maintain a fixed, reviewed evaluation set. Include missing units, incompatible constraints, fabricated material citations, malicious instructions embedded in retrieved text, tool timeouts, missing artifacts, and interrupted/resumed jobs.

| Check | Proposed acceptance behavior |
| --- | --- |
| Required field is absent | Ask for clarification or stop; never invent it. |
| Configuration violates the schema or approved bounds | Reject before submitting a solver job. |
| Solver fails or outputs are missing | Record failure; do not produce a numerical pass verdict. |
| Report contains a performance claim | Require a matching stored run/artifact and the same metric definition. |
| Campaign exceeds approved limits | Stop further submissions, cancel jobs when appropriate, and record partial results. |
| Fabrication or requirement changes are requested | Require explicit human approval. |

Use deterministic checks and RF review for engineering acceptance; an LLM judge may help assess explanations but must not be the sole judge of physical correctness. Report evaluation counts and failures rather than an unsupported "hallucination-free" score.

Treat retrieved documents and solver logs as untrusted data, not instructions. Restrict network access, filesystem scope, and available tools; isolate simulation workers; avoid arbitrary shell execution. Start with simulation-only permissions and require separate authorization for hardware actions. [NIST's Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) provides broader risk-management guidance, not an antenna certification.

## 8. Suggested implementation layout

**Proposed future structure; these files and commands do not exist yet.**

```text
configs/              # Approved specifications and campaign budgets
schemas/              # Strict input/output contracts
src/requirements/     # Interpretation, provenance, clarification
src/geometry/         # Reviewed parameterized antenna templates
src/solvers/          # openEMS or PyAEDT adapters
src/optimization/     # Baselines, constraints, surrogate-assisted search
src/verification/     # Metric extraction and numerical acceptance
src/agent/            # Tool routing, state, approvals
pipelines/            # Dataset/training/evaluation orchestration
tests/                # Contract, adapter, regression, agent evaluations
```

Keep generated datasets, solver outputs, and model binaries outside ordinary Git history. Reference them through versioned manifests or data-versioning metadata.

**Recommended implementation order:** Reproduce the upstream tutorial, build a deterministic parameter sweep, add tracking and verification, establish an optimization baseline, and only then attach a requirement-interpreting agent. Add surrogate training when accumulated data and measured solver costs justify it.

## 9. Source notes and reading order

Start with the NASA report for historical evidence, the openEMS patch tutorial for an executable external example, and Projects A-C for the software implementation path. Use the linked official MLOps documentation when introducing the corresponding lifecycle component.

Sources were reviewed on **2026-11-15**. Upstream documentation and APIs can change: pin versions/commits for actual experiments. Direct retrieval of several hosted documentation sites was unavailable during preparation; core tool capabilities were cross-checked against upstream project documentation/source, and the NASA report was located through its indexed primary-source link. This is not a claim that every external URL was live-tested or every external example executed.

External tools retain their own licenses and prerequisites; this repository's [MIT license](LICENSE) does not override them. The link collection is curated for this workflow, not an exhaustive bibliography or endorsement of a turnkey autonomous hardware-design product.

## 10. Repository PDF synthesis and investigation approach

The two repository PDFs are useful as research inputs, not as evidence that this repository has implemented or independently reproduced their methods:

- [AI-Driven RF device and Antenna Design.pdf](<AI-Driven RF device and Antenna Design.pdf>) describes antenna-design criteria and iterative design workflows. Its Figure 2 groups requirements into transmission, radiation-pattern, mechanical, reflection, and isolation indicators, then shows topology selection, full-wave simulation, parameter adjustment, and a requirements check before a final design.
- [AI Based Antenna Design NGN.pdf](<AI Based Antenna Design NGN.pdf>) proposes a data-to-optimization pipeline: define a topology-specific design space, generate full-wave simulation data, train a surrogate, generate inverse-design candidates, and use a hybrid optimizer with periodic full-wave checks. Its discussion also notes topology-specific model limitations and possible generative-model mode collapse.

### Translate the PDF workflows into implementation decisions

| Investigation stage | Consolidated information from the PDFs | Application to this repository |
| --- | --- | --- |
| Define what “meets requirements” means | Figure 2 in *AI-Driven RF device and Antenna Design* extends beyond matching: transmission behavior, radiation pattern, mechanical/environmental constraints, reflection, and coupling/isolation may all matter. | Extend the approved requirement contract only with metrics relevant to the intended antenna and platform. Give each metric units, operating range, limit, provenance, and a verification method; resolve overlaps such as reflection coefficient, S11, and VSWR before optimization. |
| Select a design family | The workflow diagram puts antenna-type/structure selection before parameter tuning; the dissertation treats different antenna topologies as distinct design spaces. | Keep an RF engineer responsible for selecting and approving the initial topology and parameterized template. Treat automatic topology recommendation as a separate experiment, not a prerequisite for the first implementation. |
| Establish simulation evidence | The dissertation describes sampling bounded geometry and evaluating it with full-wave EM simulation before training its surrogate. Its example uses Latin hypercube sampling and separate topology datasets. | Start with one reviewed topology and a deterministic, recorded simulation campaign. Store the resolved geometry, materials, solver setup, raw outputs, and failures with every sample. Choose sampling and dataset size from measured cost and coverage; do not assume the dissertation’s sample count transfers to this problem. |
| Train and use a surrogate | Both PDFs describe ML as a way to accelerate repeated performance evaluations, with full-wave simulation retained for checking or refining candidates. | Evaluate the surrogate on held-out designs and near constraint boundaries. Use it to prioritize bounded candidates, not to certify feasibility; send finalists to the solver and add informative simulation results when model errors matter. |
| Consider inverse design and advanced optimization | The dissertation proposes a conditional GAN to seed particle-swarm optimization and periodic HFSS verification. The other PDF’s pixelated-filter examples use a different RF design problem. | First compare a simple search baseline with a surrogate-assisted optimizer under the same solver-call budget. Explore generative initialization only if it improves that comparison and generated geometries pass explicit validity checks. Keep pixelated filters, RF circuits, scattering inversion, and live system adjustment outside the initial antenna-design scope. |
| Verify and report the result | The antenna-design diagram ends its adjustment loop only when requirements are met; the dissertation describes fabrication and measurement as a further validation stage. | Separate optimizer predictions, full-wave results, and physical measurements in reports. Re-run selected designs with the approved verification settings; require appropriate measurement evidence before making hardware-performance claims. |

### Diagram-derived investigation loop

The following adapts the antenna-design loop in Figure 2 of *AI-Driven RF device and Antenna Design* and the surrogate/optimizer stages in §§3.1–3.4 of *AI Based Antenna Design NGN*. The review and evidence gates make explicit what an implementation must establish:

```text
Approved requirements
(transmission, radiation, mechanical/environmental,
 reflection, coupling/isolation as applicable)
                  |
       Engineer selects topology
                  |
   Parameter bounds + solver setup reviewed
                  |
       Full-wave simulation campaign
                  |
      Versioned data and quality checks
                  |
        Surrogate evaluation gate
        /                    \
 insufficient evidence       validated for stated domain
       |                              |
 collect targeted data       bounded candidate search
       |                              |
       +---- retrain/recheck <- geometry and constraint checks
                                      |
                         Full-wave finalist verification
                                      |
                       Requirements met and independently rerun?
                              /                 \
                            No                   Yes
                            |                     |
                  revise candidate/model    Engineer review
                                                  |
                                  optional fabrication and measurement
```

### Evidence limits and follow-up

The dissertation reports results including a 50,000-sample dataset, surrogate error below 3.5%, reduced HFSS evaluations, and 12 measured prototypes. These are claims made by that document, not results reproduced here: the repository does not include its dataset, implementation, solver projects, or raw measurement records. Use them to identify candidate techniques and questions, not as acceptance thresholds or expected outcomes. In particular, test whether a surrogate generalizes within the chosen topology before considering transfer to another family, and evaluate any GAN for geometric validity and diversity before using it to initialize search.

The RF-design PDF also discusses pixelated filters, RF circuits, electromagnetic inversion, and dynamic adjustment. Those examples suggest future research directions but do not validate an antenna-design pipeline; they need distinct requirements, data, and verification plans before being brought into scope.
