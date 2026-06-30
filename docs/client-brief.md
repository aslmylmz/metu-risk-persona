# Slack Message from Sarah Chan (METU Department of Business Administration)

#research-software-initiatives · 2 days ago

Hey,

Our **dynamic-hazard BART task is getting serious interest** from other clinical and academic research labs, but they **cannot deploy it**. Several PIs and psychologists told me they have **zero software engineering experience**—one told me, "I want to run your study, but I don't know how to open a terminal, install Python, or configure a local database, and our university IT department blocks all unauthorized server setups anyway."

I'd love us to package our task into a **completely offline, standalone measurement instrument** so other labs can deploy it easily:

- A **zero-dependency desktop executable** (.msi on Windows, .dmg on Mac) that runs natively on double-click with no external installations required.
- **10 Configurable hazard presets** (Weibull, Rayleigh, exponential, standard Lejuez uniform, etc.) so researchers have absolute design freedom.
- **Offline local persistence**—the app must write raw event telemetry (JSONL) and a pre-scored, flattened CSV file containing our 30+ advanced behavioral metrics directly to their local hard drive on session completion. No manual data-wrangling scripts allowed.
- **Neuroimaging compatibility**—the task must be optimized for low motor load (fewer pumps to reach the EV-optimum) and short runtimes (<3 minutes) so fMRI and EEG labs can adopt it within scanner time budgets.

**No internet or cloud database dependencies**—our clinical labs and IRB approvals demand 100% data privacy. The webview must have a strict Content Security Policy blocking all remote network calls, and the sidecar must communicate only on loopback.

**No SciPy in the frozen runtime bundle**—it is too heavy and frequently crashes on locked-down Windows systems during PyInstaller compilation. Implement any lognormal CDF math natively using numpy and built-in Python `math.erf` to guarantee clean cross-platform builds.

**No complicated scripting configurations**—non-technical researchers must be able to load and save their parameters via standard native file dialogs (saving a `study.json` file), which we will eventually wire to a clean visual setup wizard.

Can someone scope this out? I'd like to get our JOSS paper and the next large-sample replication study ready for this upcoming independent study.

Thanks,
Sarah