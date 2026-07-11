// Regenerates the type stub python/me_mkm/_me_mkm/__init__.pyi (signatures +
// docstrings) from the Rust source. Run: `cargo run --bin stub_gen`.
// On Windows the crate links libpython, so python3xx.dll must be on PATH (it
// lives in the interpreter's base prefix — `python -c "import sys;print(sys.base_prefix)"`).
use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    // INFO logging so the run reports which stub files it wrote.
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "info")).init();
    _me_mkm::stub_info()?.generate()?;
    Ok(())
}
