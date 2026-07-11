mod memkm_rs_lib;

// Collects the #[gen_stub_*] entries into a `stub_info()` the stub_gen binary
// calls to emit python/me_mkm/_me_mkm.pyi.
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
