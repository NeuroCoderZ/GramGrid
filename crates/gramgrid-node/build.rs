fn main() {
    let path = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../native/xdna/build").canonicalize().expect("артефакты NPU не собраны");
    println!("cargo:rustc-link-search=native={}", path.display());
    println!("cargo:rustc-link-lib=dylib=gramgrid_xdna_v2");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
}
