//! Isolated GG02 forwarding demo: two logical XDNA peers on ONE device.
//! Fixed integer operators only; no Transformer, model inference, or payments.
use gramgrid_proto::mesh::{MeshDType, MeshEnvelope, MeshOp, RESULT_FLAG};
use gramgrid_scheduler::mesh_router::{MeshRouter, NpuBackend, PeerDescriptor, RoutePlan, StageProfile, TensorProfile};
use serde_json::json;
use std::{collections::BTreeMap, ffi::{c_char, c_void, CStr, CString}, io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream}, path::PathBuf,
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}, thread, time::{Duration, Instant}};

type Error = String;
#[link(name = "gramgrid_xdna_mesh")]
unsafe extern "C" {
    fn gramgrid_mesh_create(image: *const c_char, insts: *const c_char) -> *mut c_void;
    fn gramgrid_mesh_load_weights(handle: *mut c_void, weights: *const i32, count: u32) -> i32;
    fn gramgrid_mesh_forward(handle: *mut c_void, input: *const i32, out: *mut i32, count: u32) -> i32;
    fn gramgrid_mesh_destroy(handle: *mut c_void);
    fn gramgrid_mesh_error() -> *const c_char;
    fn gramgrid_mesh_device_name(handle: *mut c_void) -> *const c_char;
    fn gramgrid_mesh_weight_loads(handle: *mut c_void) -> u64;
    fn gramgrid_mesh_executions(handle: *mut c_void) -> u64;
}
struct Native(*mut c_void);
// A handle has one owning thread and is never cloned or shared. The C++ API
// also serializes weight replacement and forwards with its per-handle mutex.
unsafe impl Send for Native {}
impl Drop for Native { fn drop(&mut self) { unsafe { gramgrid_mesh_destroy(self.0) }; } }
impl Native {
    fn error() -> Error { unsafe { CStr::from_ptr(gramgrid_mesh_error()).to_string_lossy().into_owned() } }
    fn new(weights: &[i32]) -> Result<Self, Error> {
        if weights.len() != 256 { return Err("unsupported weight shape".into()); }
        let build = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../native/xdna/build");
        let image = CString::new(build.join("matmul16.xclbin").to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let insts = CString::new(build.join("matmul16.insts.bin").to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let handle = unsafe { gramgrid_mesh_create(image.as_ptr(), insts.as_ptr()) };
        if handle.is_null() { return Err(Self::error()); }
        let native = Self(handle);
        if unsafe { gramgrid_mesh_load_weights(handle, weights.as_ptr(), 256) } != 0 { return Err(Self::error()); }
        Ok(native)
    }
    fn forward(&mut self, input: &[i32]) -> Result<Vec<i32>, Error> {
        if input.len() != 256 { return Err("unsupported activation shape".into()); }
        let mut out = vec![0i32; 256];
        if unsafe { gramgrid_mesh_forward(self.0, input.as_ptr(), out.as_mut_ptr(), 256) } != 0 { return Err(Self::error()); }
        Ok(out)
    }
    fn name(&self) -> String { unsafe { CStr::from_ptr(gramgrid_mesh_device_name(self.0)).to_string_lossy().into_owned() } }
}
fn profile(start: u16, end: u16) -> StageProfile {
    let tensor = TensorProfile { hidden_dim: 16, seq_len: 16, dtype: MeshDType::I32 };
    StageProfile { layer_start: start, layer_end: end, input: tensor, output: tensor }
}
fn descriptor(name: &str, revision: [u8; 32], stage: StageProfile, compute: u64) -> PeerDescriptor {
    PeerDescriptor { peer_id: name.into(), npu_backend: NpuBackend::AmdXdna,
        model_revision: revision, stage, consent: true, resident: true, active: true,
        share_percent: 100, valid_until_ms: 60_000, compute_us: Some(compute), window_us: 1_000_000, used_us: 0 }
}
fn rpc(address: SocketAddr, request: &MeshEnvelope) -> Result<MeshEnvelope, Error> {
    if !address.ip().is_loopback() { return Err("public transport disabled".into()); }
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(1)).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    request.write_to(&mut stream)?;
    MeshEnvelope::read_from(&mut stream)
}
fn bytes(values: &[i32]) -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
fn integers(data: &[u8]) -> Result<Vec<i32>, Error> {
    if data.len() != 1024 { return Err("unsupported activation bytes".into()); }
    Ok(data.chunks_exact(4).map(|v| i32::from_le_bytes(v.try_into().unwrap())).collect())
}
fn sha256(data: &[u8]) -> Result<[u8; 32], Error> {
    // Existing local crypto tool, no network and no additional Rust dependency.
    let mut child = std::process::Command::new("/usr/bin/openssl")
        .args(["dgst", "-sha256", "-binary"])
        .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped()).spawn().map_err(|e| e.to_string())?;
    let mut stdin = child.stdin.take().ok_or("crypto input unavailable")?;
    stdin.write_all(data).map_err(|e| e.to_string())?;
    drop(stdin);
    let result = child.wait_with_output().map_err(|e| e.to_string())?;
    if !result.status.success() { return Err("local SHA256 computation failed".into()); }
    result.stdout.try_into().map_err(|_| "invalid SHA256 output length".into())
}
fn reference(a: &[i32], b: &[i32]) -> Vec<i32> {
    (0..256).map(|i| (0..16).fold(0i32, |sum, k|
        sum.wrapping_add(a[(i / 16) * 16 + k].wrapping_mul(b[k * 16 + i % 16])))).collect()
}
fn control(task: u128, session: u64, sender: u32, target: u32, steps: u16, payload: Vec<u8>) -> MeshEnvelope {
    MeshEnvelope { task_id: task, session_tag: session, op: MeshOp::AgentHandoff,
        step: 0, total_steps: steps, layer_start: 0, layer_end: 0,
        hidden_dim: 0, seq_len: 0, dtype: MeshDType::Opaque, sender_hash: sender,
        target_hash: target, flags: 0, payload }
}
#[derive(Default)]
struct PeerStats { weight_loads: u64, executions: u64, replays: u64, forwards: u64, errors: u64, device: String }
struct Cached { input: Vec<u8>, output: Vec<u8>, terminal: Option<MeshEnvelope> }
struct Context {
    route: RoutePlan, session: u64, endpoints: Vec<SocketAddr>, router: Mutex<MeshRouter>,
    clock: Instant, next_link_ns: Mutex<Option<u64>>,
}
fn serve_request(request: MeshEnvelope, stage: usize, native: &mut Native,
                 cache: &mut BTreeMap<u128, Cached>, stats: &mut PeerStats, context: &Context) -> Result<MeshEnvelope, Error> {
    let route = &context.route;
    let handle = route.peers[stage];
    let expected_source = if stage == 0 { route.origin } else { route.peers[stage - 1] };
    if request.session_tag != context.session || request.target_hash != handle
        || request.sender_hash != expected_source || request.flags != 0 { return Err("unbound session or route".into()); }
    if request.op == MeshOp::AgentHandoff {
        // This transports opaque checkpoint bytes. It does not execute an agent.
        if request.total_steps == 1 {
            let mut reply = request;
            reply.flags = RESULT_FLAG;
            reply.sender_hash = handle;
            reply.target_hash = expected_source;
            return Ok(reply);
        }
        if request.total_steps != route.peers.len() as u16 || usize::from(request.step) != stage {
            return Err("unbound checkpoint step".into());
        }
        if stage + 1 < route.peers.len() {
            let mut next = request.clone();
            next.step += 1;
            next.sender_hash = handle;
            next.target_hash = route.peers[stage + 1];
            let started = Instant::now();
            let reply = rpc(context.endpoints[stage + 1], &next)?;
            *context.next_link_ns.lock().map_err(|e| e.to_string())? = Some(started.elapsed().as_nanos() as u64);
            if reply.op != MeshOp::AgentHandoff || reply.flags != RESULT_FLAG || reply.task_id != request.task_id
                || reply.session_tag != context.session || reply.sender_hash != *route.peers.last().unwrap()
                || reply.target_hash != route.origin || reply.payload != request.payload { return Err("invalid checkpoint return".into()); }
            return Ok(reply);
        }
        let mut reply = request;
        reply.flags = RESULT_FLAG;
        reply.sender_hash = handle;
        reply.target_hash = route.origin;
        return Ok(reply);
    }
    route.validate_request(&request, stage, context.session).map_err(str::to_string)?;
    if request.dtype != MeshDType::I32 || request.hidden_dim != 16 || request.seq_len != 16 {
        return Err("this NPU kernel supports only the fixed I32 test profile".into());
    }
    if let Some(saved) = cache.get(&request.task_id) {
        if saved.input != request.payload { return Err("same task with conflicting activation".into()); }
        if let Some(reply) = &saved.terminal { stats.replays += 1; return Ok(reply.clone()); }
    } else {
        if cache.len() >= 128 { return Err("bounded checkpoint cache full".into()); }
        context.router.lock().map_err(|e| e.to_string())?
            .reserve_step(route, stage, context.clock.elapsed().as_millis() as u64).map_err(str::to_string)?;
        let output = native.forward(&integers(&request.payload)?)?;
        cache.insert(request.task_id, Cached { input: request.payload.clone(), output: bytes(&output), terminal: None });
    }
    let output = cache.get(&request.task_id).unwrap().output.clone();
    let reply = if stage + 1 < route.peers.len() {
        let next = route.request(request.task_id, context.session, request.step + 1, handle, output).map_err(str::to_string)?;
        stats.forwards += 1;
        let reply = rpc(context.endpoints[stage + 1], &next)?;
        route.validate_terminal(request.task_id, context.session, &reply).map_err(str::to_string)?;
        reply
    } else {
        let mut reply = request.clone();
        let profile = route.stages[stage].output;
        reply.flags = RESULT_FLAG;
        reply.sender_hash = handle;
        reply.target_hash = route.origin;
        reply.hidden_dim = profile.hidden_dim;
        reply.seq_len = profile.seq_len;
        reply.dtype = profile.dtype;
        reply.payload = output;
        reply
    };
    cache.get_mut(&request.task_id).unwrap().terminal = Some(reply.clone());
    Ok(reply)
}
fn actor(listener: TcpListener, stage: usize, mut native: Native, context: Arc<Context>, stop: Arc<AtomicBool>) -> PeerStats {
    let mut stats = PeerStats { device: native.name(), ..PeerStats::default() };
    let mut cache = BTreeMap::new();
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((mut stream, address)) => {
                if !address.ip().is_loopback() { stats.errors += 1; continue; }
                let result = (|| -> Result<(), Error> {
                    stream.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
                    stream.set_write_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
                    stream.set_nodelay(true).map_err(|e| e.to_string())?;
                    let request = MeshEnvelope::read_from(&mut stream)?;
                    let reply = serve_request(request, stage, &mut native, &mut cache, &mut stats, &context)?;
                    reply.write_to(&mut stream)
                })();
                if let Err(error) = result { stats.errors += 1; eprintln!("mesh peer rejected request: {error}"); }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(1)),
            Err(error) => { stats.errors += 1; eprintln!("mesh listener error: {error}"); break; }
        }
    }
    stats.weight_loads = unsafe { gramgrid_mesh_weight_loads(native.0) };
    stats.executions = unsafe { gramgrid_mesh_executions(native.0) };
    stats
}
fn run() -> Result<(), Error> {
    let manifest_id = std::env::args().nth(1).ok_or("expected the fixture manifest SHA256")?;
    if manifest_id.len() != 64 || !manifest_id.bytes().all(|b| b.is_ascii_hexdigit()) { return Err("invalid manifest digest".into()); }
    let mut revision = [0u8; 32];
    for (i, byte) in revision.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&manifest_id[i * 2..i * 2 + 2], 16).map_err(|e| e.to_string())?;
    }
    let mut nonce = [0u8; 8];
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut nonce)).map_err(|e| e.to_string())?;
    let session = u64::from_le_bytes(nonce);
    if session == 0 { return Err("zero session nonce".into()); }
    let wa: Vec<i32> = (0..256).map(|i| if i / 16 == i % 16 { 2 } else if (i / 16 + i % 16) % 7 == 0 { 1 } else { 0 }).collect();
    let wb: Vec<i32> = (0..256).map(|i| if i / 16 == i % 16 { 3 } else if (i / 16 * 3 + i % 16) % 11 == 0 { -1 } else { 0 }).collect();
    let mut fixture_weights = bytes(&wa);
    fixture_weights.extend_from_slice(&bytes(&wb));
    if sha256(&fixture_weights)? != revision { return Err("manifest digest mismatch".into()); }
    let clock = Instant::now();
    let mut engines = vec![Native::new(&wa)?, Native::new(&wb)?];
    let mut compute_us = Vec::new();
    for engine in &mut engines {
        let started = Instant::now();
        let output = engine.forward(&vec![0; 256])?;
        compute_us.push((started.elapsed().as_micros() as u64).max(1));
        if output != vec![0; 256] { return Err("NPU calibration result mismatch".into()); }
    }
    let stages = vec![profile(0, 1), profile(1, 2)];
    let mut router = MeshRouter::new();
    let a = router.register_peer(descriptor("local-xdna-a", revision, stages[0], compute_us[0])).map_err(str::to_string)?;
    let b = router.register_peer(descriptor("local-xdna-b", revision, stages[1], compute_us[1])).map_err(str::to_string)?;
    let listeners = vec![TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?, TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?];
    let endpoints = listeners.iter().map(|l| l.local_addr().unwrap()).collect::<Vec<_>>();
    for listener in &listeners { listener.set_nonblocking(true).map_err(|e| e.to_string())?; }
    // Bootstrap topology is not a latency estimate; measurements follow below.
    let topology = RoutePlan { origin: 1, model_revision: revision, peers: vec![a, b], stages: stages.clone(), estimated_us: 0 };
    let context = Arc::new(Context { route: topology, session, endpoints: endpoints.clone(), router: Mutex::new(router), clock, next_link_ns: Mutex::new(None) });
    let stop = Arc::new(AtomicBool::new(false));
    let mut threads = Vec::new();
    for (i, (listener, engine)) in listeners.into_iter().zip(engines).enumerate() {
        let ctx = context.clone(); let stopping = stop.clone();
        threads.push(thread::spawn(move || actor(listener, i, engine, ctx, stopping)));
    }
    let result = (|| -> Result<serde_json::Value, Error> {
        let checkpoint = serde_json::to_vec(&json!({"version":1,"owner":"local-fixture","model_revision":manifest_id,
            "cursor":0,"previous":"fixture-checkpoint","context":"state transport, no agent execution"})).unwrap();
        let direct = control(1, session, 1, a, 1, checkpoint.clone());
        let started = Instant::now();
        let direct_reply = rpc(endpoints[0], &direct)?;
        let root_link_us = (started.elapsed().as_micros() as u64).max(1);
        if direct_reply.payload != checkpoint { return Err("direct checkpoint corrupted".into()); }
        let handoff = control(2, session, 1, a, 2, checkpoint.clone());
        let handoff_reply = rpc(endpoints[0], &handoff)?;
        if handoff_reply.payload != checkpoint || handoff_reply.step != 1 { return Err("checkpoint handoff corrupted".into()); }
        let next_link_ns = context.next_link_ns.lock().map_err(|e| e.to_string())?.ok_or("missing peer-link measurement")?;
        let route = {
            let mut router = context.router.lock().map_err(|e| e.to_string())?;
            router.observe_link(1, a, root_link_us).map_err(str::to_string)?;
            router.observe_link(a, b, (next_link_ns / 1000).max(1)).map_err(str::to_string)?;
            router.plan(revision, &stages, 1, clock.elapsed().as_millis() as u64).map_err(str::to_string)?
        };
        let mut elapsed_ns = Vec::new();
        let mut last_input = Vec::new();
        let mut last_output = Vec::new();
        let mut last_request = None;
        for task in 0..10u128 {
            let input: Vec<i32> = (0..256).map(|i| ((i + task as usize) % 17) as i32 - 8).collect();
            let request = route.request(task + 10, session, 0, 1, bytes(&input)).map_err(str::to_string)?;
            let started = Instant::now();
            let reply = rpc(endpoints[0], &request)?;
            elapsed_ns.push(started.elapsed().as_nanos() as u64);
            route.validate_terminal(request.task_id, session, &reply).map_err(str::to_string)?;
            let output = integers(&reply.payload)?;
            if output != reference(&reference(&input, &wa), &wb) { return Err("NPU chain differs from full integer reference".into()); }
            last_input = input; last_output = output; last_request = Some(request);
        }
        let replay = rpc(endpoints[0], last_request.as_ref().unwrap())?;
        if integers(&replay.payload)? != last_output { return Err("replay result mismatch".into()); }
        elapsed_ns.sort_unstable();
        Ok(json!({"manifest_id":manifest_id,"logical_npu_peers":2,"physical_npu_devices":1,
            "payload_bytes":1024,"wire_bytes":last_request.as_ref().unwrap().encode().unwrap().len(),
            "runs":10,"all_runs_verified":true,"state_handoff_bytes_preserved":true,
            "checkpoint_bytes":checkpoint.len(),"model_inference":false,"public_listener":false,"payments_enabled":false,
            "last_input":last_input,"last_output":last_output,"latency_ns_samples_sorted":elapsed_ns,
            "median_chain_ns":elapsed_ns[elapsed_ns.len()/2],"observed_service_us":compute_us,
            "observed_link_us":[root_link_us,(next_link_ns/1000).max(1)],"route_estimate_us":route.estimated_us,
            "weight_residency":"local XRT host BO; device DMA still consumes weights on each kernel call"}))
    })();
    stop.store(true, Ordering::Release);
    let stats = threads.into_iter().map(|t| t.join().map_err(|_| "NPU peer panicked".to_string())).collect::<Result<Vec<_>, _>>()?;
    let mut report = result?;
    if stats.iter().any(|s| s.weight_loads != 1 || s.executions != 11 || s.errors != 0) || stats[0].replays != 1 {
        return Err("NPU counts, replay, or actor errors disagree".into());
    }
    report["weight_loads"] = json!(stats.iter().map(|s| s.weight_loads).collect::<Vec<_>>());
    report["executions_per_peer"] = json!(stats.iter().map(|s| s.executions).collect::<Vec<_>>());
    report["devices"] = json!(stats.iter().map(|s| &s.device).collect::<Vec<_>>());
    report["peer_forward_calls"] = json!(stats[0].forwards);
    report["replay_without_recompute"] = json!(true);
    println!("{}", report);
    Ok(())
}
fn main() { if let Err(error) = run() { eprintln!("GG02 NPU pipeline failed: {error}"); std::process::exit(1); } }
