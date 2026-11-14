//! GG02 whole-model job service: one local FLM NPU2 model, not layer sharding.
//! Loopback, owner consent and bounded in-memory replay only; no public trust.
use gramgrid_proto::mesh::{MeshDType, MeshEnvelope, MeshOp, RESULT_FLAG};
use gramgrid_scheduler::model_journal::{JournalClaim, ModelJournal};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::{Read, Write}, net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf, process::Command, sync::{Arc, atomic::{AtomicBool, Ordering}},
    thread, time::{Duration, Instant}};

type Error = String;
const MODEL: &str = "user.GramGrid-Qwen35-08B-NPU";
const CHECKPOINT: &str = "qwen3.5:0.8b";

struct Policy { session: u64, revision: String, script: PathBuf, journal: Option<ModelJournal> }
struct Worker { cache: BTreeMap<u128, (MeshEnvelope, MeshEnvelope)>, calls: usize, refused: usize }

fn preflight() -> Result<(), Error> {
    let memory = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    let available = memory.lines().find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1)).and_then(|s| s.parse::<u64>().ok()).ok_or("unknown free memory")?;
    if available < 10 * 1024 * 1024 { return Err("insufficient memory reserve for model load".into()); }
    for entry in std::fs::read_dir("/proc").map_err(|e| e.to_string())?.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().bytes().all(|b| b.is_ascii_digit()) { continue; }
        let Ok(comm) = std::fs::read_to_string(entry.path().join("comm")) else { continue; };
        if !comm.trim().starts_with("flm") { continue; }
        let Ok(args) = std::fs::read(entry.path().join("cmdline")) else { continue; };
        let parts: Vec<_> = args.split(|&b| b == 0).collect();
        if parts.contains(&b"serve".as_slice()) && !parts.contains(&CHECKPOINT.as_bytes()) {
            return Err("another NPU model is running; do not displace it".into());
        }
    }
    Ok(())
}

fn model_forward(prompt: &str, tokens: u64, policy: &Policy) -> Result<Value, Error> {
    let output = Command::new("python3").arg(&policy.script)
        .arg("--model").arg(MODEL).arg("--max-tokens").arg(tokens.to_string())
        .arg(format!("--prompt={prompt}")).output().map_err(|e| e.to_string())?;
    if !output.status.success() { return Err("NPU pilot process failed; no compute substitution".into()); }
    if output.stdout.len() > 1024 * 1024 { return Err("oversized model receipt".into()); }
    let receipt: Value = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    if receipt["model"] != MODEL || receipt["checkpoint"] != CHECKPOINT || receipt["recipe"] != "flm"
        || receipt["npu_recipe_verified"] != true || receipt["text"].as_str().is_none_or(|s| s.trim().is_empty()) {
        return Err("unverified model recipe or empty completion".into());
    }
    Ok(receipt)
}

fn handle(request: &MeshEnvelope, policy: &Policy, worker: &mut Worker) -> Result<MeshEnvelope, Error> {
    request.validate().map_err(str::to_string)?;
    if request.op != MeshOp::ModelQuery || request.flags != 0 || request.session_tag != policy.session
        || request.sender_hash != 1 || request.target_hash != 2 {
        return Err("unbound model session or route".into());
    }
    let body: Value = serde_json::from_slice(&request.payload).map_err(|_| "invalid bounded job body")?;
    let object = body.as_object().ok_or("job object required")?;
    if object.len() != 3 || object.keys().any(|k| !["prompt", "max_tokens", "model_revision"].contains(&k.as_str())) {
        return Err("unknown job fields".into());
    }
    let prompt = body["prompt"].as_str().ok_or("text prompt required")?;
    let tokens = body["max_tokens"].as_u64().ok_or("integer output budget required")?;
    if prompt.is_empty() || prompt.len() > 4096 || !(1..=128).contains(&tokens) {
        return Err("job budget exceeded".into());
    }
    if body["model_revision"] != policy.revision { return Err("model revision mismatch".into()); }
    if let Some((previous, reply)) = worker.cache.get(&request.task_id) {
        if previous != request { return Err("same task with conflicting input".into()); }
        return Ok(reply.clone());
    }
    if worker.cache.len() >= 8 { return Err("bounded replay cache full".into()); }
    if let Some(journal) = &policy.journal {
        match journal.inspect(request)? {
            Some(JournalClaim::Replay(reply)) => return Ok(reply),
            Some(JournalClaim::Pending) => return Err("previous task is incomplete; generation outcome is unknown".into()),
            _ => {}
        }
    }
    preflight()?;
    if let Some(journal) = &policy.journal {
        if journal.begin(request)? != JournalClaim::New { return Err("task claim changed before execution".into()); }
    }
    worker.calls += 1;
    let receipt = model_forward(prompt, tokens, policy)?;
    let mut reply = request.clone();
    reply.flags = RESULT_FLAG;
    reply.sender_hash = 2;
    reply.target_hash = 1;
    reply.payload = serde_json::to_vec(&receipt).map_err(|e| e.to_string())?;
    reply.validate().map_err(str::to_string)?;
    if let Some(journal) = &policy.journal { journal.complete(request, &reply)?; }
    worker.cache.insert(request.task_id, (request.clone(), reply.clone()));
    Ok(reply)
}

fn rpc(address: SocketAddr, request: &MeshEnvelope) -> Result<MeshEnvelope, Error> {
    if !address.ip().is_loopback() { return Err("non-loopback model RPC disabled".into()); }
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(120))).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    request.write_to(&mut stream)?;
    let reply = MeshEnvelope::read_from(&mut stream)?;
    if reply.op != MeshOp::ModelQuery || reply.flags != RESULT_FLAG || reply.task_id != request.task_id
        || reply.session_tag != request.session_tag || reply.sender_hash != request.target_hash
        || reply.target_hash != request.sender_hash { return Err("unbound model reply".into()); }
    Ok(reply)
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let state_dir = match args.as_slice() {
        [consent] if consent == "--consent" => None,
        [consent, option, path] if consent == "--consent" && option == "--state-dir" => Some(PathBuf::from(path)),
        _ => return Err("explicit --consent and optional --state-dir PATH are required".into()),
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("docs/evidence/qwen08-model-budget.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let revision = manifest["npu_package"]["revision"].as_str().ok_or("declared model revision is missing")?.to_string();
    let mut nonce = [0u8; 8];
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut nonce)).map_err(|e| e.to_string())?;
    let session = u64::from_le_bytes(nonce);
    if session == 0 { return Err("zero session nonce".into()); }
    let journal = state_dir.as_ref().map(|path| ModelJournal::open(path, session, &revision)).transpose()?;
    let session = journal.as_ref().map_or(session, ModelJournal::session_tag);
    let policy = Policy { session, revision: revision.clone(), script: root.join("scripts/npu_llm_pilot.py"), journal };
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let address = listener.local_addr().map_err(|e| e.to_string())?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let stopping = stop.clone();
    let thread = thread::spawn(move || {
        let mut worker = Worker { cache: BTreeMap::new(), calls: 0, refused: 0 };
        while !stopping.load(Ordering::Acquire) {
            match listener.accept() {
                Ok((mut stream, peer)) => {
                    if !peer.ip().is_loopback() { worker.refused += 1; continue; }
                    let result = (|| -> Result<(), Error> {
                        stream.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
                        stream.set_write_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
                        let request = MeshEnvelope::read_from(&mut stream)?;
                        let reply = match handle(&request, &policy, &mut worker) {
                            Ok(reply) => reply,
                            Err(error) => {
                                worker.refused += 1;
                                let mut reply = request;
                                reply.flags = RESULT_FLAG; reply.sender_hash = 2; reply.target_hash = 1;
                                reply.payload = serde_json::to_vec(&json!({"error":error})).unwrap(); reply
                            }
                        };
                        reply.write_to(&mut stream)
                    })();
                    if let Err(error) = result { eprintln!("model node refused: {error}"); }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(2)),
                Err(_) => break,
            }
        }
        worker
    });
    let result = (|| -> Result<Value, Error> {
        let request = MeshEnvelope { task_id: 1, op: MeshOp::ModelQuery, step: 0, total_steps: 1,
            layer_start: 0, layer_end: 0, hidden_dim: 0, seq_len: 0, sender_hash: 1, target_hash: 2,
            dtype: MeshDType::Opaque, flags: 0, session_tag: session,
            payload: serde_json::to_vec(&json!({"prompt":"Напиши только код Python функции sum_even(nums), возвращающей сумму чётных чисел списка. Без объяснений.","max_tokens":128,"model_revision":revision})).unwrap() };
        let started = Instant::now();
        let first = rpc(address, &request)?;
        let elapsed = started.elapsed().as_secs_f64();
        let receipt: Value = serde_json::from_slice(&first.payload).map_err(|e| e.to_string())?;
        if receipt["text"].as_str().is_none_or(|s| s.trim().is_empty()) { return Err("model node returned no text".into()); }
        let replay = rpc(address, &request)?;
        let mut digest = Command::new("/usr/bin/openssl").args(["dgst", "-sha256"])
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        let mut input = digest.stdin.take().ok_or("digest input unavailable")?;
        input.write_all(&first.encode().map_err(str::to_string)?).map_err(|e| e.to_string())?;
        drop(input);
        let hashed = digest.wait_with_output().map_err(|e| e.to_string())?;
        if !hashed.status.success() { return Err("response digest failed".into()); }
        let response_digest = String::from_utf8(hashed.stdout).map_err(|e| e.to_string())?
            .split_whitespace().last().ok_or("response digest missing")?.to_string();
        if response_digest.len() != 64 || !response_digest.bytes().all(|b| b.is_ascii_hexdigit()) { return Err("invalid response digest".into()); }
        let mut changed = request.clone(); changed.payload.push(b' ');
        let changed_input_refused = serde_json::from_slice::<Value>(&rpc(address, &changed)?.payload).map_err(|e| e.to_string())?["error"].is_string();
        let mut foreign = request.clone(); foreign.session_tag ^= 1;
        if foreign.session_tag == 0 { foreign.session_tag = 2; }
        let foreign_session_refused = serde_json::from_slice::<Value>(&rpc(address, &foreign)?.payload).map_err(|e| e.to_string())?["error"].is_string();
        let mut wrong_revision = request.clone(); wrong_revision.task_id = 2;
        wrong_revision.payload = serde_json::to_vec(&json!({"prompt":"fixture","max_tokens":16,"model_revision":"0".repeat(40)})).unwrap();
        let foreign_revision_refused = serde_json::from_slice::<Value>(&rpc(address, &wrong_revision)?.payload).map_err(|e| e.to_string())?["error"].is_string();
        if !changed_input_refused || !foreign_session_refused || !foreign_revision_refused || replay != first { return Err("model job binding or replay failed".into()); }
        Ok(json!({"text":receipt["text"],"recipe":receipt["recipe"],"checkpoint":receipt["checkpoint"],"usage":receipt["usage"],
            "model":MODEL,"declared_model_revision":revision,"wall_seconds":elapsed,"request_bytes":request.encode().unwrap().len(),
            "response_bytes":first.encode().unwrap().len(),"replay_same_response":true,"changed_input_refused":changed_input_refused,
            "foreign_session_refused":foreign_session_refused,"foreign_revision_refused":foreign_revision_refused,
            "physical_npu_devices":1,"public_listener":false,"model_layer_sharding":false,"cryptographic_peer_attestation":false,
            "response_digest":response_digest,"journal_enabled":state_dir.is_some()}))
    })();
    stop.store(true, Ordering::Release);
    let worker = thread.join().map_err(|_| "model worker panicked")?;
    let mut report = result?;
    if worker.calls > 1 || (state_dir.is_none() && worker.calls != 1) { return Err("replay repeated NPU generation".into()); }
    report["model_calls"] = json!(worker.calls);
    report["replay_model_calls"] = json!(0);
    report["persistent_replay_after_restart"] = json!(state_dir.is_some() && worker.calls == 0);
    report["refused_requests"] = json!(worker.refused);
    println!("{}", report);
    Ok(())
}

fn main() {
    if let Err(error) = run() { eprintln!("model-query node failed: {error}"); std::process::exit(1); }
}
