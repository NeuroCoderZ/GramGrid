//! Bounded immutable model-job records. Pending is not completed or safe to regenerate.
use gramgrid_proto::mesh::{MeshEnvelope, MeshOp, RESULT_FLAG, HEADER_LEN, MAX_PAYLOAD_BYTES};
use serde::{Deserialize, Serialize};
use std::{fs::{self, File, OpenOptions, DirBuilder}, io::{Read, Write}, path::{Path, PathBuf}};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};

type Error = String;
const MAX_JOBS: usize = 8;
const MAX_FRAME: usize = HEADER_LEN + MAX_PAYLOAD_BYTES;

#[derive(Debug, PartialEq, Eq)]
pub enum JournalClaim { New, Pending, Replay(MeshEnvelope) }

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Session { version: u8, session_tag: u64, revision: String }

pub struct ModelJournal { root: PathBuf, session: Session, _writer_lock: File }

fn check_file(path: &Path) -> Result<(), Error> {
    let info = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !info.is_file() || info.file_type().is_symlink() || info.permissions().mode() & 0o777 != 0o600 {
        return Err("journal files must be private regular files".into());
    }
    Ok(())
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, Error> {
    check_file(path)?;
    let mut bytes = Vec::new();
    File::open(path).map_err(|e| e.to_string())?.take((limit + 1) as u64)
        .read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() > limit { return Err("journal record exceeds its bound".into()); }
    Ok(bytes)
}
fn create_record(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600)
        .open(path).map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}
fn read_frame(path: &Path) -> Result<MeshEnvelope, Error> {
    MeshEnvelope::decode(&read_bounded(path, MAX_FRAME)?).map_err(str::to_string)
}
fn valid_reply(request: &MeshEnvelope, reply: &MeshEnvelope) -> Result<(), Error> {
    reply.validate().map_err(str::to_string)?;
    if reply.op != MeshOp::ModelQuery || reply.flags != RESULT_FLAG || reply.task_id != request.task_id
        || reply.session_tag != request.session_tag || reply.sender_hash != request.target_hash
        || reply.target_hash != request.sender_hash { return Err("reply is not bound to the durable request".into()); }
    Ok(())
}

impl ModelJournal {
    pub fn open(root: &Path, new_session: u64, revision: &str) -> Result<Self, Error> {
        if new_session == 0 || revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("invalid journal session or revision".into());
        }
        if !root.exists() { DirBuilder::new().recursive(true).mode(0o700).create(root).map_err(|e| e.to_string())?; }
        let info = fs::symlink_metadata(root).map_err(|e| e.to_string())?;
        if !info.is_dir() || info.file_type().is_symlink() || info.permissions().mode() & 0o777 != 0o700 {
            return Err("journal directory must be private and not a symlink".into());
        }
        let lock_path = root.join("writer.lock");
        if lock_path.exists() { check_file(&lock_path)?; }
        let lock = OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600)
            .open(&lock_path).map_err(|e| e.to_string())?;
        lock.try_lock().map_err(|e| format!("journal already has a writer: {e:?}"))?;
        let session_path = root.join("session.json");
        let session = if session_path.exists() {
            let saved: Session = serde_json::from_slice(&read_bounded(&session_path, 4096)?).map_err(|e| e.to_string())?;
            if saved.version != 1 || saved.session_tag == 0 || saved.revision != revision {
                return Err("journal session format or model revision mismatch".into());
            }
            saved
        } else {
            let saved = Session { version: 1, session_tag: new_session, revision: revision.into() };
            create_record(&session_path, &serde_json::to_vec(&saved).map_err(|e| e.to_string())?)?;
            File::open(root).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
            saved
        };
        Ok(Self { root: root.into(), session, _writer_lock: lock })
    }
    pub fn session_tag(&self) -> u64 { self.session.session_tag }
    fn path(&self, task: u128, suffix: &str) -> PathBuf { self.root.join(format!("{task:032x}.{suffix}.gg02")) }
    fn sync_directory(&self) -> Result<(), Error> { File::open(&self.root).and_then(|f| f.sync_all()).map_err(|e| e.to_string()) }
    fn validate_request(&self, request: &MeshEnvelope) -> Result<(), Error> {
        request.validate().map_err(str::to_string)?;
        if request.op != MeshOp::ModelQuery || request.flags != 0 || request.session_tag != self.session.session_tag {
            return Err("request does not belong to the durable session".into());
        }
        Ok(())
    }
    pub fn inspect(&self, request: &MeshEnvelope) -> Result<Option<JournalClaim>, Error> {
        self.validate_request(request)?;
        let input = self.path(request.task_id, "request");
        let output = self.path(request.task_id, "result");
        if input.exists() {
            let saved = read_frame(&input)?;
            if saved != *request { return Err("same durable task has different input".into()); }
            if output.exists() {
                let reply = read_frame(&output)?;
                valid_reply(request, &reply)?;
                return Ok(Some(JournalClaim::Replay(reply)));
            }
            return Ok(Some(JournalClaim::Pending));
        }
        if output.exists() || self.path(request.task_id, "pending").exists() { return Err("orphan result cannot start a new task".into()); }
        Ok(None)
    }
    pub fn begin(&self, request: &MeshEnvelope) -> Result<JournalClaim, Error> {
        if let Some(existing) = self.inspect(request)? { return Ok(existing); }
        let input = self.path(request.task_id, "request");
        let count = fs::read_dir(&self.root).map_err(|e| e.to_string())?.filter_map(Result::ok)
            .filter(|f| f.file_name().to_string_lossy().ends_with(".request.gg02")).count();
        if count >= MAX_JOBS { return Err("durable journal capacity reached; no automatic deletion".into()); }
        create_record(&input, &request.encode().map_err(str::to_string)?)?;
        self.sync_directory()?;
        Ok(JournalClaim::New)
    }
    pub fn complete(&self, request: &MeshEnvelope, reply: &MeshEnvelope) -> Result<(), Error> {
        self.validate_request(request)?;
        valid_reply(request, reply)?;
        if read_frame(&self.path(request.task_id, "request"))? != *request { return Err("completion input mismatch".into()); }
        let output = self.path(request.task_id, "result");
        if output.exists() {
            if read_frame(&output)? == *reply { return Ok(()); }
            return Err("durable result is immutable".into());
        }
        let pending = self.path(request.task_id, "pending");
        create_record(&pending, &reply.encode().map_err(str::to_string)?)?;
        fs::rename(&pending, &output).map_err(|e| e.to_string())?;
        self.sync_directory()
    }
}
