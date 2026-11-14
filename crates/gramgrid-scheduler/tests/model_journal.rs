use gramgrid_proto::mesh::{MeshDType, MeshEnvelope, MeshOp, RESULT_FLAG};
use gramgrid_scheduler::model_journal::{JournalClaim, ModelJournal};
use std::{path::PathBuf, time::{SystemTime, UNIX_EPOCH}};

fn state_dir(label: &str) -> PathBuf {
    let root = std::env::var_os("TMPDIR").expect("managed TMPDIR required for fixtures");
    PathBuf::from(root).join(format!("gramgrid-journal-{label}-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()))
}
fn request(session: u64) -> MeshEnvelope {
    MeshEnvelope { task_id: 1, op: MeshOp::ModelQuery, step: 0, total_steps: 1,
        layer_start: 0, layer_end: 0, hidden_dim: 0, seq_len: 0, sender_hash: 1, target_hash: 2,
        dtype: MeshDType::Opaque, flags: 0, session_tag: session, payload: b"unit-test request, no model inference".to_vec() }
}
fn reply(request: &MeshEnvelope) -> MeshEnvelope {
    let mut result = request.clone(); result.sender_hash = 2; result.target_hash = 1;
    result.flags = RESULT_FLAG; result.payload = b"unit-test result, no model inference".to_vec(); result
}

#[test]
fn completed_reply_is_loaded_after_reopening_without_a_new_claim() {
    let path = state_dir("completed");
    let first = ModelJournal::open(&path, 7, &"1".repeat(40)).unwrap();
    let query = request(first.session_tag()); let answer = reply(&query);
    assert!(matches!(first.begin(&query).unwrap(), JournalClaim::New));
    first.complete(&query, &answer).unwrap();
    drop(first);
    let recovered = ModelJournal::open(&path, 99, &"1".repeat(40)).unwrap();
    assert_eq!(recovered.session_tag(), 7);
    match recovered.begin(&query).unwrap() {
        JournalClaim::Replay(saved) => assert_eq!(saved, answer),
        _ => panic!("completed query was not restored"),
    }
}

#[test]
fn inspection_does_not_claim_a_task_before_resources_are_admitted() {
    let path = state_dir("inspect");
    let journal = ModelJournal::open(&path, 7, &"1".repeat(40)).unwrap();
    assert_eq!(journal.inspect(&request(7)).unwrap(), None);
    assert_eq!(std::fs::read_dir(path).unwrap().count(), 2, "inspection created a task record");
}

#[test]
fn an_unfinished_request_stays_pending_after_reopening() {
    let path = state_dir("pending");
    let journal = ModelJournal::open(&path, 7, &"1".repeat(40)).unwrap();
    assert!(matches!(journal.begin(&request(7)).unwrap(), JournalClaim::New));
    drop(journal);
    let restarted = ModelJournal::open(&path, 99, &"1".repeat(40)).unwrap();
    assert_eq!(restarted.begin(&request(7)).unwrap(), JournalClaim::Pending);
}

#[test]
fn a_second_writer_cannot_enter_the_same_state_directory() {
    let path = state_dir("lock");
    let first = ModelJournal::open(&path, 7, &"1".repeat(40)).unwrap();
    assert!(ModelJournal::open(&path, 7, &"1".repeat(40)).is_err());
    drop(first);
    assert!(ModelJournal::open(&path, 7, &"1".repeat(40)).is_ok());
}

#[test]
fn conflicts_and_corrupt_saved_replies_do_not_trigger_new_claims() {
    let path = state_dir("corrupt");
    let journal = ModelJournal::open(&path, 7, &"1".repeat(40)).unwrap();
    let query = request(7);
    journal.begin(&query).unwrap(); journal.complete(&query, &reply(&query)).unwrap();
    let mut conflict = query.clone(); conflict.payload.push(1);
    assert!(journal.begin(&conflict).is_err());
    let saved = path.join("00000000000000000000000000000001.result.gg02");
    let mut bytes = std::fs::read(&saved).unwrap(); bytes[64] ^= 1;
    std::fs::write(&saved, bytes).unwrap();
    assert!(journal.begin(&query).is_err(), "corrupt result was treated as new work");
}

#[test]
fn journal_capacity_rejects_the_ninth_job_without_deleting_old_records() {
    let path = state_dir("capacity");
    let journal = ModelJournal::open(&path, 7, &"1".repeat(40)).unwrap();
    for id in 1..=8 { let mut task = request(7); task.task_id = id; journal.begin(&task).unwrap(); }
    let mut ninth = request(7); ninth.task_id = 9;
    assert!(journal.begin(&ninth).is_err());
    assert_eq!(std::fs::read_dir(path).unwrap().count(), 10);
}
