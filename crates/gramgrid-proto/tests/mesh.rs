use gramgrid_proto::mesh::{MeshDType, MeshEnvelope, MeshOp, HEADER_LEN};

fn packet() -> MeshEnvelope {
    MeshEnvelope {
        task_id: 9,
        op: MeshOp::StageForward,
        step: 0,
        total_steps: 2,
        layer_start: 0,
        layer_end: 4,
        hidden_dim: 4,
        seq_len: 2,
        sender_hash: 1,
        target_hash: 2,
        dtype: MeshDType::I32,
        flags: 0,
        session_tag: 7,
        payload: (-4i32..4).flat_map(i32::to_le_bytes).collect(),
    }
}

#[test]
fn a_typed_stage_roundtrips_with_a_fixed_little_endian_header() {
    let message = packet();
    let wire = message.encode().unwrap();
    assert_eq!(HEADER_LEN, 64);
    assert_eq!(wire.len(), HEADER_LEN + message.payload.len());
    assert_eq!(&wire[0..4], b"GG02");
    assert_eq!(&wire[4..6], &2u16.to_le_bytes());
    assert_eq!(&wire[8..24], &9u128.to_le_bytes());
    assert_eq!(MeshEnvelope::decode(&wire).unwrap(), message);
}

#[test]
fn corruption_of_routing_header_or_payload_is_detected() {
    let original = packet().encode().unwrap();
    for offset in [8, 40, 56, HEADER_LEN] {
        let mut corrupt = original.clone();
        corrupt[offset] ^= 1;
        assert!(MeshEnvelope::decode(&corrupt).is_err(), "accepted corruption at {offset}");
    }
}

#[test]
fn invalid_route_step_or_tensor_contract_cannot_be_encoded() {
    let mutations: Vec<(&str, fn(&mut MeshEnvelope))> = vec![
        ("task", |p| p.task_id = 0),
        ("sender", |p| p.sender_hash = 0),
        ("target", |p| p.target_hash = 0),
        ("session", |p| p.session_tag = 0),
        ("step", |p| p.step = p.total_steps),
        ("count", |p| p.total_steps = 0),
        ("layers", |p| p.layer_end = p.layer_start),
        ("dimension", |p| p.hidden_dim = 0),
        ("sequence", |p| p.seq_len = 0),
        ("dtype", |p| p.dtype = MeshDType::Opaque),
        ("flags", |p| p.flags = 2),
        ("bytes", |p| { p.payload.pop(); }),
    ];
    for (name, change) in mutations {
        let mut message = packet();
        change(&mut message);
        assert!(message.encode().is_err(), "accepted invalid {name}");
    }
}

#[test]
fn a_stream_can_read_exactly_one_bounded_frame() {
    let first = packet();
    let mut second = packet();
    second.task_id = 10;
    let mut bytes = Vec::new();
    first.write_to(&mut bytes).unwrap();
    second.write_to(&mut bytes).unwrap();
    let mut stream = std::io::Cursor::new(bytes);
    assert_eq!(MeshEnvelope::read_from(&mut stream).unwrap(), first);
    assert_eq!(MeshEnvelope::read_from(&mut stream).unwrap(), second);
    assert!(MeshEnvelope::read_from(&mut stream).is_err());
}

#[test]
fn opaque_agent_context_has_no_tensor_dimensions_and_is_preserved() {
    let mut message = packet();
    message.op = MeshOp::AgentHandoff;
    message.dtype = MeshDType::Opaque;
    message.layer_start = 0;
    message.layer_end = 0;
    message.hidden_dim = 0;
    message.seq_len = 0;
    message.payload = br#"{"cursor":2,"context_ref":"fixture"}"#.to_vec();
    let wire = message.encode().unwrap();
    assert_eq!(MeshEnvelope::decode(&wire).unwrap(), message);
    message.hidden_dim = 1;
    assert!(message.encode().is_err());
}

#[test]
fn malformed_lengths_short_headers_and_unsupported_types_are_rejected() {
    let wire = packet().encode().unwrap();
    for size in 0..HEADER_LEN { assert!(MeshEnvelope::decode(&wire[..size]).is_err()); }
    for offset in [4, 6, 52] {
        let mut changed = wire.clone();
        changed[offset..offset + 2].copy_from_slice(&999u16.to_le_bytes());
        assert!(MeshEnvelope::decode(&changed).is_err());
    }
    let mut huge = wire.clone();
    huge[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(MeshEnvelope::decode(&huge).is_err());
    assert!(MeshEnvelope::read_from(&mut std::io::Cursor::new(huge)).is_err());
    assert!(MeshEnvelope::decode(&wire[..wire.len() - 1]).is_err());
    let mut trailing = wire;
    trailing.push(0);
    assert!(MeshEnvelope::decode(&trailing).is_err());
}
