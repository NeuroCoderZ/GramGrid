use gramgrid_proto::mesh::{MeshDType, MeshEnvelope, MeshOp};
use gramgrid_scheduler::mesh_router::{MeshRouter, NpuBackend, PeerDescriptor, StageProfile, TensorProfile};

fn stage(start: u16, end: u16) -> StageProfile {
    let tensor = TensorProfile { hidden_dim: 16, seq_len: 16, dtype: MeshDType::I32 };
    StageProfile { layer_start: start, layer_end: end, input: tensor, output: tensor }
}

fn peer(name: &str, profile: StageProfile) -> PeerDescriptor {
    PeerDescriptor {
        peer_id: name.into(),
        npu_backend: NpuBackend::AmdXdna,
        model_revision: [1; 32],
        stage: profile,
        consent: true,
        resident: true,
        active: true,
        share_percent: 100,
        valid_until_ms: 1000,
        compute_us: Some(100),
        window_us: 1000,
        used_us: 0,
    }
}

#[test]
fn a_resident_same_revision_npu_chain_creates_real_sized_requests() {
    let mut router = MeshRouter::new();
    let a = router.register_peer(peer("a", stage(0, 4))).unwrap();
    let b = router.register_peer(peer("b", stage(4, 8))).unwrap();
    router.observe_link(1, a, 20).unwrap();
    router.observe_link(a, b, 30).unwrap();
    let route = router.plan([1; 32], &[stage(0, 4), stage(4, 8)], 1, 10).unwrap();
    assert_eq!(route.peers, vec![a, b]);
    assert_eq!(route.estimated_us, 250);
    let request = route.request(9, 7, 0, 1, vec![0; 16 * 16 * 4]).unwrap();
    assert_eq!(request.op, MeshOp::StageForward);
    assert_eq!(request.total_steps, 2);
    assert_eq!(request.target_hash, a);
    assert_eq!(request.layer_end, 4);
    assert_eq!(MeshEnvelope::decode(&request.encode().unwrap()).unwrap(), request);
}

#[test]
fn disabled_stale_unconsented_or_over_budget_peers_are_not_routed() {
    let mutations: Vec<(&str, fn(&mut PeerDescriptor))> = vec![
        ("inactive", |p| p.active = false),
        ("not resident", |p| p.resident = false),
        ("no consent", |p| p.consent = false),
        ("zero share", |p| p.share_percent = 0),
        ("expired", |p| p.valid_until_ms = 10),
        ("unknown service", |p| p.compute_us = None),
        ("budget", |p| p.used_us = 950),
    ];
    for (name, change) in mutations {
        let mut router = MeshRouter::new();
        let mut descriptor = peer("bad", stage(0, 4));
        change(&mut descriptor);
        let handle = router.register_peer(descriptor).unwrap();
        router.observe_link(1, handle, 20).unwrap();
        assert!(router.plan([1; 32], &[stage(0, 4)], 1, 10).is_err(), "accepted {name}");
    }
}

#[test]
fn default_construction_never_allocates_reserved_route_handles() {
    let mut router = MeshRouter::default();
    assert!(router.register_peer(peer("a", stage(0, 4))).unwrap() >= 2);
}

#[test]
fn routing_minimizes_the_whole_chain_not_just_the_next_hop() {
    let mut router = MeshRouter::new();
    let a = router.register_peer(peer("near-first", stage(0, 4))).unwrap();
    let b = router.register_peer(peer("better-chain", stage(0, 4))).unwrap();
    let c = router.register_peer(peer("last", stage(4, 8))).unwrap();
    router.observe_link(1, a, 1).unwrap();
    router.observe_link(1, b, 5).unwrap();
    router.observe_link(a, c, 1000).unwrap();
    router.observe_link(b, c, 1).unwrap();
    let route = router.plan([1; 32], &[stage(0, 4), stage(4, 8)], 1, 10).unwrap();
    assert_eq!(route.peers, vec![b, c]);
    assert_eq!(route.estimated_us, 100 + 5 + 100 + 1);
}

#[test]
fn live_participation_and_reservations_are_checked_again_at_dispatch() {
    let mut router = MeshRouter::new();
    let a = router.register_peer(peer("a", stage(0, 4))).unwrap();
    router.observe_link(1, a, 20).unwrap();
    router.set_share_percent(a, 20).unwrap();
    let route = router.plan([1; 32], &[stage(0, 4)], 1, 10).unwrap();
    assert_eq!(router.reserve_step(&route, 0, 10).unwrap(), 100);
    assert_eq!(router.reserve_step(&route, 0, 10).unwrap(), 100);
    assert!(router.reserve_step(&route, 0, 10).is_err());
    router.set_share_percent(a, 0).unwrap();
    assert!(router.reserve_step(&route, 0, 10).is_err());
    assert!(router.set_share_percent(a, 101).is_err());
}

#[test]
fn request_and_terminal_result_are_bound_to_the_pinned_session_and_route() {
    let mut router = MeshRouter::new();
    let a = router.register_peer(peer("a", stage(0, 4))).unwrap();
    let b = router.register_peer(peer("b", stage(4, 8))).unwrap();
    router.observe_link(1, a, 20).unwrap();
    router.observe_link(a, b, 30).unwrap();
    let route = router.plan([1; 32], &[stage(0, 4), stage(4, 8)], 1, 10).unwrap();
    let first = route.request(9, 7, 0, 1, vec![0; 1024]).unwrap();
    route.validate_request(&first, 0, 7).unwrap();
    assert!(route.validate_request(&first, 0, 8).is_err());
    let mut bad = first.clone();
    bad.sender_hash = b;
    assert!(route.validate_request(&bad, 0, 7).is_err());
    let mut last = route.request(9, 7, 1, a, vec![0; 1024]).unwrap();
    last.flags = 1;
    last.sender_hash = b;
    last.target_hash = 1;
    route.validate_terminal(9, 7, &last).unwrap();
    last.task_id = 10;
    assert!(route.validate_terminal(9, 7, &last).is_err());
}

#[test]
fn observed_links_cannot_expand_the_registry_with_unknown_sources() {
    let mut router = MeshRouter::new();
    let a = router.register_peer(peer("a", stage(0, 4))).unwrap();
    assert!(router.observe_link(999, a, 1).is_err());
    assert!(router.observe_link(1, 999, 1).is_err());
}
