//! Routing of declared resident NPU stages. The registry is a control-plane
//! prototype, not remote hardware attestation or a public P2P discovery service.
use std::collections::BTreeMap;
use gramgrid_proto::mesh::{MeshDType, MeshEnvelope, MeshOp, MAX_PAYLOAD_BYTES};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuBackend { AmdXdna, QualcommQnn, AppleAne }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorProfile {
    pub hidden_dim: u16,
    pub seq_len: u16,
    pub dtype: MeshDType,
}
impl TensorProfile {
    fn validate(self) -> Result<(), &'static str> {
        let width = match self.dtype {
            MeshDType::I32 => 4,
            MeshDType::F16 => 2,
            MeshDType::Opaque => return Err("stage tensor cannot be opaque"),
        };
        let bytes = usize::from(self.hidden_dim).checked_mul(usize::from(self.seq_len))
            .and_then(|n| n.checked_mul(width)).ok_or("stage tensor overflow")?;
        if bytes == 0 || bytes > MAX_PAYLOAD_BYTES { return Err("stage tensor exceeds limits"); }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageProfile {
    pub layer_start: u16,
    pub layer_end: u16,
    pub input: TensorProfile,
    pub output: TensorProfile,
}
impl StageProfile {
    fn validate(self) -> Result<(), &'static str> {
        if self.layer_start >= self.layer_end { return Err("invalid half-open layer range"); }
        self.input.validate()?;
        self.output.validate()
    }
}

#[derive(Debug, Clone)]
pub struct PeerDescriptor {
    pub peer_id: String,
    pub npu_backend: NpuBackend,
    pub model_revision: [u8; 32],
    pub stage: StageProfile,
    pub consent: bool,
    pub resident: bool,
    pub active: bool,
    pub share_percent: u8,
    /// Expiry in the caller's monotonic millisecond clock domain.
    pub valid_until_ms: u64,
    /// Observed service time, never a guessed TOPS conversion.
    pub compute_us: Option<u64>,
    pub window_us: u64,
    pub used_us: u64,
}

#[derive(Debug, Clone)]
pub struct RoutePlan {
    pub origin: u32,
    pub model_revision: [u8; 32],
    pub peers: Vec<u32>,
    pub stages: Vec<StageProfile>,
    pub estimated_us: u64,
}
impl RoutePlan {
    pub fn validate_request(&self, message: &MeshEnvelope, step: usize, session_tag: u64) -> Result<(), &'static str> {
        message.validate()?;
        let profile = self.stages.get(step).ok_or("route stage not found")?;
        let target = *self.peers.get(step).ok_or("route peer not found")?;
        let source = if step == 0 { self.origin } else {
            *self.peers.get(step - 1).ok_or("previous route peer not found")?
        };
        if message.op != MeshOp::StageForward || message.flags != 0 || message.session_tag != session_tag
            || usize::from(message.step) != step || usize::from(message.total_steps) != self.stages.len()
            || message.sender_hash != source || message.target_hash != target
            || message.layer_start != profile.layer_start || message.layer_end != profile.layer_end
            || message.hidden_dim != profile.input.hidden_dim || message.seq_len != profile.input.seq_len
            || message.dtype != profile.input.dtype { return Err("GG02 request differs from pinned route"); }
        Ok(())
    }

    /// Validate terminal routing metadata, not numerical correctness.
    pub fn validate_terminal(&self, task_id: u128, session_tag: u64, message: &MeshEnvelope) -> Result<(), &'static str> {
        message.validate()?;
        let step = self.stages.len().checked_sub(1).ok_or("empty route")?;
        let profile = &self.stages[step];
        let source = *self.peers.get(step).ok_or("terminal peer not found")?;
        if message.op != MeshOp::StageForward || message.flags != 1
            || message.task_id != task_id || message.session_tag != session_tag
            || usize::from(message.step) != step || usize::from(message.total_steps) != self.stages.len()
            || message.sender_hash != source || message.target_hash != self.origin
            || message.layer_start != profile.layer_start || message.layer_end != profile.layer_end
            || message.hidden_dim != profile.output.hidden_dim || message.seq_len != profile.output.seq_len
            || message.dtype != profile.output.dtype { return Err("GG02 result differs from pinned route"); }
        Ok(())
    }

    pub fn request(&self, task_id: u128, session_tag: u64, step: u16,
                   sender: u32, payload: Vec<u8>) -> Result<MeshEnvelope, &'static str> {
        let profile = *self.stages.get(usize::from(step)).ok_or("route step not found")?;
        let target = *self.peers.get(usize::from(step)).ok_or("route peer not found")?;
        let message = MeshEnvelope {
            task_id, session_tag, op: MeshOp::StageForward, step,
            total_steps: u16::try_from(self.stages.len()).map_err(|_| "route too long")?,
            layer_start: profile.layer_start, layer_end: profile.layer_end,
            hidden_dim: profile.input.hidden_dim, seq_len: profile.input.seq_len,
            dtype: profile.input.dtype, sender_hash: sender, target_hash: target,
            flags: 0, payload,
        };
        message.validate()?;
        Ok(message)
    }
}

pub struct MeshRouter {
    peers: BTreeMap<u32, PeerDescriptor>,
    links: BTreeMap<(u32, u32), u64>,
    next_handle: u32,
}
impl Default for MeshRouter {
    fn default() -> Self { Self { peers: BTreeMap::new(), links: BTreeMap::new(), next_handle: 2 } }
}
impl PeerDescriptor {
    fn eligible_cost(&self, now_ms: u64) -> Option<u64> {
        if !self.active || !self.consent || !self.resident || self.share_percent == 0
            || self.valid_until_ms <= now_ms { return None; }
        let compute = self.compute_us?;
        if compute == 0 { return None; }
        let limit = u128::from(self.window_us) * u128::from(self.share_percent) / 100;
        if u128::from(self.used_us) + u128::from(compute) > limit { return None; }
        Some(compute)
    }
}
impl MeshRouter {
    pub fn new() -> Self { Self::default() }

    pub fn set_share_percent(&mut self, handle: u32, percent: u8) -> Result<(), &'static str> {
        if percent > 100 { return Err("participation percentage exceeds 100"); }
        self.peers.get_mut(&handle).ok_or("unknown NPU peer")?.share_percent = percent;
        Ok(())
    }

    /// Reserve the estimated compute budget. This is not hardware pre-emption,
    /// a monetary settlement, or measured completion-time accounting.
    pub fn reserve_step(&mut self, route: &RoutePlan, step: usize, now_ms: u64) -> Result<u64, &'static str> {
        let handle = *route.peers.get(step).ok_or("route peer not found")?;
        let profile = route.stages.get(step).ok_or("route stage not found")?;
        let peer = self.peers.get_mut(&handle).ok_or("unknown NPU peer")?;
        if peer.model_revision != route.model_revision || peer.stage != *profile {
            return Err("resident stage no longer matches pinned route");
        }
        let cost = peer.eligible_cost(now_ms).ok_or("NPU admission or budget refused")?;
        peer.used_us = peer.used_us.checked_add(cost).ok_or("NPU budget overflow")?;
        Ok(cost)
    }

    pub fn register_peer(&mut self, descriptor: PeerDescriptor) -> Result<u32, &'static str> {
        descriptor.stage.validate()?;
        if descriptor.peer_id.is_empty() || descriptor.peer_id.len() > 128
            || descriptor.peer_id.chars().any(char::is_control) || descriptor.model_revision == [0; 32]
            || descriptor.share_percent > 100 || descriptor.window_us == 0 {
            return Err("invalid peer descriptor");
        }
        if self.peers.len() >= 256 { return Err("peer registry limit reached"); }
        if self.peers.values().any(|p| p.peer_id == descriptor.peer_id) {
            return Err("peer identity already registered");
        }
        let handle = self.next_handle;
        self.next_handle = handle.checked_add(1).ok_or("peer handle exhausted")?;
        self.peers.insert(handle, descriptor);
        Ok(handle)
    }

    pub fn observe_link(&mut self, source: u32, target: u32, roundtrip_us: u64) -> Result<(), &'static str> {
        if (source != 1 && !self.peers.contains_key(&source)) || !self.peers.contains_key(&target) {
            return Err("unknown link endpoint");
        }
        self.links.insert((source, target), roundtrip_us);
        Ok(())
    }

    pub fn plan(&self, model_revision: [u8; 32], stages: &[StageProfile],
                origin: u32, now_ms: u64) -> Result<RoutePlan, &'static str> {
        if stages.is_empty() || stages.len() > 128 || origin == 0 || model_revision == [0; 32] {
            return Err("invalid model route");
        }
        for (index, stage) in stages.iter().enumerate() {
            stage.validate()?;
            if index > 0 && (stages[index - 1].layer_end != stage.layer_start
                || stages[index - 1].output != stage.input) {
                return Err("stage graph has a gap or incompatible tensor boundary");
            }
        }
        let mut frontier: BTreeMap<u32, (u64, Vec<u32>)> = BTreeMap::new();
        frontier.insert(origin, (0, Vec::new()));
        for stage in stages {
            let mut next = BTreeMap::new();
            for (&handle, peer) in &self.peers {
                if peer.model_revision != model_revision || peer.stage != *stage { continue; }
                let Some(compute) = peer.eligible_cost(now_ms) else { continue };
                let best = frontier.iter().filter_map(|(&source, (cost, path))| {
                    if path.contains(&handle) { return None; }
                    let link = self.links.get(&(source, handle))?;
                    let total = cost.checked_add(*link)?.checked_add(compute)?;
                    let mut route = path.clone();
                    route.push(handle);
                    Some((total, route))
                }).min_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                if let Some(best) = best { next.insert(handle, best); }
            }
            if next.is_empty() { return Err("no compatible measured NPU stage"); }
            frontier = next;
        }
        let (estimated_us, path) = frontier.into_values()
            .min_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)))
            .ok_or("no complete NPU route")?;
        Ok(RoutePlan { origin, model_revision, peers: path, stages: stages.to_vec(), estimated_us })
    }
}
