//! GG02 carries stage activations or an opaque agent-state handoff.
//! Routing handles and CRC are not identities, signatures, or proof of computation.

pub const GG02_MAGIC: &[u8; 4] = b"GG02";
pub const HEADER_LEN: usize = 64;
pub const MAX_PAYLOAD_BYTES: usize = 1024 * 1024;
pub const RESULT_FLAG: u16 = 1;

#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshOp {
    StageForward = 1,
    MoeDispatch = 2,
    AgentHandoff = 3,
    VerifyChallenge = 4,
    ModelQuery = 5,
}

#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshDType {
    I32 = 1,
    F16 = 2,
    Opaque = 3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshEnvelope {
    pub task_id: u128,
    pub op: MeshOp,
    pub step: u16,
    pub total_steps: u16,
    /// Zero-based half-open layer interval [start, end).
    pub layer_start: u16,
    pub layer_end: u16,
    pub hidden_dim: u16,
    pub seq_len: u16,
    /// Session-local route handles; not cryptographic identity hashes.
    pub sender_hash: u32,
    pub target_hash: u32,
    pub dtype: MeshDType,
    pub flags: u16,
    pub session_tag: u64,
    pub payload: Vec<u8>,
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn wire_checksum(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for (index, &byte) in bytes.iter().enumerate() {
        let value = if (48..52).contains(&index) { 0 } else { byte };
        crc ^= u32::from(value);
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

#[cfg(test)]
mod checksum_tests {
    #[test]
    fn ieee_crc32_matches_the_independent_zlib_reference() {
        assert_eq!(super::wire_checksum(b"123456789"), 0xcbf4_3926);
    }
}

impl MeshEnvelope {
    pub fn write_to(&self, writer: &mut impl std::io::Write) -> Result<(), String> {
        let wire = self.encode().map_err(str::to_string)?;
        writer.write_all(&wire).map_err(|e| e.to_string())
    }

    pub fn read_from(reader: &mut impl std::io::Read) -> Result<Self, String> {
        let mut header = [0u8; HEADER_LEN];
        reader.read_exact(&mut header).map_err(|e| e.to_string())?;
        if &header[0..4] != GG02_MAGIC || u16_at(&header, 4) != 2 {
            return Err("invalid GG02 stream header".into());
        }
        let length = u32_at(&header, 36) as usize;
        if length == 0 || length > MAX_PAYLOAD_BYTES {
            return Err("invalid GG02 stream payload length".into());
        }
        let mut wire = Vec::with_capacity(HEADER_LEN + length);
        wire.extend_from_slice(&header);
        wire.resize(HEADER_LEN + length, 0);
        reader.read_exact(&mut wire[HEADER_LEN..]).map_err(|e| e.to_string())?;
        Self::decode(&wire).map_err(str::to_string)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.task_id == 0 || self.session_tag == 0 || self.sender_hash == 0 || self.target_hash == 0 {
            return Err("GG02 identifiers must be nonzero");
        }
        if self.total_steps == 0 || self.step >= self.total_steps || self.flags & !RESULT_FLAG != 0 {
            return Err("invalid GG02 step or flags");
        }
        if self.payload.is_empty() || self.payload.len() > MAX_PAYLOAD_BYTES {
            return Err("GG02 payload exceeds contract");
        }
        if self.op == MeshOp::ModelQuery && (self.total_steps != 1 || self.step != 0) {
            return Err("whole model queries have exactly one service step");
        }
        match self.op {
            MeshOp::StageForward | MeshOp::MoeDispatch => {
                if self.layer_start >= self.layer_end || self.hidden_dim == 0 || self.seq_len == 0 {
                    return Err("invalid GG02 tensor shape or layer range");
                }
                let element_bytes = match self.dtype {
                    MeshDType::I32 => 4,
                    MeshDType::F16 => 2,
                    MeshDType::Opaque => return Err("tensor dtype required"),
                };
                let expected = usize::from(self.hidden_dim).checked_mul(usize::from(self.seq_len))
                    .and_then(|n| n.checked_mul(element_bytes)).ok_or("GG02 tensor size overflow")?;
                if expected != self.payload.len() {
                    return Err("GG02 tensor byte count disagrees with dtype and shape");
                }
            }
            MeshOp::AgentHandoff | MeshOp::VerifyChallenge | MeshOp::ModelQuery => {
                if self.dtype != MeshDType::Opaque || self.hidden_dim != 0 || self.seq_len != 0
                    || self.layer_start != 0 || self.layer_end != 0 {
                    return Err("opaque state is not a tensor layer");
                }
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, &'static str> {
        self.validate()?;
        if self.payload.len() > MAX_PAYLOAD_BYTES {
            return Err("GG02 payload exceeds limit");
        }
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(GG02_MAGIC);
        bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
        bytes[6..8].copy_from_slice(&(self.op as u16).to_le_bytes());
        bytes[8..24].copy_from_slice(&self.task_id.to_le_bytes());
        for (offset, value) in [
            (24, self.step), (26, self.total_steps), (28, self.layer_start),
            (30, self.layer_end), (32, self.hidden_dim), (34, self.seq_len),
            (52, self.dtype as u16), (54, self.flags),
        ] {
            bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }
        for (offset, value) in [
            (36, self.payload.len() as u32), (40, self.sender_hash), (44, self.target_hash),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[56..64].copy_from_slice(&self.session_tag.to_le_bytes());
        bytes.extend_from_slice(&self.payload);
        let crc = wire_checksum(&bytes);
        bytes[48..52].copy_from_slice(&crc.to_le_bytes());
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_LEN || &bytes[0..4] != GG02_MAGIC {
            return Err("invalid GG02 header");
        }
        if u16_at(bytes, 4) != 2 {
            return Err("unsupported GG02 version");
        }
        let payload_len = u32_at(bytes, 36) as usize;
        if payload_len > MAX_PAYLOAD_BYTES || bytes.len() != HEADER_LEN + payload_len {
            return Err("invalid GG02 payload length");
        }
        if u32_at(bytes, 48) != wire_checksum(bytes) {
            return Err("GG02 CRC32 mismatch");
        }
        let op = match u16_at(bytes, 6) {
            1 => MeshOp::StageForward,
            2 => MeshOp::MoeDispatch,
            3 => MeshOp::AgentHandoff,
            4 => MeshOp::VerifyChallenge,
            5 => MeshOp::ModelQuery,
            _ => return Err("unsupported GG02 opcode"),
        };
        let dtype = match u16_at(bytes, 52) {
            1 => MeshDType::I32,
            2 => MeshDType::F16,
            3 => MeshDType::Opaque,
            _ => return Err("unsupported GG02 dtype"),
        };
        let message = Self {
            task_id: u128::from_le_bytes(bytes[8..24].try_into().unwrap()),
            op,
            step: u16_at(bytes, 24),
            total_steps: u16_at(bytes, 26),
            layer_start: u16_at(bytes, 28),
            layer_end: u16_at(bytes, 30),
            hidden_dim: u16_at(bytes, 32),
            seq_len: u16_at(bytes, 34),
            sender_hash: u32_at(bytes, 40),
            target_hash: u32_at(bytes, 44),
            dtype,
            flags: u16_at(bytes, 54),
            session_tag: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
            payload: bytes[HEADER_LEN..].to_vec(),
        };
        message.validate()?;
        Ok(message)
    }
}
