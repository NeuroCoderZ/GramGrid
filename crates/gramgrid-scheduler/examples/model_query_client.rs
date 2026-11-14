//! Отдельный локальный клиент GG02; вычисление выполняет только NPU-сервер.
use gramgrid_proto::mesh::{MeshDType, MeshEnvelope, MeshOp, RESULT_FLAG};
use serde_json::{json, Value};
use std::{collections::BTreeMap, net::{SocketAddr, TcpStream}, time::Duration};

type Error = String;

fn validate_budget(prompt: &str, tokens: u64) -> Result<(), Error> {
    if !(1..=128).contains(&tokens) { return Err("client output budget refused".into()); }
    if prompt.is_empty() || prompt.len() > 4096 { return Err("client input budget refused".into()); }
    Ok(())
}

fn validate_receipt(payload: &Value) -> Result<(), Error> {
    if payload["recipe"] != "flm" { return Err("non-NPU model receipt refused".into()); }
    if payload["checkpoint"] != "qwen3.5:0.8b" { return Err("foreign model checkpoint refused".into()); }
    if payload["model"] != "user.GramGrid-Qwen35-08B-NPU" { return Err("foreign model alias refused".into()); }
    if payload["npu_recipe_verified"] != true { return Err("unverified NPU recipe refused".into()); }
    if payload["text"].as_str().is_none_or(|s| s.trim().is_empty()) {
        return Err("model server returned no text".into());
    }
    Ok(())
}

fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 12 { return Err("required: --address --session --revision --task --max-tokens --prompt".into()); }
    let mut values = BTreeMap::new();
    for pair in args.chunks_exact(2) {
        if !["--address", "--session", "--revision", "--task", "--max-tokens", "--prompt"].contains(&pair[0].as_str())
            || values.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err("unknown or duplicated client argument".into());
        }
    }
    let address: SocketAddr = values.get("--address").ok_or("address missing")?
        .parse().map_err(|_| "literal socket address required")?;
    if !address.ip().is_loopback() { return Err("non-loopback model address refused".into()); }
    let session: u64 = values.get("--session").ok_or("session missing")?
        .parse().map_err(|_| "integer session required")?;
    let task: u128 = values.get("--task").ok_or("task missing")?
        .parse().map_err(|_| "integer task required")?;
    let tokens: u64 = values.get("--max-tokens").ok_or("output budget missing")?
        .parse().map_err(|_| "integer output budget required")?;
    let revision = *values.get("--revision").ok_or("revision missing")?;
    let prompt = *values.get("--prompt").ok_or("prompt missing")?;
    validate_budget(prompt, tokens)?;
    let request = MeshEnvelope { task_id:task, session_tag:session, op:MeshOp::ModelQuery,
        step:0, total_steps:1, layer_start:0, layer_end:0, hidden_dim:0, seq_len:0,
        dtype:MeshDType::Opaque, sender_hash:1, target_hash:2, flags:0,
        payload:serde_json::to_vec(&json!({"prompt":prompt,"max_tokens":tokens,"model_revision":revision})).map_err(|e| e.to_string())? };
    request.validate().map_err(str::to_string)?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(120))).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
    request.write_to(&mut stream)?;
    let reply = MeshEnvelope::read_from(&mut stream)?;
    if reply.op != MeshOp::ModelQuery || reply.flags != RESULT_FLAG
        || reply.task_id != request.task_id || reply.session_tag != request.session_tag
        || reply.sender_hash != request.target_hash || reply.target_hash != request.sender_hash {
        return Err("unbound model reply".into());
    }
    let payload: Value = serde_json::from_slice(&reply.payload).map_err(|e| e.to_string())?;
    if let Some(error) = payload["error"].as_str() { return Err(error.to_string()); }
    validate_receipt(&payload)?;
    println!("{payload}");
    Ok(())
}

fn main() {
    if let Err(error) = run() { eprintln!("model client refused: {error}"); std::process::exit(1); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_oversized_utf8_prompt_is_refused_before_connection() {
        assert!(validate_budget("", 32).is_err());
        assert!(validate_budget(&"я".repeat(2049), 32).is_err());
    }

    #[test]
    fn excessive_output_budget_is_refused_before_connection() {
        assert!(validate_budget("проверочная фикстура", 129).is_err());
    }

    #[test]
    fn unverified_recipe_flag_is_refused_by_the_client() {
        let fixture = json!({"text":"проверочная фикстура","recipe":"flm",
            "checkpoint":"qwen3.5:0.8b","model":"user.GramGrid-Qwen35-08B-NPU",
            "npu_recipe_verified":false});
        assert!(validate_receipt(&fixture).is_err());
    }

    #[test]
    fn another_model_alias_is_refused_by_the_client() {
        let fixture = json!({"text":"проверочная фикстура","recipe":"flm",
            "checkpoint":"qwen3.5:0.8b","model":"other-model","npu_recipe_verified":true});
        assert!(validate_receipt(&fixture).is_err());
    }

    #[test]
    fn another_checkpoint_is_refused_by_the_client() {
        let fixture = json!({"text":"проверочная фикстура","recipe":"flm",
            "checkpoint":"other-checkpoint","model":"user.GramGrid-Qwen35-08B-NPU",
            "npu_recipe_verified":true});
        assert!(validate_receipt(&fixture).is_err());
    }

    #[test]
    fn non_npu_recipe_is_refused_by_the_client() {
        let fixture = json!({"text":"проверочная фикстура, не ответ модели","recipe":"llamacpp"});
        assert!(validate_receipt(&fixture).is_err());
    }
}

