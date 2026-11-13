use gramgrid_proto::admission::{ResourceSample,ResourceLimits};
use serde_json::{json,Value};
use std::fs;

pub fn parse_memory(text:&str)->Option<u64> {
    let line=text.lines().find(|s|s.starts_with("MemAvailable:"))?;
    let mut fields=line.split_whitespace();
    if fields.next()!=Some("MemAvailable:"){return None;}
    let number=fields.next()?.parse::<u64>().ok()?;
    if fields.next()!=Some("kB"){return None;}
    number.checked_mul(1024)
}

pub fn sample()->ResourceSample {
    let memory_available_bytes=fs::read_to_string("/proc/meminfo").ok().and_then(|s|parse_memory(&s));
    let mut host_temperature_millic=None;
    if let Ok(entries)=fs::read_dir("/sys/class/hwmon") {
        for entry in entries.flatten() {
            if fs::read_to_string(entry.path().join("name")).ok().is_some_and(|x|x.trim()=="k10temp") {
                host_temperature_millic=fs::read_to_string(entry.path().join("temp1_input")).ok()
                    .and_then(|s|s.trim().parse::<i64>().ok()).filter(|t|(-50000..=150000).contains(t));
                break;
            }
        }
    }
    ResourceSample{memory_available_bytes,host_temperature_millic}
}
pub fn limits()->Result<ResourceLimits,String> {
    let minimum_memory_bytes=match std::env::var("GRAMGRID_MIN_MEMORY_BYTES") {
        Ok(s)=>s.parse().map_err(|_|"некорректный GRAMGRID_MIN_MEMORY_BYTES")?,Err(_)=>0};
    let maximum_host_temperature_millic=match std::env::var("GRAMGRID_MAX_HOST_TEMP_MILLIC") {
        Ok(s)=>Some(s.parse().map_err(|_|"некорректный GRAMGRID_MAX_HOST_TEMP_MILLIC")?),Err(_)=>None};
    if maximum_host_temperature_millic.is_some_and(|t|!(40000..=120000).contains(&t)){return Err("предел температуры вне разрешённого диапазона".into());}
    Ok(ResourceLimits{minimum_memory_bytes,maximum_host_temperature_millic})
}
pub fn as_json(sample:ResourceSample)->Value {json!({"memory_available_bytes":sample.memory_available_bytes,
    "host_temperature_millic":sample.host_temperature_millic,"temperature_source":"host_cpu_k10temp",
    "power_watts":null,"npu_utilization":null})}
pub fn limits_json(limits:ResourceLimits)->Value {json!({"minimum_memory_bytes":limits.minimum_memory_bytes,
    "maximum_host_temperature_millic":limits.maximum_host_temperature_millic})}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mem_available_is_parsed_without_inventing_an_unknown_value() {
        assert_eq!(parse_memory("MemTotal: 123456 kB\nMemAvailable: 100 kB\n"),Some(102400));
        assert_eq!(parse_memory("MemAvailable: bad kB"),None);
        assert_eq!(parse_memory("MemTotal: 100 kB"),None);
    }
}
