#[derive(Clone,Copy,Default)]
pub struct ResourceSample {
    pub memory_available_bytes:Option<u64>,
    pub host_temperature_millic:Option<i64>,
}
#[derive(Clone,Copy,Default)]
pub struct ResourceLimits {
    pub minimum_memory_bytes:u64,
    pub maximum_host_temperature_millic:Option<i64>,
}
pub fn check(consent:bool,sample:ResourceSample,limits:ResourceLimits)->Result<(),&'static str> {
    if !consent{return Err("нет явного согласия владельца");}
    if limits.minimum_memory_bytes>0 && sample.memory_available_bytes.is_none_or(|n|n<limits.minimum_memory_bytes){return Err("свободная память неизвестна или недостаточна");}
    if let Some(max)=limits.maximum_host_temperature_millic {
        if sample.host_temperature_millic.is_none_or(|t|t>=max){return Err("температура неизвестна или достигла настроенного предела");}
    }
    Ok(())
}
