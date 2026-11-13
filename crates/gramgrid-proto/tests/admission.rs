#[path="../src/admission.rs"]
mod admission;
use admission::{check,ResourceSample,ResourceLimits};

#[test]
fn configured_temperature_limit_requires_a_cool_measured_host() {
    let limits=ResourceLimits{minimum_memory_bytes:0,maximum_host_temperature_millic:Some(90000)};
    assert!(check(true,ResourceSample::default(),limits).is_err());
    assert!(check(true,ResourceSample{memory_available_bytes:None,host_temperature_millic:Some(90000)},limits).is_err());
    assert!(check(true,ResourceSample{memory_available_bytes:None,host_temperature_millic:Some(80000)},limits).is_ok());
}


#[test]
fn unknown_or_insufficient_memory_is_not_admitted() {
    let limits=ResourceLimits{minimum_memory_bytes:100,maximum_host_temperature_millic:None};
    assert!(check(true,ResourceSample::default(),limits).is_err());
    assert!(check(true,ResourceSample{memory_available_bytes:Some(99),host_temperature_millic:None},limits).is_err());
}


#[test]
fn a_node_cannot_enrol_without_declared_owner_consent() {
    assert!(check(false,ResourceSample::default(),ResourceLimits::default()).is_err());
}
