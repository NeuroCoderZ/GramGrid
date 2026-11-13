use gramgrid_proto::{AddTask, ComputeResult, decode_add, decode_result, encode_add, encode_result};

#[test]
fn add_packet_preserves_all_signed_bits() {
    let task = AddTask { id: 17, attempt: 3,
        a: vec![i32::MIN, i32::MAX, 0, -1], b: vec![1, -1, 7, 9] };
    let bytes = encode_add(&task).unwrap();
    let recovered = decode_add(&bytes).unwrap();
    assert_eq!(recovered.a, task.a);
    assert_eq!(recovered.b, task.b);
    assert_eq!(recovered.id, task.id);
    assert_eq!(recovered.attempt, task.attempt);
}

#[test]
fn result_packet_is_integer_exact() {
    let result = ComputeResult { id: 17, attempt: 3, op: 2, m: 4, k: 1, n: 1,
        elapsed_ns: 12345, values: vec![i32::MIN, i32::MAX, 7, -9] };
    assert_eq!(decode_result(&encode_result(&result).unwrap()).unwrap(), result);
}
