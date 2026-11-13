use gramgrid_proto::{MatmulTask, decode_matmul, encode_matmul};

fn task() -> MatmulTask {
    MatmulTask { id: 7, attempt: 0, m: 1, k: 2, n: 1, a: vec![1, 2], b: vec![3, 4] }
}

#[test]
fn input_length_must_match_shape() {
    let mut bad = task();
    bad.a.pop();
    assert!(encode_matmul(&bad).is_err(), "несогласованный буфер должен отклоняться");
}

#[test]
fn oversized_dimensions_are_rejected() {
    let mut bad = task();
    bad.m = 1025;
    bad.a = vec![1; 2050];
    assert!(encode_matmul(&bad).is_err(), "размер сверх предела должен отклоняться");
}

#[test]
fn trailing_bytes_are_not_silently_accepted() {
    let mut bytes = encode_matmul(&task()).unwrap();
    bytes.push(1);
    assert!(decode_matmul(&bytes).is_err(), "посторонний хвост должен отклоняться");
}
