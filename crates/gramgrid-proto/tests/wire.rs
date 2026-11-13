use gramgrid_proto::{MatmulTask, decode_matmul, encode_matmul};

#[test]
fn signed_matrix_roundtrip_has_stable_header() {
    let task = MatmulTask {
        id: 0x112233445566778899aabbccddeeff01,
        attempt: 2,
        m: 2,
        k: 3,
        n: 2,
        a: vec![-128, -1, 0, 1, 2, 127],
        b: vec![7, -8, 9, -10, 11, -12],
    };
    let bytes = encode_matmul(&task).expect("корректное задание должно кодироваться");
    assert_eq!(&bytes[0..4], b"GG01");
    assert_eq!(bytes.len(), 64 + 12);
    assert_eq!(decode_matmul(&bytes).unwrap(), task);
}
