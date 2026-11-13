use gramgrid_proto::{MatmulTask, encode_matmul, decode_matmul};

#[test]
fn version_two_carries_checked_kernel_parameters() {
    let task=MatmulTask{id:31,attempt:0,m:1,k:2,n:1,a:vec![1,-2],b:vec![3,4]};
    let mut packet=encode_matmul(&task).unwrap();
    packet[4..6].copy_from_slice(&2u16.to_le_bytes());
    packet[56..60].copy_from_slice(&8u32.to_le_bytes());
    assert_eq!(decode_matmul(&packet).unwrap(),task);
    packet[56..60].copy_from_slice(&99u32.to_le_bytes());
    assert!(decode_matmul(&packet).is_err());
}
