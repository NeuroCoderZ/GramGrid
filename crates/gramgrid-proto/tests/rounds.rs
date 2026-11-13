use gramgrid_proto::{MatmulTask, VERIFICATION_ROUNDS, verify_matmul};

#[test]
fn independent_rounds_accept_correct_result_and_reject_forgery() {
    let task = MatmulTask { id: 11, attempt: 0, m: 2, k: 2, n: 2,
        a: vec![1, 0, 0, 1], b: vec![-128, 127, 5, -6] };
    let expected: Vec<i32> = task.b.iter().map(|x| *x as i32).collect();
    let mut bits = vec![1; VERIFICATION_ROUNDS * task.n as usize];
    bits[1] = 0;
    assert!(verify_matmul(&task, &expected, &bits).unwrap());
    let mut forged = expected;
    forged[0] += 7;
    forged[1] -= 7;
    assert!(!verify_matmul(&task, &forged, &bits).unwrap());
    assert!(verify_matmul(&task, &forged, &[]).is_err());
}

#[test]
fn maximum_contraction_dimension_remains_integer_exact() {
    let task = MatmulTask { id: 12, attempt: 0, m: 1, k: 1024, n: 1,
        a: vec![-128; 1024], b: vec![-128; 1024] };
    let expected: i32 = task.a.iter().zip(&task.b).map(|(a, b)| *a as i32 * *b as i32).sum();
    assert!(verify_matmul(&task, &[expected], &vec![1; VERIFICATION_ROUNDS]).unwrap());
}
