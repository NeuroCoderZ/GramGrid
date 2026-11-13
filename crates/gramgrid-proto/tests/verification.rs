use gramgrid_proto::{MatmulTask, verify_matmul_with_vector};

fn identity_task() -> MatmulTask {
    MatmulTask { id: 9, attempt: 0, m: 2, k: 2, n: 2,
        a: vec![1, 0, 0, 1], b: vec![-128, 127, 5, -6] }
}

#[test]
fn exact_signed_product_passes_freivalds_check() {
    let task = identity_task();
    let result: Vec<i32> = task.b.iter().map(|x| *x as i32).collect();
    assert!(verify_matmul_with_vector(&task, &result, &[1, 1]).unwrap());
    assert!(verify_matmul_with_vector(&task, &result, &[1, 0]).unwrap());
}

#[test]
fn public_challenge_allows_a_wrong_matrix_to_pass() {
    let task = identity_task();
    let mut forged: Vec<i32> = task.b.iter().map(|x| *x as i32).collect();
    forged[0] += 7;
    forged[1] -= 7;
    assert!(verify_matmul_with_vector(&task, &forged, &[1, 1]).unwrap(),
        "заранее известный вектор допускает согласованную подделку");
    assert!(!verify_matmul_with_vector(&task, &forged, &[1, 0]).unwrap(),
        "другой вектор выявляет подделку");
}
