use gramgrid_proto::{AddTask, verify_add};

#[test]
fn addition_uses_explicit_twos_complement_wrapping() {
    let task = AddTask { id: 18, attempt: 0,
        a: vec![i32::MAX, i32::MIN, -7, 0], b: vec![1, -1, 9, 0] };
    let result: Vec<i32> = task.a.iter().zip(&task.b).map(|(a, b)| a.wrapping_add(*b)).collect();
    assert!(verify_add(&task, &result).unwrap());
    let mut wrong = result;
    wrong[0] = 0;
    assert!(!verify_add(&task, &wrong).unwrap());
}
