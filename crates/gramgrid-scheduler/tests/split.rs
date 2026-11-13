#[path = "../src/split.rs"]
mod split;
use gramgrid_proto::MatmulTask;

#[test]
fn uneven_rows_are_partitioned_once_without_gaps() {
    let task = MatmulTask { id: 21, attempt: 0, m: 5, k: 2, n: 1,
        a: (0..10).collect(), b: vec![3, -4] };
    let shards = split::split_rows(&task, &[101, 102]).unwrap();
    assert_eq!(shards.len(), 2);
    assert_eq!(shards[0].start_row, 0);
    assert_eq!(shards[1].start_row, shards[0].task.m as usize);
    assert_eq!(shards.iter().map(|s| s.task.m).sum::<u32>(), task.m);
    assert_eq!(shards.iter().flat_map(|s| s.task.a.clone()).collect::<Vec<_>>(), task.a);
    assert!(shards.iter().all(|s| s.task.b == task.b));
    assert!(split::split_rows(&task, &[]).is_err());
}
