#[path = "../src/fair.rs"]
mod fair;
use fair::{FairQueue, QueueClass};

#[test]
fn both_busy_queues_receive_equal_cost_and_idle_share_is_reused() {
    let mut queue = FairQueue::default();
    for i in 0..100 {
        queue.push(QueueClass::Evolution, 100, i);
        queue.push(QueueClass::Rental, 100, i);
    }
    let mut costs = [0u64; 2];
    for _ in 0..200 {
        let item = queue.pop_where(|_| true).expect("непустая очередь должна выдавать работу");
        costs[item.class.index()] += item.cost;
    }
    assert_eq!(costs[0], costs[1]);
    assert_eq!(queue.lengths(), [0, 0]);
    for i in 0..5 { queue.push(QueueClass::Rental, 10, i); }
    for _ in 0..5 { assert_eq!(queue.pop_where(|_| true).unwrap().class, QueueClass::Rental); }
    assert!(queue.pop_where(|_| true).is_none());
}

#[test]
fn blocked_head_does_not_block_a_compatible_task() {
    let mut queue = FairQueue::default();
    queue.push(QueueClass::Evolution, 1, 1);
    queue.push(QueueClass::Rental, 1, 2);
    let found = queue.pop_where(|x| *x == 2).unwrap();
    assert_eq!(found.value, 2);
    assert_eq!(queue.lengths(), [1, 0]);
    queue.retain(|_| false);
    assert_eq!(queue.lengths(), [0, 0]);
}
