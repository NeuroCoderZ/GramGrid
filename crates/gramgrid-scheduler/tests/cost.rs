#[path = "../src/cost.rs"]
mod cost;
use cost::CostModel;

#[test]
fn costs_follow_observed_time_not_incomparable_operation_counts() {
    let mut model=CostModel::default();
    model.observe(1,1000,1_000_000);
    model.observe(2,1000,10_000_000);
    assert_eq!(model.estimate(2,1000),model.estimate(1,1000)*10);
    assert_eq!(model.samples().len(),2);
}
