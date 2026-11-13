use std::collections::BTreeMap;

#[derive(Default)]
pub struct CostModel { samples: BTreeMap<u32,(u64,u64)>, minimum: BTreeMap<u32,u64> }
impl CostModel {
    pub fn observe(&mut self, op:u32, ops:u64, observed_ns:u64) {
        if ops==0||observed_ns==0{return;}
        let sample=self.samples.entry(op).or_insert((0,0));
        sample.0=sample.0.saturating_add(ops);sample.1=sample.1.saturating_add(observed_ns);
        self.minimum.entry(op).and_modify(|x|*x=(*x).min(observed_ns)).or_insert(observed_ns);
    }
    pub fn estimate(&self, op:u32, ops:u64)->u64 {
        match self.samples.get(&op) {
            Some((work,ns))=>((ops as u128*(*ns as u128)/(*work as u128)).min(u64::MAX as u128) as u64)
                .max(self.minimum[&op]).max(1),
            None=>ops.max(1),
        }
    }
    pub fn samples(&self)->&BTreeMap<u32,(u64,u64)> { &self.samples }
}
