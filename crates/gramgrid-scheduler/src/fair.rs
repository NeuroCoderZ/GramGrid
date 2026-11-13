use std::collections::VecDeque;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum QueueClass { Evolution, Rental }

impl QueueClass { pub fn index(self) -> usize { match self { Self::Evolution => 0, Self::Rental => 1 } } }

pub struct Ticket<T> { pub class: QueueClass, pub cost: u64, pub value: T }

pub struct FairQueue<T> {
    queues: [VecDeque<Ticket<T>>; 2],
    served: [u128; 2],
    prefer: usize,
}

impl<T> Default for FairQueue<T> {
    fn default() -> Self { Self { queues: [VecDeque::new(), VecDeque::new()], served: [0, 0], prefer: 0 } }
}

impl<T> FairQueue<T> {
    pub fn push(&mut self, class: QueueClass, cost: u64, value: T) {
        self.queues[class.index()].push_back(Ticket { class, cost: cost.max(1), value });
    }
    pub fn pop_where(&mut self, eligible: impl Fn(&T) -> bool) -> Option<Ticket<T>> {
        let positions = [self.queues[0].iter().position(|x| eligible(&x.value)),
            self.queues[1].iter().position(|x| eligible(&x.value))];
        let index = match positions {
            [None, None] => return None,
            [Some(_), None] => 0,
            [None, Some(_)] => 1,
            [Some(_), Some(_)] => {
                if self.served[0] == self.served[1] { self.prefer }
                else if self.served[0] < self.served[1] { 0 } else { 1 }
            },
        };
        let ticket = self.queues[index].remove(positions[index].unwrap()).unwrap();
        if positions.iter().all(Option::is_some) {
            self.served[index] = self.served[index].saturating_add(ticket.cost as u128);
        } else {
            let total = self.served[0].max(self.served[1]).saturating_add(ticket.cost as u128);
            self.served = [total, total];
        }
        self.prefer = 1 - index;
        Some(ticket)
    }
    pub fn retain(&mut self, keep: impl Fn(&T) -> bool) {
        for queue in &mut self.queues { queue.retain(|ticket| keep(&ticket.value)); }
    }
    pub fn lengths(&self) -> [usize; 2] { [self.queues[0].len(), self.queues[1].len()] }
}
