use gramgrid_proto::MatmulTask;

#[derive(Debug)]
pub struct RowShard { pub start_row: usize, pub task: MatmulTask }

pub fn split_rows(task: &MatmulTask, ids: &[u128]) -> Result<Vec<RowShard>, &'static str> {
    gramgrid_proto::validate_matmul(task)?;
    if ids.is_empty() || ids.iter().any(|id| *id == 0) {
        return Err("нужны непустые идентификаторы частей");
    }
    let count = ids.len().min(task.m as usize);
    let unique: std::collections::BTreeSet<u128> = ids[..count].iter().copied().collect();
    if unique.len() != count { return Err("идентификаторы частей повторяются"); }
    let base = task.m as usize / count;
    let remainder = task.m as usize % count;
    let mut offset = 0usize;
    let mut shards = Vec::with_capacity(count);
    for (index, id) in ids.iter().take(count).enumerate() {
        let rows = base + usize::from(index < remainder);
        let start = offset * task.k as usize;
        let end = (offset + rows) * task.k as usize;
        shards.push(RowShard { start_row: offset,
            task: MatmulTask { id: *id, attempt: task.attempt, m: rows as u32,
                k: task.k, n: task.n, a: task.a[start..end].to_vec(), b: task.b.clone() } });
        offset += rows;
    }
    Ok(shards)
}
