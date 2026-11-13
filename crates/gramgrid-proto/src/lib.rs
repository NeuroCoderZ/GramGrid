pub mod admission;
pub mod mesh;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulTask {
    pub id: u128,
    pub attempt: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub a: Vec<i8>,
    pub b: Vec<i8>,
}

pub const HEADER_SIZE: usize = 64;
pub const MAX_DIM: u32 = 1024;
pub const MAX_PACKET: usize = 8 * 1024 * 1024;

pub fn validate_matmul(task: &MatmulTask) -> Result<(), &'static str> {
    if task.id == 0 || task.m == 0 || task.k == 0 || task.n == 0
        || task.m > MAX_DIM || task.k > MAX_DIM || task.n > MAX_DIM {
        return Err("идентификатор или размеры вне допустимого диапазона");
    }
    if task.a.len() != task.m as usize * task.k as usize
        || task.b.len() != task.k as usize * task.n as usize {
        return Err("размер буфера не соответствует форме матрицы");
    }
    Ok(())
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

pub fn encode_matmul(task: &MatmulTask) -> Result<Vec<u8>, &'static str> {
    validate_matmul(task)?;
    let mut bytes = vec![0u8; HEADER_SIZE];
    bytes[..4].copy_from_slice(b"GG01");
    bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
    bytes[6..8].copy_from_slice(&1u16.to_le_bytes());
    bytes[8..24].copy_from_slice(&task.id.to_le_bytes());
    put_u32(&mut bytes, 24, task.attempt);
    put_u32(&mut bytes, 28, 1);
    put_u32(&mut bytes, 32, task.m);
    put_u32(&mut bytes, 36, task.k);
    put_u32(&mut bytes, 40, task.n);
    put_u32(&mut bytes, 44, task.a.len() as u32);
    put_u32(&mut bytes, 48, task.b.len() as u32);
    put_u32(&mut bytes, 52, task.m.saturating_mul(task.n));
    bytes.extend(task.a.iter().map(|x| *x as u8));
    bytes.extend(task.b.iter().map(|x| *x as u8));
    Ok(bytes)
}

pub fn decode_matmul(bytes: &[u8]) -> Result<MatmulTask, &'static str> {
    if bytes.len() < HEADER_SIZE || bytes.len() > MAX_PACKET || &bytes[..4] != b"GG01"
        || (bytes[4..6] != 1u16.to_le_bytes() && bytes[4..6] != 2u16.to_le_bytes())
        || bytes[6..8] != 1u16.to_le_bytes()
        || u32_at(bytes, 28) != 1 {
        return Err("неверный заголовок пакета");
    }
    let la = u32_at(bytes, 44) as usize;
    let lb = u32_at(bytes, 48) as usize;
    let end_a = HEADER_SIZE.checked_add(la).ok_or("переполнение длины")?;
    let end_b = end_a.checked_add(lb).ok_or("переполнение длины")?;
    let a = bytes.get(HEADER_SIZE..end_a).ok_or("обрезанный вход A")?;
    let b = bytes.get(end_a..end_b).ok_or("обрезанный вход B")?;
    if end_b != bytes.len() {
        return Err("неверная длина пакета");
    }
    let task = MatmulTask {
        id: u128::from_le_bytes(bytes[8..24].try_into().unwrap()),
        attempt: u32_at(bytes, 24),
        m: u32_at(bytes, 32), k: u32_at(bytes, 36), n: u32_at(bytes, 40),
        a: a.iter().map(|x| *x as i8).collect(),
        b: b.iter().map(|x| *x as i8).collect(),
    };
    validate_matmul(&task)?;
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    let variant = u32_at(bytes, 56);
    let parameters_valid = if version == 1 { bytes[56..64] == [0; 8] }
        else { matches!(variant, 0 | 4 | 8 | 16) && bytes[60..64] == [0; 4] };
    if u32_at(bytes, 52) != task.m * task.n || !parameters_valid {
        return Err("неверный размер выхода или параметр ядра");
    }
    Ok(task)
}

pub fn verify_matmul_with_vector(
    task: &MatmulTask, result: &[i32], vector: &[u8],
) -> Result<bool, &'static str> {
    validate_matmul(task)?;
    let (m, k, n) = (task.m as usize, task.k as usize, task.n as usize);
    if result.len() != m * n || vector.len() != n || vector.iter().any(|x| *x > 1) {
        return Err("неверный размер результата или проверочного вектора");
    }
    let mut br = vec![0i64; k];
    for (i, value) in br.iter_mut().enumerate() {
        for (j, bit) in vector.iter().enumerate() {
            *value += task.b[i * n + j] as i64 * *bit as i64;
        }
    }
    for row in 0..m {
        let mut lhs = 0i64;
        let mut rhs = 0i64;
        for (col, value) in br.iter().enumerate() {
            lhs += task.a[row * k + col] as i64 * value;
        }
        for (col, bit) in vector.iter().enumerate() {
            rhs += result[row * n + col] as i64 * *bit as i64;
        }
        if lhs != rhs { return Ok(false); }
    }
    Ok(true)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddTask {
    pub id: u128,
    pub attempt: u32,
    pub a: Vec<i32>,
    pub b: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeResult {
    pub id: u128,
    pub attempt: u32,
    pub op: u32,
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub elapsed_ns: u64,
    pub values: Vec<i32>,
}

pub const MAX_ADD_ELEMENTS: usize = 65536;

fn check_frame(bytes: &[u8], kind: u16, op: u32) -> Result<(), &'static str> {
    if bytes.len() < HEADER_SIZE || bytes.len() > MAX_PACKET || &bytes[..4] != b"GG01"
        || (bytes[4..6] != 1u16.to_le_bytes() && bytes[4..6] != 2u16.to_le_bytes()) || bytes[6..8] != kind.to_le_bytes()
        || u32_at(bytes, 28) != op || u128::from_le_bytes(bytes[8..24].try_into().unwrap()) == 0 {
        return Err("неверный заголовок пакета");
    }
    Ok(())
}

fn header(kind: u16, op: u32, id: u128, attempt: u32, m: u32, k: u32, n: u32) -> Vec<u8> {
    let mut bytes = vec![0; HEADER_SIZE];
    bytes[..4].copy_from_slice(b"GG01");
    bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
    bytes[6..8].copy_from_slice(&kind.to_le_bytes());
    bytes[8..24].copy_from_slice(&id.to_le_bytes());
    put_u32(&mut bytes, 24, attempt);
    put_u32(&mut bytes, 28, op);
    put_u32(&mut bytes, 32, m);
    put_u32(&mut bytes, 36, k);
    put_u32(&mut bytes, 40, n);
    put_u32(&mut bytes, 52, m * n);
    bytes
}

pub fn validate_add(task: &AddTask) -> Result<(), &'static str> {
    if task.id == 0 || task.a.is_empty() || task.a.len() > MAX_ADD_ELEMENTS
        || task.a.len() != task.b.len() {
        return Err("неверный размер векторного задания");
    }
    Ok(())
}

fn read_i32s(bytes: &[u8]) -> Vec<i32> {
    bytes.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect()
}

pub fn encode_add(task: &AddTask) -> Result<Vec<u8>, &'static str> {
    validate_add(task)?;
    let mut bytes = header(1, 2, task.id, task.attempt, task.a.len() as u32, 1, 1);
    put_u32(&mut bytes, 44, (task.a.len() * 4) as u32);
    put_u32(&mut bytes, 48, (task.b.len() * 4) as u32);
    for value in task.a.iter().chain(&task.b) { bytes.extend(value.to_le_bytes()); }
    Ok(bytes)
}

pub fn decode_add(bytes: &[u8]) -> Result<AddTask, &'static str> {
    check_frame(bytes, 1, 2)?;
    let len = u32_at(bytes, 32) as usize;
    if len == 0 || len > MAX_ADD_ELEMENTS || u32_at(bytes, 36) != 1 || u32_at(bytes, 40) != 1
        || u32_at(bytes, 44) as usize != len * 4 || u32_at(bytes, 48) as usize != len * 4
        || u32_at(bytes, 52) as usize != len || bytes.len() != HEADER_SIZE + len * 8
        || bytes[56..64] != [0; 8] {
        return Err("неверная форма векторного пакета");
    }
    Ok(AddTask { id: u128::from_le_bytes(bytes[8..24].try_into().unwrap()),
        attempt: u32_at(bytes, 24),
        a: read_i32s(&bytes[HEADER_SIZE..HEADER_SIZE + len * 4]),
        b: read_i32s(&bytes[HEADER_SIZE + len * 4..]) })
}

fn check_result_shape(result: &ComputeResult) -> Result<(), &'static str> {
    if result.id == 0 { return Err("пустой идентификатор результата"); }
    match result.op {
        1 if result.m > 0 && result.k > 0 && result.n > 0
            && result.m <= MAX_DIM && result.k <= MAX_DIM && result.n <= MAX_DIM => {},
        2 if result.m > 0 && result.m as usize <= MAX_ADD_ELEMENTS && result.k == 1 && result.n == 1 => {},
        _ => return Err("неизвестная операция или неверная форма результата"),
    }
    if result.values.len() != result.m as usize * result.n as usize {
        return Err("неверное количество элементов результата");
    }
    Ok(())
}

pub fn encode_result(result: &ComputeResult) -> Result<Vec<u8>, &'static str> {
    check_result_shape(result)?;
    let mut bytes = header(2, result.op, result.id, result.attempt, result.m, result.k, result.n);
    bytes[56..64].copy_from_slice(&result.elapsed_ns.to_le_bytes());
    for value in &result.values { bytes.extend(value.to_le_bytes()); }
    Ok(bytes)
}

pub fn decode_result(bytes: &[u8]) -> Result<ComputeResult, &'static str> {
    if bytes.len() < HEADER_SIZE { return Err("обрезанный результат"); }
    let op = u32_at(bytes, 28);
    check_frame(bytes, 2, op)?;
    let count = u32_at(bytes, 52) as usize;
    if !matches!(op, 1 | 2) || count > MAX_DIM as usize * MAX_DIM as usize
        || u32_at(bytes, 44) != 0 || u32_at(bytes, 48) != 0
        || bytes.len() != HEADER_SIZE + count * 4 {
        return Err("неверная длина результата");
    }
    let result = ComputeResult { id: u128::from_le_bytes(bytes[8..24].try_into().unwrap()),
        attempt: u32_at(bytes, 24), op, m: u32_at(bytes, 32), k: u32_at(bytes, 36), n: u32_at(bytes, 40),
        elapsed_ns: u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
        values: read_i32s(&bytes[HEADER_SIZE..]) };
    check_result_shape(&result)?;
    Ok(result)
}

pub fn verify_add(task: &AddTask, result: &[i32]) -> Result<bool, &'static str> {
    validate_add(task)?;
    if result.len() != task.a.len() { return Err("неверная длина результата сложения"); }
    Ok(task.a.iter().zip(&task.b).zip(result).all(|((a, b), c)| a.wrapping_add(*b) == *c))
}

pub const VERIFICATION_ROUNDS: usize = 40;

pub fn verify_matmul(
    task: &MatmulTask, result: &[i32], random_bytes: &[u8],
) -> Result<bool, &'static str> {
    validate_matmul(task)?;
    let n = task.n as usize;
    if random_bytes.len() != VERIFICATION_ROUNDS * n {
        return Err("требуется ровно сорок независимых проверочных векторов");
    }
    for round in random_bytes.chunks_exact(n) {
        let bits: Vec<u8> = round.iter().map(|x| x & 1).collect();
        if !verify_matmul_with_vector(task, result, &bits)? { return Ok(false); }
    }
    Ok(true)
}


