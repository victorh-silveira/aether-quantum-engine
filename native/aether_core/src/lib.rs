use pyo3::prelude::*;

#[pyclass]
pub struct RustRingBuffer {
    capacity: usize,
    mask: usize,
    sequence: usize,
    epochs: Vec<i64>,
    prices: Vec<f64>,
}

#[pymethods]
impl RustRingBuffer {
    #[new]
    pub fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 || (capacity & (capacity - 1)) != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Capacity must be a positive power of two",
            ));
        }
        Ok(RustRingBuffer {
            capacity,
            mask: capacity - 1,
            sequence: 0,
            epochs: vec![0; capacity],
            prices: vec![0.0; capacity],
        })
    }

    pub fn push_tick(&mut self, epoch_ms: i64, price: f64) -> usize {
        let slot = self.sequence & self.mask;
        self.epochs[slot] = epoch_ms;
        self.prices[slot] = price;
        let seq = self.sequence;
        self.sequence += 1;
        seq
    }

    pub fn count(&self) -> usize {
        std::cmp::min(self.sequence, self.capacity)
    }

    pub fn latest_tick(&self) -> Option<(i64, f64)> {
        if self.sequence == 0 {
            None
        } else {
            let slot = (self.sequence - 1) & self.mask;
            Some((self.epochs[slot], self.prices[slot]))
        }
    }

    pub fn clear(&mut self) {
        self.sequence = 0;
        self.epochs.fill(0);
        self.prices.fill(0.0);
    }
}

#[pyfunction]
pub fn compute_realized_volatility_rust(prices: Vec<f64>) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for i in 1..prices.len() {
        let p0 = prices[i - 1];
        let p1 = prices[i];
        if p0 > 0.0 && p1 > 0.0 {
            let log_ret = (p1 / p0).ln();
            sum_sq += log_ret * log_ret;
        }
    }
    sum_sq.sqrt()
}

#[pymodule]
fn aether_core_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustRingBuffer>()?;
    m.add_function(wrap_pyfunction!(compute_realized_volatility_rust, m)?)?;
    Ok(())
}
