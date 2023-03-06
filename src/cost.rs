use ndarray::Array2;

/// Mean Root Squared Error
pub fn mrse(ideal: &Array2<f32>, actual: &Array2<f32>) -> f32 {
    let vector = (ideal - actual).mapv(|x| 0.5 * x * x);

    vector.sum()
}
