pub struct Matrix {
    pub cols: usize,
    pub rows: usize,
    pub data: Vec<Vec<f32>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(rows);

        for _ in 0..rows {
            let mut row: Vec<f32> = Vec::with_capacity(cols);

            for _ in 0..cols {
                row.push(0f32)
            }

            data.push(row);
        }

        Matrix { rows, cols, data }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    use super::Matrix;

    #[test]
    fn test_new_matrix() {
        let matrix = Matrix::new(2, 2);
        assert_eq!(matrix.data, vec![vec![0f32, 0f32], vec![0f32, 0f32]]);
    }
}
