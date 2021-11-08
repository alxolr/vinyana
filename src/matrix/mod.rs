use std::ops::{Add, Mul};

#[derive(PartialEq)]
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

impl Add<f32> for Matrix {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + rhs;
            }
        }

        result
    }
}

/// Multiply float values
impl Mul<f32> for Matrix {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * rhs;
            }
        }

        result
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

    #[test]
    fn test_add_f32() {
        let matrix = Matrix::new(2, 2);
        assert_eq!((matrix + 1.0).data, vec![vec![1f32, 1f32], vec![1f32, 1f32]])
    }

    #[test]
    fn test_multiply_f32() {
        let matrix = Matrix::new(2, 2) + 1.5;
        
        assert_eq!((matrix * 2.0).data, vec![vec![3f32, 3f32], vec![3f32, 3f32]])
    }
}
