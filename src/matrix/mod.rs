use std::ops::{Add, Mul, Sub};

#[derive(PartialEq, Debug, Clone)]
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

    pub fn from_vec(vec: &Vec<Vec<f32>>) -> Matrix {
        let rows = vec.len();
        let cols = vec[0].len();

        let mut matrix = Matrix::new(rows, cols);

        matrix.data = vec.to_vec();

        matrix
    }

    pub fn transpose(&self) -> Matrix {
        let rows = self.cols;
        let cols = self.rows;
        let mut trans = Matrix::new(rows, cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                trans.data[j][i] = self.data[i][j];
            }
        }

        trans
    }

    pub fn map<F>(&self, func: F) -> Matrix
    where
        F: Fn(f32) -> f32,
    {
        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = func(self.data[i][j]);
            }
        }

        result
    }

    pub fn mul(&self, rhs: &Matrix) -> Matrix {
        let mut matrix = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix.data[i][j] = self.data[i][j] * rhs.data[i][j]
            }
        }

        matrix
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        let mut matrix = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }

        matrix
    }
}

impl Add<f32> for &Matrix {
    type Output = Matrix;

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

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        let mut result = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }

        result
    }
}

/// Multiply float values
impl Mul<f32> for &Matrix {
    type Output = Matrix;

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

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        let cols = rhs.cols;
        let rows = self.rows;
        let mut result = Matrix::new(rows, cols);

        for i in 0..self.rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..rhs.rows {
                    sum += self.data[i][k] * rhs.data[k][j];
                }

                result.data[i][j] = sum;
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
        assert_eq!(
            (&matrix + 1.0).data,
            vec![vec![1f32, 1f32], vec![1f32, 1f32]]
        );
    }

    #[test]
    fn test_multiply_f32() {
        let matrix = &Matrix::new(2, 2) + 1.5;

        assert_eq!(
            (&matrix * 2.0).data,
            vec![vec![3f32, 3f32], vec![3f32, 3f32]]
        );
    }

    #[test]
    fn test_transpose() {
        let matrix = Matrix::new(1, 2);

        assert_eq!((matrix.transpose()).data, vec![vec![0f32], vec![0f32]]);
    }

    #[test]
    fn test_transpose_complex() {
        let mut matrix = Matrix::new(2, 3);
        matrix.data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let expected = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];

        assert_eq!((matrix.transpose()).data, expected);
    }

    #[test]
    fn test_matrix_multiply_matrix() {
        let mut m1 = Matrix::new(1, 2);
        m1.data = vec![vec![1f32, 2f32]];

        let mut m2 = Matrix::new(2, 1);
        m2.data = vec![vec![3f32], vec![4f32]];

        let scenarios = vec![
            (
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![3.0], vec![4.0]],
                vec![vec![11.0], vec![25.0]],
            ),
            (
                vec![vec![1f32, 2f32]],
                vec![vec![3f32], vec![4f32]],
                vec![vec![11.0]],
            ),
        ];

        for (first, second, exp) in scenarios {
            let m1 = Matrix::from_vec(&first);
            let m2 = Matrix::from_vec(&second);

            assert_eq!((&m1 * &m2).data, exp);
        }
    }
}
