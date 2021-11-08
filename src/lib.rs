use std::ops::Range;

use matrix::Matrix;
use rand::{prelude::ThreadRng, random, thread_rng, Rng};

mod matrix;
pub const E: f32 = 2.7182818284590451f32;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

/// First iteration of Neural Network with only 3 layers input -> hidden -> output
pub struct NeuralNetwork {
    pub input_nodes: usize,
    pub hidden_nodes: usize,
    pub output_nodes: usize,
    pub learning_rate: f32,
    ih_weights: Matrix,
    ho_weights: Matrix,
}

impl NeuralNetwork {
    /// The posibility to initialise a completly new neural network
    /// All the weights will be random from -1 to 1 floats
    pub fn new(inputs: usize, hidden: usize, outputs: usize, learning_rate: f32) -> NeuralNetwork {
        let rng = thread_rng();

        let mut ih_weights = Matrix::new(hidden, inputs);
        let mut ho_weights = Matrix::new(outputs, hidden);

        let randomize = |_| rng.clone().gen_range(-1.0..1.0);

        ih_weights.map(randomize);
        ho_weights.map(randomize);

        NeuralNetwork {
            input_nodes: inputs,
            hidden_nodes: hidden,
            output_nodes: outputs,
            learning_rate,
            ih_weights,
            ho_weights,
        }
    }

    pub fn train(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) {
        unimplemented!()
    }

    pub fn predict(&self, inputs: Vec<Vec<f32>>) -> Matrix {
        let inputs_m = Matrix::from_vec(inputs);

        let mut hidden_inputs = self.ih_weights.clone() * inputs_m;
        let hidden_outputs = hidden_inputs.map(sigmoid);

        let mut final_inputs = self.ho_weights.clone() * hidden_outputs;
        let final_outputs = final_inputs.map(sigmoid);

        final_outputs
    }
}

#[cfg(test)]
mod tests {
    use crate::NeuralNetwork;

    #[test]
    fn test_neural_network_initialisation() {
        let nn = NeuralNetwork::new(3, 3, 1, 0.01);

        assert_eq!(nn.hidden_nodes, 3);
        assert_eq!(nn.input_nodes, 3);
        assert_eq!(nn.output_nodes, 1);
    }

    #[test]
    fn test_predict_is_building() {
        let mut nn = NeuralNetwork::new(2, 2, 1, 0.01);

        let cor1 = nn.predict(vec![vec![0.0, 1.0]]);
        let cor2 = nn.predict(vec![vec![0.0, 0.0]]);
        let cor3 = nn.predict(vec![vec![1.0, 0.0]]);
        let cor4 = nn.predict(vec![vec![1.0, 1.0]]);

        println!(
            "predicted {:?} {:?} {:?} {:?}",
            cor1.data, cor2.data, cor3.data, cor4.data
        );
    }
}
