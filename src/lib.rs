use std::{error::Error, fs::File, io::Read, path::Path};

use matrix::Matrix;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

mod matrix;
pub const E: f32 = 2.7182818284590451f32;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid_derivate(x: f32) -> f32 {
    x * (1.0 - x)
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn tahn(x: f32) -> f32 {
    (E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x))
}

/// First iteration of Neural Network with only 3 layers input -> hidden -> output
#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub input_nodes: usize,
    pub hidden_nodes: usize,
    pub output_nodes: usize,
    pub learning_rate: f32,
    pub w_input_hidden: Matrix,
    pub w_hidden_output: Matrix,
    pub w_bias_hidden: Matrix,
    pub w_bias_output: Matrix,
}

impl NeuralNetwork {
    /// The posibility to initialise a completly new neural network
    /// All the weights will be random from -1 to 1 floats
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        learning_rate: f32,
    ) -> NeuralNetwork {
        let rng = thread_rng();

        let mut w_input_hidden = Matrix::new(hidden_nodes, input_nodes);
        let mut w_hidden_output = Matrix::new(output_nodes, hidden_nodes);

        let mut w_bias_hidden = Matrix::new(hidden_nodes, 1);
        let mut w_bias_output = Matrix::new(output_nodes, 1);

        let randomize = |_| rng.clone().gen_range(-1.0..1.0);

        // randomize weights
        w_input_hidden = w_input_hidden.map(randomize);
        w_hidden_output = w_hidden_output.map(randomize);

        // randomize biases
        w_bias_hidden = w_bias_hidden.map(randomize);
        w_bias_output = w_bias_output.map(randomize);

        NeuralNetwork {
            input_nodes,
            hidden_nodes,
            output_nodes,
            learning_rate,
            w_input_hidden,
            w_hidden_output,
            w_bias_hidden,
            w_bias_output,
        }
    }

    pub fn load(filepath: &Path) -> Result<NeuralNetwork, Box<dyn Error>> {
        let mut file = File::open(filepath)?;

        let mut contents = vec![];
        file.read_to_end(&mut contents)?;

        let nn: NeuralNetwork = bincode::deserialize(&contents)?;

        Ok(nn)
    }

    pub fn save(&self, filepath: &Path) -> Result<(), Box<dyn Error>> {
        let encoded = bincode::serialize(self).expect("Could not serialize the model");
        std::fs::write(filepath, encoded).expect("Could not save the model in the provided path");

        Ok(())
    }

    pub fn train(&mut self, inputs: &Vec<f32>, targets: &Vec<f32>) {
        // Do the prediction logic
        let inputs = Matrix::from_vec(inputs);

        let hidden_inputs = &(&self.w_input_hidden * &inputs) + &self.w_bias_hidden;
        let hidden_outputs = hidden_inputs.map(sigmoid);

        let final_inputs = &(&self.w_hidden_output * &hidden_outputs) + &self.w_bias_output;
        let final_outputs = final_inputs.map(sigmoid);

        let targets = Matrix::from_vec(targets);
        let output_errors = &targets - &final_outputs;

        // Compute the output gradient
        let output_gradients = final_outputs.map(sigmoid_derivate);
        let output_gradients = output_gradients.mul(&output_errors);
        let output_gradients = &output_gradients * self.learning_rate;

        let w_hidden_output_deltas = &output_gradients * &(hidden_outputs.transpose());

        self.w_hidden_output = &self.w_hidden_output + &w_hidden_output_deltas;
        self.w_bias_output = &self.w_bias_output + &output_gradients;

        // compute the hidden layer errors

        let w_hidden_output_transposed = self.w_hidden_output.transpose();
        let hidden_errors = &w_hidden_output_transposed * &output_errors;

        let hidden_gradients = hidden_outputs.map(sigmoid_derivate);
        let hidden_gradients = hidden_gradients.mul(&hidden_errors);
        let hidden_gradients = &hidden_gradients * self.learning_rate;

        let inputs_transposed = inputs.transpose();
        let w_input_hidden_deltas = &hidden_gradients * &inputs_transposed;

        self.w_input_hidden = &self.w_input_hidden + &w_input_hidden_deltas;
        self.w_bias_hidden = &self.w_bias_hidden + &hidden_gradients;
    }
    pub fn predict(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let inputs = Matrix::from_vec(inputs);

        let hidden_inputs = &(&self.w_input_hidden * &inputs) + &self.w_bias_hidden;
        let hidden_outputs = hidden_inputs.map(sigmoid);

        let final_inputs = &(&self.w_hidden_output * &hidden_outputs) + &self.w_bias_output;
        let final_outputs = final_inputs.map(sigmoid);

        final_outputs
            .data
            .iter()
            .flatten()
            .map(|x| *x)
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {

    use crate::NeuralNetwork;

    #[test]
    fn test_neural_network_initialisation() {
        let nn = NeuralNetwork::new(3, 3, 1, 0.001);

        assert_eq!(nn.hidden_nodes, 3);
        assert_eq!(nn.input_nodes, 3);
        assert_eq!(nn.output_nodes, 1);
    }
}
