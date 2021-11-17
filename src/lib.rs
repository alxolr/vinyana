use std::{error::Error, fs::File, io::Read, path::Path};

use ndarray::Array2;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

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
    pub w_input_hidden: Array2<f32>,
    pub w_hidden_output: Array2<f32>,
    pub w_bias_hidden: Array2<f32>,
    pub w_bias_output: Array2<f32>,
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

        let mut w_input_hidden = Array2::zeros((hidden_nodes, input_nodes));
        let mut w_hidden_output = Array2::zeros((output_nodes, hidden_nodes));

        let mut w_bias_hidden = Array2::zeros((hidden_nodes, 1));
        let mut w_bias_output = Array2::zeros((output_nodes, 1));

        let randomize = |_| rng.clone().gen_range(-1.0..1.0);

        // randomize weights
        w_input_hidden = w_input_hidden.mapv(randomize);
        w_hidden_output = w_hidden_output.mapv(randomize);

        // randomize biases
        w_bias_hidden = w_bias_hidden.mapv(randomize);
        w_bias_output = w_bias_output.mapv(randomize);

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

    pub fn train(&mut self, inputs: Vec<f32>, targets: Vec<f32>) {
        let inputs = Array2::from_shape_vec((self.input_nodes, 1), inputs).unwrap();
        let targets = Array2::from_shape_vec((self.output_nodes, 1), targets).unwrap();

        let hidden_inputs = &self.w_input_hidden.dot(&inputs) + &self.w_bias_hidden;
        let hidden_outputs = hidden_inputs.mapv(sigmoid);

        let final_inputs = &self.w_hidden_output.dot(&hidden_outputs) + &self.w_bias_output;
        let final_outputs = final_inputs.mapv(sigmoid);

        let output_errors = &targets - &final_outputs;
        
        // Compute the output gradient
        let output_gradients = final_outputs.mapv(sigmoid_derivate);
        let output_gradients = output_gradients * &output_errors;
        let output_gradients = &output_gradients * self.learning_rate;

        let w_hidden_output_deltas = output_gradients.dot(&hidden_outputs.t());

        self.w_hidden_output = &self.w_hidden_output + &w_hidden_output_deltas;
        self.w_bias_output = &self.w_bias_output + &output_gradients;

        // compute the hidden layer errors

        let w_hidden_output_transposed = self.w_hidden_output.t();
        let hidden_errors = w_hidden_output_transposed.dot(&output_errors);

        let hidden_gradients = hidden_outputs.mapv(sigmoid_derivate);
        let hidden_gradients = hidden_gradients * &hidden_errors;
        let hidden_gradients = &hidden_gradients * self.learning_rate;

        let inputs_transposed = inputs.t();
        let w_input_hidden_deltas = hidden_gradients.dot(&inputs_transposed);

        self.w_input_hidden = &self.w_input_hidden + &w_input_hidden_deltas;
        self.w_bias_hidden = &self.w_bias_hidden + &hidden_gradients;
    }

    pub fn predict(&self, inputs: Vec<f32>) -> Array2<f32> {
        let inputs = Array2::from_shape_vec((self.input_nodes, 1), inputs).unwrap();

        let hidden_inputs = &self.w_input_hidden.dot(&inputs) + &self.w_bias_hidden;
        let hidden_outputs = hidden_inputs.mapv(sigmoid);

        let final_inputs = &self.w_hidden_output.dot(&hidden_outputs) + &self.w_bias_output;
        let final_outputs = final_inputs.mapv(sigmoid);

        final_outputs
    }
}

#[cfg(test)]
mod tests {
    use rand::{prelude::SliceRandom, thread_rng};

    use crate::NeuralNetwork;

    #[test]
    fn test_neural_network_initialisation() {
        let nn = NeuralNetwork::new(3, 3, 1, 0.001);

        assert_eq!(nn.hidden_nodes, 3);
        assert_eq!(nn.input_nodes, 3);
        assert_eq!(nn.output_nodes, 1);
    }

    #[test]
    fn test_xor_problem() {
        let mut nn = NeuralNetwork::new(2, 2, 1, 0.05);
        let mut rng = thread_rng();

        let mut train_dataset = vec![
            (vec![1f32, 0.], vec![1f32]),
            (vec![1., 1.], vec![0.]),
            (vec![0., 0.], vec![0.]),
            (vec![0., 1.], vec![1.]),
        ];

        for _ in 0..8000 {
            train_dataset.shuffle(&mut rng);

            train_dataset
                .iter()
                .for_each(|(inputs, targets)| nn.train(inputs.clone(), targets.clone()));
        }

        let result = nn.predict(vec![1.0, 0.0]);
        let value = result[(0, 0)];
        assert_eq!(value > 0.75, true);
    }
}
