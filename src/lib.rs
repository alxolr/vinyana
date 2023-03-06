pub mod activation;
pub mod cost;

use std::{error::Error, fs::File, io::Read, path::Path};

use activation::Activation;
use ndarray::Array2;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

use crate::cost::mrse;

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    input_nodes: usize,
    output_nodes: usize,
    learning_rate: f32,
    weights: Vec<Array2<f32>>,
    biases: Vec<Array2<f32>>,
    activation: Activation,
}

impl NeuralNetwork {
    /// The posibility to initialise a completly new neural network
    /// All the weights will be random from -1 to 1 floats
    pub fn new(layers: Vec<usize>) -> NeuralNetwork {
        let rng = thread_rng();
        let mut weights: Vec<Array2<f32>> = vec![];
        let mut biases: Vec<Array2<f32>> = vec![];

        let input_nodes = layers.first().unwrap().clone();
        let output_nodes = layers.last().unwrap().clone();

        let randomize = |_| rng.clone().gen_range(-1.0..1.0);

        layers.windows(2).into_iter().for_each(|nodes| {
            let mut layer = Array2::zeros((nodes[1], nodes[0]));
            layer = layer.mapv(randomize);

            let mut bias = Array2::zeros((nodes[1], 1));
            bias = bias.mapv(randomize);

            weights.push(layer);
            biases.push(bias);
        });

        NeuralNetwork {
            input_nodes,
            output_nodes,
            learning_rate: 0.001,
            weights,
            biases,
            // w_input_hidden,
            // w_hidden_output,
            // w_bias_hidden,
            // w_bias_output,
            activation: Activation::new(activation::ActivationType::Relu),
        }
    }

    pub fn train(&mut self, inputs: Vec<f32>, targets: Vec<f32>) {
        let inputs = Array2::from_shape_vec((self.input_nodes, 1), inputs).unwrap();
        let targets = Array2::from_shape_vec((self.output_nodes, 1), targets).unwrap();

        let output_results =
            self.weights
                .iter()
                .enumerate()
                .fold(vec![inputs], |mut agg, (idx, weight)| {
                    let inputs = agg.last().unwrap();

                    // net on the line
                    // to compute gradients we will need this values ???
                    let net = weight.dot(inputs) + &self.biases[idx];

                    // after activation
                    let out = net.mapv(self.activation.f);
                    agg.push(out);

                    agg
                });

        // println!("{}", output_results.len());

        let ff_interm_outputs = output_results.iter().skip(1).rev().collect::<Vec<_>>();

        println!("ff interim {}", ff_interm_outputs.len());

        let final_outputs = ff_interm_outputs.first().unwrap().clone();
        let ff_interm_iter = ff_interm_outputs.into_iter().skip(1);

        let total_error = mrse(&targets, final_outputs);

        println!("{}", total_error);

        // let output_gradients =
        //     final_outputs.mapv(self.activation.df) * &output_errors * self.learning_rate;
        // let first_delta = output_gradients.dot()

        // let hidden_inputs = &self.w_input_hidden.dot(&inputs) + &self.w_bias_hidden;
        // let hidden_outputs = hidden_inputs.mapv(self.activation.f);

        // let final_inputs = &self.w_hidden_output.dot(&hidden_outputs) + &self.w_bias_output;
        // let final_outputs = final_inputs.mapv(self.activation.f);

        // let output_errors = &targets - &final_outputs;

        // // Compute the output gradient
        // let output_gradients = final_outputs.mapv(self.activation.df);
        // let output_gradients = output_gradients * &output_errors;
        // let output_gradients = &output_gradients * self.learning_rate;

        // let w_hidden_output_deltas = output_gradients.dot(&hidden_outputs.t());

        // self.w_hidden_output = &self.w_hidden_output + &w_hidden_output_deltas;
        // self.w_bias_output = &self.w_bias_output + &output_gradients;

        // // compute the hidden layer errors

        // let w_hidden_output_transposed = self.w_hidden_output.t();
        // let hidden_errors = w_hidden_output_transposed.dot(&output_errors);

        // let hidden_gradients = hidden_outputs.mapv(self.activation.df);
        // let hidden_gradients = hidden_gradients * &hidden_errors;
        // let hidden_gradients = &hidden_gradients * self.learning_rate;

        // let inputs_transposed = inputs.t();
        // let w_input_hidden_deltas = hidden_gradients.dot(&inputs_transposed);

        // self.w_input_hidden = &self.w_input_hidden + &w_input_hidden_deltas;
        // self.w_bias_hidden = &self.w_bias_hidden + &hidden_gradients;
    }

    pub fn predict(&self, inputs: Vec<f32>) -> Array2<f32> {
        let inputs = Array2::from_shape_vec((self.input_nodes, 1), inputs).unwrap();

        self.feed_forward(inputs)
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

    fn feed_forward(&self, inputs: Array2<f32>) -> Array2<f32> {
        self.weights
            .iter()
            .enumerate()
            .fold(inputs, |output, (idx, weight)| {
                let outputs = weight.dot(&output) + &self.biases[idx];
                outputs.mapv(self.activation.f)
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::NeuralNetwork;
    use rand::{prelude::SliceRandom, thread_rng};
    use std::path::Path;

    #[test]
    fn test_neural_network_initialisation() {
        let nn = NeuralNetwork::new(vec![3, 3, 1]);

        assert_eq!(nn.input_nodes, 3);
        assert_eq!(nn.output_nodes, 1);
    }

    #[test]
    fn test_xor_problem() {
        let mut nn = NeuralNetwork::new(vec![2, 2, 1]);
        let mut rng = thread_rng();

        let mut train_dataset = vec![
            (vec![1f32, 0.], vec![1f32]),
            (vec![1., 1.], vec![0.]),
            (vec![0., 0.], vec![0.]),
            (vec![0., 1.], vec![1.]),
        ];

        // for _ in 0..20000 {
        train_dataset.shuffle(&mut rng);

        train_dataset
            .iter()
            .for_each(|(inputs, targets)| nn.train(inputs.clone(), targets.clone()));
        // }

        let result = nn.predict(vec![1.0, 0.0]);
        let value = result[(0, 0)];

        println!("value {}", value);

        assert_eq!(value > 0.75, true);
    }

    #[test]
    fn test_load_xor_model() {
        let nn = NeuralNetwork::load(Path::new("./models/xor_model.bin"))
            .expect("couldn't deserialize the model");

        let result = nn.predict(vec![0.0, 1.0]);
        let value = result[(0, 0)];

        println!("{}", value);

        assert_eq!(value > 0.75, true);
    }
}
