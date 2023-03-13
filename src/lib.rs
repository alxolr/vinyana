pub mod activation;
pub mod cost;

use std::{error::Error, fs::File, io::Read, path::Path};

use activation::Activation;
use ndarray::Array2;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

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
            learning_rate: 0.01,
            weights,
            biases,
            activation: Activation::new(activation::ActivationType::Sigmoid),
        }
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    pub fn set_activation(&mut self, activation: activation::ActivationType) {
        self.activation = Activation::new(activation);
    }

    pub fn train(&mut self, inputs: Vec<f32>, targets: Vec<f32>) {
        let inputs = Array2::from_shape_vec((self.input_nodes, 1), inputs).unwrap();
        let targets = Array2::from_shape_vec((self.output_nodes, 1), targets).unwrap();

        let mut output_results = vec![];
        self.weights
            .iter()
            .enumerate()
            .fold(vec![inputs], |mut agg, (idx, weight)| {
                let inputs = agg.last().unwrap();

                let net = weight.dot(inputs) + &self.biases[idx];

                let out = net.mapv(self.activation.f);
                agg.push(out.clone());
                output_results.push(out);

                agg
            });

        let mut error = targets - output_results.last().unwrap();
        let mut gradients = vec![];

        for (idx, output) in output_results.iter().enumerate().rev() {
            let gradient = output.mapv(self.activation.df) * &error;
            gradients.push(gradient);

            error = self.weights[idx].t().dot(&error);
        }

        gradients.reverse();

        for (idx, weight) in self.weights.iter_mut().enumerate() {
            let gradient = gradients[idx].clone();
            let inputs = output_results[idx].clone();

            let delta = gradient.dot(&inputs.t()) * self.learning_rate;
            *weight += &delta;
        }

        for (idx, bias) in self.biases.iter_mut().enumerate() {
            let gradient = gradients[idx].clone();
            let delta = gradient * self.learning_rate;
            *bias += &delta;
        }
    }

    pub fn predict(&self, inputs: Vec<f32>) -> Vec<f32> {
        let inputs = Array2::from_shape_vec((self.input_nodes, 1), inputs).unwrap();

        let result = self.feed_forward(inputs);

        result.iter().map(|x| *x).collect()
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
    use crate::{activation, NeuralNetwork};
    use rand::{prelude::SliceRandom, thread_rng};

    #[test]
    fn test_neural_network_initialisation() {
        let nn = NeuralNetwork::new(vec![3, 3, 1]);

        assert_eq!(nn.input_nodes, 3);
        assert_eq!(nn.output_nodes, 1);
    }

    #[test]
    fn test_xor_problem() {
        let mut nn = NeuralNetwork::new(vec![2, 2, 1]);
        nn.set_activation(activation::ActivationType::Tanh);
        nn.set_learning_rate(0.01);

        let mut rng = thread_rng();

        let mut train_dataset = vec![
            (vec![1f32, 0.], vec![1f32]),
            (vec![1., 1.], vec![0.]),
            (vec![0., 0.], vec![0.]),
            (vec![0., 1.], vec![1.]),
        ];

        for _ in 0..50000 {
            train_dataset.shuffle(&mut rng);

            train_dataset
                .iter()
                .for_each(|(inputs, targets)| nn.train(inputs.clone(), targets.clone()));
        }

        let result = nn.predict(vec![1.0, 0.0]);
        let value = result.first().unwrap();

        assert_eq!(value > &0.75, true);
    }
}
