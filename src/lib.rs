use matrix::Matrix;
use rand::{thread_rng, Rng};

mod matrix;
pub const E: f32 = 2.7182818284590451f32;

type Float2D = Vec<Vec<f32>>;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn reverse(x: &Matrix) -> Matrix {
    let matrix = x * -1.0;

    &matrix + 1.0
}

/// First iteration of Neural Network with only 3 layers input -> hidden -> output
pub struct NeuralNetwork {
    pub input_nodes: usize,
    pub hidden_nodes: usize,
    pub output_nodes: usize,
    pub learning_rate: f32,
    pub wih: Matrix,
    pub who: Matrix,
}

impl NeuralNetwork {
    /// The posibility to initialise a completly new neural network
    /// All the weights will be random from -1 to 1 floats
    pub fn new(inputs: usize, hidden: usize, outputs: usize, learning_rate: f32) -> NeuralNetwork {
        let rng = thread_rng();

        let mut wih = Matrix::new(hidden, inputs);
        let mut who = Matrix::new(outputs, hidden);

        let randomize = |_| rng.clone().gen_range(-1.0..1.0);

        wih = wih.map(randomize);
        who = who.map(randomize);

        NeuralNetwork {
            input_nodes: inputs,
            hidden_nodes: hidden,
            output_nodes: outputs,
            learning_rate,
            wih,
            who,
        }
    }

    pub fn train(&mut self, inputs: &Float2D, targets: &Float2D) {
        let inputs = Matrix::from_vec(inputs);
        let targets = Matrix::from_vec(targets);

        let hidden_inputs = &self.wih * &inputs;
        let hidden_outputs = hidden_inputs.map(sigmoid);

        let final_inputs = &self.who * &hidden_outputs;
        let final_outputs = final_inputs.map(sigmoid);

        let output_errors = &targets - &final_outputs;
        let hidden_errors = &self.who.transpose() * &output_errors;

        self.who =
            &self.who + &(self.back_propagate(&output_errors, &final_outputs, &hidden_inputs));

        self.wih = &self.wih + &(self.back_propagate(&hidden_errors, &hidden_outputs, &inputs));
    }

    fn back_propagate(&self, errors: &Matrix, outputs: &Matrix, inputs: &Matrix) -> Matrix {
        let errors_outputs = errors.mul(outputs);
        let rev_outputs = reverse(outputs);
        let err_rev_outputs = errors_outputs.mul(&rev_outputs);
        let err_inputs = &err_rev_outputs * &inputs.transpose();

        &err_inputs * self.learning_rate
    }

    pub fn predict(&self, inputs: &Float2D) -> Matrix {
        let inputs_m = Matrix::from_vec(inputs);

        let hidden_inputs = &self.wih.clone() * &inputs_m;
        let hidden_outputs = hidden_inputs.map(sigmoid);

        let final_inputs = &self.who.clone() * &hidden_outputs;
        let final_outputs = final_inputs.map(sigmoid);

        final_outputs
    }
}

#[cfg(test)]
mod tests {
    use rand::{prelude::SliceRandom, thread_rng, Rng};

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
        let nn = NeuralNetwork::new(2, 2, 1, 0.01);

        let cor1 = nn.predict(&vec![vec![1.0], vec![0.0]]);
        let cor2 = nn.predict(&vec![vec![1.0], vec![0.0]]);
        let cor3 = nn.predict(&vec![vec![1.0], vec![0.0]]);
        let cor4 = nn.predict(&vec![vec![1.0], vec![0.0]]);

        println!(
            "predicted {:?} {:?} {:?} {:?}",
            cor1.data, cor2.data, cor3.data, cor4.data
        );
    }

    #[test]
    fn test_train_is_building() {
        let mut nn = NeuralNetwork::new(2, 2, 1, 0.05);

        let mut scenarios = vec![
            (vec![vec![0.0], vec![1.0]], vec![vec![1.0]]),
            (vec![vec![1.0], vec![0.0]], vec![vec![1.0]]),
            (vec![vec![0.0], vec![0.0]], vec![vec![0.0]]),
            (vec![vec![1.0], vec![1.0]], vec![vec![0.0f32]]),
        ];

        println!("it's training");
        // training epochs
        for _ in 0..10000 {
            // let random = thread_rng().gen_range(0..scenarios.len());
            // // let (train_data, target_data) = scenarios.get(random).unwrap();
            // let mut rng = thread_rng();
            // scenarios.shuffle(&mut rng);

            scenarios
                .iter()
                .for_each(|(train_data, target_data)| nn.train(&train_data, target_data))
        }

        // for scenario in scenarios {}

        let t = nn.predict(&vec![vec![1.0], vec![1.0]]);
        let f = nn.predict(&vec![vec![1.0], vec![1.0]]);

        println!("Predict true {:?}", &t.data);
        println!("Predic false {:?}", &f.data);

        // println!("error {:?}", res.data);
    }
}
