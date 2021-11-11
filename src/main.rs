use std::path::Path;

use rand::{thread_rng, Rng};
use vinyana::NeuralNetwork;

fn main() {
    // train();
    let nn = NeuralNetwork::load(Path::new("xor_model.bin")).unwrap();

    let result = nn.predict(&vec![1.0, 1.0]);
    println!("{:?}", result);


}

fn train() {
    let mut nn = NeuralNetwork::new(2, 2, 1, 0.05);

    let scenarios = vec![
        (vec![1.0, 1.0], vec![0.0f32]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 0.0], vec![0.0]),
    ];

    println!("it's training");
    // training epochs

    let mut rng = thread_rng();
    for _ in 0..300000 {
        let random = rng.gen_range(0..4) as usize;
        let (train_data, target_data) = scenarios.get(random).unwrap();
        nn.train(train_data, target_data)
    }

    // we can store our trained model and play with it later
    nn.save(Path::new("xor_model.bin")).unwrap();
}
