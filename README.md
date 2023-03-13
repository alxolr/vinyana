# vinyana@0.3.1 [![Build Status](https://app.travis-ci.com/alxolr/sila.svg?branch=main)](https://app.travis-ci.com/alxolr/sila)[![codecov](https://codecov.io/gh/alxolr/vinyana/branch/main/graph/badge.svg?token=JMIBMAGT6I)](https://codecov.io/gh/alxolr/vinyana)

_vinyana_ - stands for mind in pali language.

## Goal

The goal of this project is to create a neural network library that is easy to use and understand in rust.

## Usage

```rust
use std::path::Path;

use rand::prelude::*;
use vinyana::{activation::ActivationType, NeuralNetwork};

fn main() {
    let mut nn = NeuralNetwork::new(vec![2, 2, 1]);

    nn.set_learning_rate(0.01); // default is 0.01 but you can change it
    nn.set_activation(ActivationType::Tanh); // default is Sigmoid but you can change it

    // We will train this network with 4 scenarios of XOR problem
    let scenarios = vec![
        (vec![1.0, 1.0], vec![0.0f32]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 0.0], vec![0.0]),
    ];

    let mut rng = thread_rng();
    for _ in 0..500000 {
        let random = rng.gen_range(0..4) as usize;
        let (train_data, target_data) = scenarios.get(random).unwrap();

        // we will pick a random scenario from the dataset and feed it to the network with the expected target
        nn.train(train_data.clone(), target_data.clone())
    }

    let result = nn.predict(vec![1.0, 0.0]);
    println!("Result: {:?}", result);

    // we can store our trained model and play with it later
    nn.save(Path::new("xor_model.nn")).unwrap();
}
```

```rust
// Load your model from file

let nn = NeuralNetwork::load(Path::new("xor_model.nn")).unwrap();

let result = nn.predict(vec![1.0, 1.0]);
println!("{:?}", result);
```
