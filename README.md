# vinyana [![Build Status](https://app.travis-ci.com/alxolr/sila.svg?branch=main)](https://app.travis-ci.com/alxolr/sila)[![codecov](https://codecov.io/gh/alxolr/vinyana/branch/main/graph/badge.svg?token=JMIBMAGT6I)](https://codecov.io/gh/alxolr/vinyana)


_vinyana_ - stands for mind in pali language.

## Goal

To implement a simple Neural Network Library in order to understand the maths behind it.

This is a learning project, not intended to become mainstream lib.


The mantra here is:

> In order to understand something you need to build it yourself.


## Usage

```rust
// In this example we will teach our NeuralNetwork to solve XOR problem

// we will create a 3 layer neural network with 2 inputs 2 hidden and 1 output node

use vinyana::NeuralNetwork;

let mut nn = NeuralNetwork::new(2, 2, 1, 0.05);


// We will train this network 
let scenarios = vec![
    (vec![1.0, 1.0], vec![0.0f32]),
    (vec![0.0, 1.0], vec![1.0]),
    (vec![1.0, 0.0], vec![1.0]),
    (vec![0.0, 0.0], vec![0.0]),
];

let mut rng = thread_rng();
for _ in 0..300000 {
    let random = rng.gen_range(0..4) as usize;
    let (train_data, target_data) = scenarios.get(random).unwrap();

    // we will pick a random scenario from the dataset and feed it to the network with the expected target
    nn.train(train_data, target_data)
}

// we can store our trained model and play with it later
nn.save(Path::new("xor_model.bin")).unwrap();
```


```rust
// Load your model from file

let nn = NeuralNetwork::load(Path::new("xor_model.bin")).unwrap();

let result = nn.predict(&vec![1.0, 1.0]);
println!("{:?}", result);
```