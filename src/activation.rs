/// List of activation functions and their derivates
/// For now we support ReLU, Tahn and Sigmoid
use serde::{de::Visitor, Deserialize, Serialize};

pub type Ff32 = fn(f32) -> f32;
pub const E: f32 = 2.7182818284590451f32;

pub enum ActivationType {
    Sigmoid,
    Relu,
    Tanh,
}

#[derive(Debug)]
pub struct Activation {
    /// The activation function
    pub f: Ff32,
    /// Derivative of the activation function used for backward propagation algorithm
    pub df: Ff32,
    /// This value is used for serialization, we serialize a string and build the activation from it
    t: String,
}

impl Activation {
    pub fn new(tp: ActivationType) -> Self {
        match tp {
            ActivationType::Sigmoid => Activation {
                f: sigmoid,
                df: sigmoid_derivative,
                t: "Sigmoid".to_string(),
            },
            ActivationType::Relu => Activation {
                f: relu,
                df: relu_derivative,
                t: "Relu".to_string(),
            },
            ActivationType::Tanh => Activation {
                f: tahn,
                df: tahn_derivative,
                t: "Tahn".to_string(),
            },
        }
    }
}

impl Serialize for Activation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.t)
    }
}

struct ActivationVisitor;

impl<'de> Visitor<'de> for ActivationVisitor {
    type Value = Activation;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a string one of the following (Sigmoid, Relu, Tahn)")
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        match v.as_ref() {
            "Sigmoid" => Ok(Activation::new(ActivationType::Sigmoid)),
            "Relu" => Ok(Activation::new(ActivationType::Relu)),
            "Tahn" => Ok(Activation::new(ActivationType::Tanh)),
            _ => panic!("looks like other value should be sent"),
        }
    }
}

impl<'de> Deserialize<'de> for Activation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(ActivationVisitor)
    }
}

// Sigmoid Activations
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

// ReLU activations
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_derivative(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        1.0
    }
}

// Tahn Activations
fn tahn(x: f32) -> f32 {
    (E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x))
}

fn tahn_derivative(x: f32) -> f32 {
    1.0 - x * x
}
