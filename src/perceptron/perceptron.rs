/* Perceptron structure */

use std::error::Error;
use std::iter::zip;

use float_cmp::approx_eq;
use simple_error::bail;

pub struct Perceptron {
    weights: Vec<f64>,
}

impl Perceptron {
    pub fn new(size: usize) -> Self {
        Self { weights: vec![1.0; size] }
    }

    pub fn execute(&mut self, input: Vec<f64>, bias: f64) -> Result<f64, Box<dyn Error>> {
        if input.len() != self.weights.len() {
            bail!("Input does not have correct size! Expected {} Actual {}", self.weights.len(), input.len())
        } else {
            Ok(zip(input.iter(), self.weights.iter()).fold(0.0, |a, i| a + (i.0 * i.1)) + bias)
        }
    }

    pub fn stochastic_gradient_descent(&mut self, learning_rate: f64, expected: f64, predicted: f64, input: Vec<f64>) -> Result<(), Box<dyn Error>> {
        if input.len() != self.weights.len() {
            bail!("Input does not have correct size! Expected {} Actual {}", self.weights.len(), input.len())
        } else if approx_eq!(f64, learning_rate, 0.0) {
            bail!("Learning rate is approximately 0!")
        } else {
            for (i, w) in self.weights.iter_mut().enumerate() {
                *w = *w + (learning_rate * (expected - predicted) * input[i]);
            }
            Ok(())
        }
    }
}