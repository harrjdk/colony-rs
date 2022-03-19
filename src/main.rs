extern crate simple_error;

use colony_rs::perceptron::perceptron::Perceptron;
use rand::distributions::{Distribution, Uniform};

fn main() {
    // demo for perceptron for now...
    let mut rng = rand::thread_rng();
    let die = Uniform::from(1.0..10.0);
    println!("Begining perceptron test...");
    let mut p0 = Perceptron::new(2);
    println!("Approximating x * y...");
    let mut hits = 0;
    let mut misses = 0;
    for _ in 0..10000000 {
        let x:f64 = die.sample(&mut rng);
        let y:f64 = die.sample(&mut rng);
        let z = x * y;
        let input = vec![x,y];
        match p0.execute(input, 0.0) {
            Err(e) =>  {
                eprintln!("Error! {}", e);
                break;
            },
            Ok(r) => {
                p0.stochastic_gradient_descent(0.0001, z, r, vec![x,y]).unwrap();
                let diff = (z-r).abs();
                if diff < z*0.1 {
                    hits += 1;
                } else {
                    misses += 1;
                }
            },
        };
    }
    if hits > 0 {
        let total = hits + misses;
        println!("Hits: {}\tMisses: {}\tTotal: {}", hits, misses, total);
        println!("Hit rate: {}", (hits as f64)/(total as f64));
    } else {
        println!("All misses!");
    }
}
