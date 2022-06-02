use polars::prelude::*;
use polars::df;
use std::boxed::Box;
use std::fmt;

pub trait Split {
    fn split(self: Self) -> Self;
}

pub enum Node {
    DecisionNode {
        feature: String,
        threshold: f32,
        left: Box<&Node>,
        right: Box<&Node>
    },
    ClassNode {
        parent: Box<Option<Node>>,
        class: usize,
        samples: Vec<usize>
    }
}

impl Node {
    fn set_parent(&mut self, par: Node) -> &mut Self {
        if let Self::ClassNode {parent, class:_, samples:_} = self { 
            *parent = Box::new(Some(par));
            return self;
        } else {
            panic!("Cannot set parent of decision node");
        }
    }
}

impl Split for Node {
    fn split(self: Node) -> Node {
        match self {
            Node::ClassNode{parent: _, class:_, samples:_} => {
                let best_feature: String = String::from("b");
                let best_threshold: f32 = 0.0;
                let mut left = Node::ClassNode {parent: Box::new(None), class: 0, samples: vec![0; 5] };
                let mut right = Node::ClassNode {parent: Box::new(None), class: 1, samples: vec![1; 5] };
                let splitted: Node = Node::DecisionNode { 
                    feature: best_feature, 
                    threshold: best_threshold, 
                    left: Box::new(None), 
                    right: Box::new(None) 
                };
                return splitted;
            },
            Node::DecisionNode{feature:_, threshold:_, left:_, right:_} => {
                panic!("Cannot split a decision node.");
            }
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Node::ClassNode {
                parent: _,
                class, 
                samples: _
            } => write!(f, "{}", format!("ClassNode({})", class)),
            Node::DecisionNode {
                feature, 
                threshold, 
                left: _, 
                right: _
            } => write!(f, "{}", format!("DecisionNode([{}]<=[{}])", feature, threshold))
        }
    }
}

pub struct DecisionTree {
    
}

fn main() {
    let frame: DataFrame = df!(
        "index" => [1, 2, 3, 4, 5, 6, 7]
    ).unwrap();

    println!("Frame = {}", frame);

    let n = Node::ClassNode{parent: Box::new(None), class: 0, samples: vec![0,0,0,0,0,5,5,5,5,5]};
    println!("{}", n);
    
}
