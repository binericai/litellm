use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;

use crate::model::{Descriptor, Window};

#[derive(Debug, PartialEq)]
pub struct Hold {
    pub descriptor: Option<Descriptor>,
    pub reserved_tokens: i64,
    pub window: Option<Window>,
}

#[derive(Debug, Default)]
pub struct Stash {
    pub priority_class_share: Option<Descriptor>,
    pub priority: Option<Value>,
    pub borrowing_limit: Option<Descriptor>,
    pub reserved_tokens: i64,
    pub reserved_window: Option<Window>,
    pub fallback_holds: Vec<(String, Arc<Hold>)>,
    pub steered_from: Option<String>,
}

impl Stash {
    pub fn hold(&self, model: &str) -> Option<Arc<Hold>> {
        self.fallback_holds
            .iter()
            .find(|(name, _)| name == model)
            .map(|(_, hold)| Arc::clone(hold))
    }

    pub fn put_hold(&mut self, model: &str, hold: &Arc<Hold>) {
        match self
            .fallback_holds
            .iter_mut()
            .find(|(name, _)| name == model)
        {
            Some((_, slot)) => *slot = Arc::clone(hold),
            None => self
                .fallback_holds
                .push((model.to_owned(), Arc::clone(hold))),
        }
    }

    pub fn drop_hold(&mut self, hold: &Arc<Hold>) {
        self.fallback_holds
            .retain(|(_, held)| !Arc::ptr_eq(held, hold));
    }
}

pub type SharedStash = Arc<Mutex<Stash>>;
