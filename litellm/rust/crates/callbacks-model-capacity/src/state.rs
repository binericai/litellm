use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

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

const ORPHAN_AGE: Duration = Duration::from_secs(3600);

struct Entry {
    stash: SharedStash,
    since: Instant,
    anchored: bool,
}

#[derive(Default)]
pub struct Stashes {
    entries: Mutex<HashMap<String, Entry>>,
}

impl Stashes {
    pub fn claim(&self, call_id: &str, anchored: bool) -> SharedStash {
        let mut entries = self.entries.lock();
        let now = Instant::now();
        entries.retain(|_, entry| entry.anchored || now.duration_since(entry.since) < ORPHAN_AGE);
        let entry = entries.entry(call_id.to_owned()).or_insert_with(|| Entry {
            stash: SharedStash::default(),
            since: now,
            anchored: false,
        });
        entry.anchored |= anchored;
        Arc::clone(&entry.stash)
    }

    pub fn find(&self, call_id: &str) -> Option<SharedStash> {
        self.entries
            .lock()
            .get(call_id)
            .map(|entry| Arc::clone(&entry.stash))
    }

    pub fn release(&self, call_id: &str, stash: &SharedStash) {
        let mut entries = self.entries.lock();
        if entries
            .get(call_id)
            .is_some_and(|entry| Arc::ptr_eq(&entry.stash, stash))
        {
            entries.remove(call_id);
        }
    }
}

pub struct Anchor {
    stashes: Arc<Stashes>,
    claimed: Mutex<Option<(String, SharedStash)>>,
}

impl Anchor {
    pub fn new(stashes: Arc<Stashes>) -> Self {
        Self {
            stashes,
            claimed: Mutex::new(None),
        }
    }

    pub fn hold(&self, call_id: &str) -> SharedStash {
        let stash = self.stashes.claim(call_id, true);
        *self.claimed.lock() = Some((call_id.to_owned(), Arc::clone(&stash)));
        stash
    }
}

impl Drop for Anchor {
    fn drop(&mut self) {
        if let Some((call_id, stash)) = self.claimed.lock().take() {
            self.stashes.release(&call_id, &stash);
        }
    }
}
