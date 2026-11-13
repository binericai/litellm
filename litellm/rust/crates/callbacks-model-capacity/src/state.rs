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
    fn register(&self, call_id: &str, stash: &SharedStash, anchored: bool) {
        let mut entries = self.entries.lock();
        let now = Instant::now();
        entries.retain(|_, entry| entry.anchored || now.duration_since(entry.since) < ORPHAN_AGE);
        entries.insert(
            call_id.to_owned(),
            Entry {
                stash: Arc::clone(stash),
                since: now,
                anchored,
            },
        );
    }

    pub fn find(&self, call_id: &str) -> Option<SharedStash> {
        self.entries
            .lock()
            .get(call_id)
            .map(|entry| Arc::clone(&entry.stash))
    }

    pub fn find_or_claim(&self, call_id: Option<&str>) -> SharedStash {
        if let Some(stash) = call_id.and_then(|call_id| self.find(call_id)) {
            return stash;
        }
        let stash = SharedStash::default();
        if let Some(call_id) = call_id {
            self.register(call_id, &stash, false);
        }
        stash
    }

    fn release(&self, call_id: &str, stash: &SharedStash) {
        let mut entries = self.entries.lock();
        if entries
            .get(call_id)
            .is_some_and(|entry| Arc::ptr_eq(&entry.stash, stash))
        {
            entries.remove(call_id);
        }
    }
}

#[derive(Default)]
struct Claimed {
    stash: Option<SharedStash>,
    owner: Option<String>,
    registered: Vec<String>,
}

pub struct Anchor {
    stashes: Arc<Stashes>,
    claimed: Mutex<Claimed>,
}

impl Anchor {
    pub fn new(stashes: Arc<Stashes>) -> Self {
        Self {
            stashes,
            claimed: Mutex::new(Claimed::default()),
        }
    }

    pub fn claim(&self, call_id: Option<&str>) -> SharedStash {
        let mut claimed = self.claimed.lock();
        let stash = Arc::clone(claimed.stash.get_or_insert_with(SharedStash::default));
        if let Some(call_id) = call_id {
            claimed.owner = Some(call_id.to_owned());
            self.stashes.register(call_id, &stash, true);
            claimed.registered.push(call_id.to_owned());
        }
        stash
    }

    pub fn current(&self) -> Option<(Option<String>, SharedStash)> {
        let claimed = self.claimed.lock();
        claimed
            .stash
            .as_ref()
            .map(|stash| (claimed.owner.clone(), Arc::clone(stash)))
    }

    pub fn for_call(&self, call_id: Option<&str>) -> Option<SharedStash> {
        let (owner, stash) = self.current()?;
        match (owner, call_id) {
            (Some(owner), Some(call_id)) => (owner == call_id).then_some(stash),
            _ => Some(stash),
        }
    }
}

impl Drop for Anchor {
    fn drop(&mut self) {
        let claimed = std::mem::take(&mut *self.claimed.lock());
        if let Some(stash) = claimed.stash {
            for call_id in claimed.registered {
                self.stashes.release(&call_id, &stash);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::same_call(Some("a"), Some("a"), true)]
    #[case::other_call(Some("a"), Some("b"), false)]
    #[case::no_call_id(Some("a"), None, true)]
    #[case::no_owner(None, Some("b"), true)]
    fn a_request_stash_answers_for_a_call_like_python(
        #[case] owner: Option<&str>,
        #[case] call_id: Option<&str>,
        #[case] found: bool,
    ) {
        let anchor = Anchor::new(Arc::default());
        let _ = anchor.claim(owner);
        assert_eq!(anchor.for_call(call_id).is_some(), found);
    }

    #[rstest]
    fn an_unclaimed_request_has_no_stash() {
        let anchor = Anchor::new(Arc::default());
        assert!(anchor.for_call(None).is_none());
    }

    #[rstest]
    fn dropping_the_anchor_forgets_its_call_ids() {
        let stashes = Arc::new(Stashes::default());
        let anchor = Anchor::new(Arc::clone(&stashes));
        let _ = anchor.claim(Some("a"));
        assert!(stashes.find("a").is_some());
        drop(anchor);
        assert!(stashes.find("a").is_none());
    }
}
