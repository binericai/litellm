use std::{
    collections::BTreeSet,
    sync::{Mutex, MutexGuard, PoisonError},
};

use litellm_redis_store::RequestRedis;
use serde::{Deserialize, Serialize};

use crate::{engine::SlotAcquisition, status::Response};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitedModel {
    pub requested: String,
    pub group: String,
}

#[derive(Clone, Debug, Default)]
pub struct Stash {
    pub owned: bool,
    pub first_pass_admitted: bool,
    pub owner_call_id: Option<String>,
    pub rate_limit_response: Option<Response>,
    pub parallel_slot: Option<SlotAcquisition>,
    pub reserved_tokens: i64,
    pub reserved_model: Option<RateLimitedModel>,
    pub reserved_scopes: BTreeSet<(String, String)>,
    pub fallback_models: BTreeSet<String>,
    pub reservation_released: bool,
    pub tpm_limited_tags: BTreeSet<String>,
}

impl Stash {
    pub fn discard_released_tpm_reservation(&mut self) {
        if self.reservation_released {
            self.reserved_tokens = 0;
            self.reservation_released = false;
        }
    }

    pub fn for_call(&self, call_id: Option<&str>) -> bool {
        match (&self.owner_call_id, call_id) {
            (Some(owner), Some(call_id)) => owner == call_id,
            _ => true,
        }
    }
}

#[derive(Default)]
pub struct LimiterState {
    stash: Mutex<Stash>,
    request: Mutex<Option<RequestRedis>>,
    pub(crate) release: tokio::sync::Mutex<()>,
}

impl LimiterState {
    pub fn stash(&self) -> MutexGuard<'_, Stash> {
        self.stash.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn snapshot(&self) -> Stash {
        self.stash().clone()
    }

    pub fn bind_request(&self, request: Option<RequestRedis>) {
        *self.request.lock().unwrap_or_else(PoisonError::into_inner) = request;
    }

    pub fn request(&self) -> Option<RequestRedis> {
        self.request
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}
