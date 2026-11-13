use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak, atomic::Ordering},
    time::SystemTime,
};

use bytes::Bytes;
use futures_util::future::{BoxFuture, join_all};
use litellm_auth_aws::AwsAuthService;
use litellm_callbacks::{
    Activation, Callback, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, OptOutBy,
    Selection, Subscription,
};
use serde::Deserialize;
use serde_json::Value;
use tokio::time::Instant;

use crate::{
    key::{Stamp, download_name, logging_id, object_key, parent},
    limiter::Limiter,
    settings::{Environment, Granularity, Settings},
    upload::{Backoff, Clock, Object, Outcome, Uploader},
};

pub const NAME: &str = "s3_v2";
const JSON: &str = "application/json";
const NDJSON: &str = "application/x-ndjson";

pub trait Host: Send + Sync {
    fn callback_failed(&self);
    fn cold_storage_is_s3_v2(&self) -> bool;
}

pub struct S3LogParts {
    pub settings: Settings,
    pub client: litellm_http::Client,
    pub environment: Arc<Environment>,
    pub clock: Clock,
    pub backoff: Backoff,
    pub host: Arc<dyn Host>,
}

#[derive(Clone, Debug)]
struct Element {
    object: Object,
    retrying_since: Option<Instant>,
    is_file: bool,
}

#[derive(Default)]
struct Queue {
    items: Vec<Element>,
    requeued: usize,
}

pub struct S3Log {
    manifest: Manifest,
    uploader: Uploader,
    limiter: Limiter,
    host: Arc<dyn Host>,
    queue: Mutex<Queue>,
    flush_lock: tokio::sync::Mutex<()>,
    periodic: OnceLock<()>,
    this: Weak<S3Log>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MetadataKind {
    Missing,
    None,
    Mapping,
    Other,
}

#[derive(Debug, Deserialize)]
struct Event {
    body: String,
    start: Option<Stamp>,
    id: Option<String>,
    metadata: MetadataKind,
    team_alias: Option<Value>,
    key_alias: Option<Value>,
    cold_storage_object_key: Option<String>,
    cold_storage_s3_v2: bool,
}

fn manifest() -> Manifest {
    Manifest {
        selection: Selection {
            activation: Activation::DefaultOn,
            opt_out_by: vec![OptOutBy::Admin, OptOutBy::Caller],
            honors_no_log: true,
            ..Selection::named(NAME)
        },
        subscriptions: vec![Subscription {
            reads: FieldSet::of(&[Field::StandardLoggingPayload]),
            ..Subscription::on(Hook::AttemptEnded)
        }],
        has_request_state: false,
    }
}

impl S3Log {
    pub fn new(parts: S3LogParts) -> Arc<Self> {
        let settings = Arc::new(parts.settings);
        let limiter = Limiter::new(
            settings.max_concurrent_uploads,
            settings.adaptive_concurrency,
            settings.max_adaptive_concurrency,
        );
        Arc::new_cyclic(|this| Self {
            manifest: manifest(),
            uploader: Uploader {
                settings,
                client: parts.client,
                auth: AwsAuthService::default(),
                environment: parts.environment,
                clock: parts.clock,
                backoff: parts.backoff,
                retries: Default::default(),
            },
            limiter,
            host: parts.host,
            queue: Mutex::default(),
            flush_lock: tokio::sync::Mutex::new(()),
            periodic: OnceLock::new(),
            this: Weak::clone(this),
        })
    }

    pub fn settings(&self) -> &Settings {
        &self.uploader.settings
    }

    pub fn partition_granularity(&self) -> Granularity {
        self.settings()
            .partition_granularity(self.uploader.environment.as_ref())
    }

    pub fn log_prompts_only(&self) -> bool {
        self.settings()
            .log_prompts_only(self.uploader.environment.as_ref())
    }

    fn queue(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn queued_keys(&self) -> Vec<String> {
        self.queue()
            .items
            .iter()
            .map(|element| element.object.key.clone())
            .collect()
    }

    fn push(&self, element: Element) -> usize {
        self.start_periodic_flush();
        let mut queue = self.queue();
        queue.items.push(element);
        queue.items.len()
    }

    fn element(&self, event: Event) -> Result<Element, String> {
        let settings = self.settings();
        let start = event
            .start
            .ok_or_else(|| "the payload carries no start time".to_owned())?;
        let mut prefixes = Vec::new();
        for (enabled, alias) in [
            (settings.use_team_prefix, &event.team_alias),
            (settings.use_key_prefix, &event.key_alias),
        ] {
            if !enabled {
                continue;
            }
            if matches!(event.metadata, MetadataKind::None | MetadataKind::Other) {
                return Err("metadata has no get".to_owned());
            }
            match alias {
                None | Some(Value::Null) => {}
                Some(Value::String(alias)) => prefixes.push(alias.clone()),
                Some(_) => return Err("sequence item: expected str instance".to_owned()),
            }
        }
        let mut prefix = prefixes.join("/");
        if !prefix.is_empty() {
            prefix.push('/');
        }
        let file_name = event
            .id
            .as_deref()
            .map(|id| logging_id(&start, id))
            .unwrap_or_default();
        let path = settings.path.clone().unwrap_or_default();
        let key_for = |granularity| object_key(&path, &prefix, &start, &file_name, granularity);
        let cold_storage_object_key = match (event.cold_storage_s3_v2, &event.metadata) {
            (true, MetadataKind::Other) => return Err("metadata has no get".to_owned()),
            (true, MetadataKind::Mapping) => event.cold_storage_object_key,
            _ => None,
        };
        let key = match cold_storage_object_key {
            Some(cold)
                if cold == key_for(Granularity::Day) || cold == key_for(Granularity::Hour) =>
            {
                cold
            }
            _ => key_for(self.partition_granularity()),
        };
        let id = event
            .id
            .ok_or_else(|| "the payload id is missing or not a string".to_owned())?;
        Ok(Element {
            object: Object {
                key,
                body: Bytes::from(event.body),
                content_type: JSON,
                download_name: download_name(&start, &id),
            },
            retrying_since: None,
            is_file: false,
        })
    }

    pub fn submit(&self, value: &Value) {
        if value.is_null() {
            return;
        }
        let built = Event::deserialize(value)
            .map_err(|error| error.to_string())
            .and_then(|event| self.element(event));
        match built {
            Ok(element) => {
                self.push(element);
            }
            Err(error) => {
                tracing::error!("s3 Layer Error - {error}");
                self.host.callback_failed();
            }
        }
    }

    pub async fn submit_audit(&self, body: String, id: &str) {
        let now = Stamp::utc((self.uploader.clock)());
        let key = object_key(
            self.settings().path.as_deref().unwrap_or_default(),
            "audit_logs/",
            &now,
            &format!("{}_{id}", now.clock()),
            self.partition_granularity(),
        );
        let queued = self.push(Element {
            object: Object {
                key,
                body: Bytes::from(body),
                content_type: JSON,
                download_name: format!("audit-{id}.json"),
            },
            retrying_since: None,
            is_file: false,
        });
        if queued >= self.settings().batch_size {
            self.flush().await;
        }
    }

    pub async fn upload_now(&self, key: String, body: String, download_name: String) -> bool {
        let object = Object {
            key,
            body: Bytes::from(body),
            content_type: NDJSON,
            download_name,
        };
        self.start_periodic_flush();
        let outcome = self.uploader.upload(&object, &self.limiter, false).await;
        if outcome != Outcome::Delivered {
            self.host.callback_failed();
        }
        outcome == Outcome::Delivered
    }

    pub async fn download(&self, key: &str) -> Option<Bytes> {
        self.uploader.download(key).await
    }

    fn start_periodic_flush(&self) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        self.periodic.get_or_init(|| {
            let Some(sink) = self.this.upgrade() else {
                return;
            };
            runtime.spawn(async move {
                loop {
                    tokio::time::sleep(sink.settings().flush_interval).await;
                    sink.flush().await;
                }
            });
        });
    }

    pub async fn flush(&self) {
        let _flushing = self.flush_lock.lock().await;
        let (batch, requeued) = {
            let queue = self.queue();
            if queue.items.is_empty() {
                return;
            }
            (queue.items.clone(), queue.requeued)
        };
        let length = batch.len();
        if self.send_batch(batch, requeued).await {
            self.queue().items.drain(..length);
            return;
        }
        let max_queue_size = self.settings().max_queue_size;
        let mut queue = self.queue();
        let overflow = queue.items.len().saturating_sub(max_queue_size);
        if overflow > 0 {
            queue.items.drain(..overflow);
            tracing::warn!(
                "CustomLogger: log queue exceeded max_queue_size={max_queue_size}; dropped {overflow} oldest events."
            );
        }
    }

    fn batch_file_mode(&self) -> bool {
        if !self.settings().batch_file_upload {
            return false;
        }
        if self.host.cold_storage_is_s3_v2() {
            tracing::warn!(
                "s3 logging: s3_batch_file_upload is ignored because s3_v2 is the cold storage logger; per-request objects are required for spend log lookups"
            );
            return false;
        }
        true
    }

    fn batch_files(&self, batch: &[Element]) -> Vec<Element> {
        let now = Stamp::utc((self.uploader.clock)());
        let mut groups: BTreeMap<&str, Vec<&Element>> = BTreeMap::new();
        for element in batch.iter().filter(|element| !element.is_file) {
            groups
                .entry(parent(&element.object.key))
                .or_default()
                .push(element);
        }
        let files = groups.into_iter().map(|(folder, elements)| {
            let name = format!("batch_{}_{}", now.clock(), uuid::Uuid::new_v4().simple());
            let mut body = Vec::new();
            for (index, element) in elements.iter().enumerate() {
                if index > 0 {
                    body.push(b'\n');
                }
                body.extend_from_slice(&element.object.body);
            }
            Element {
                object: Object {
                    key: if folder.is_empty() {
                        format!("{name}.jsonl")
                    } else {
                        format!("{folder}/{name}.jsonl")
                    },
                    body: Bytes::from(body),
                    content_type: NDJSON,
                    download_name: format!("{name}.jsonl"),
                },
                retrying_since: elements
                    .iter()
                    .filter_map(|element| element.retrying_since)
                    .min(),
                is_file: true,
            }
        });
        batch
            .iter()
            .filter(|element| element.is_file)
            .cloned()
            .chain(files)
            .collect()
    }

    async fn upload_bounded(&self, element: &Element) -> Outcome {
        let _permit = self.limiter.acquire().await;
        let outcome = self
            .uploader
            .upload(&element.object, &self.limiter, true)
            .await;
        if outcome != Outcome::Delivered {
            self.host.callback_failed();
        }
        outcome
    }

    async fn send_batch(&self, batch: Vec<Element>, requeued_count: usize) -> bool {
        let uploads = if self.batch_file_mode() {
            self.batch_files(&batch)
        } else {
            batch.clone()
        };
        self.uploader.retries.store(0, Ordering::Relaxed);
        let stale = if uploads.len() == batch.len() {
            requeued_count.min(uploads.len())
        } else {
            0
        };
        let order: Vec<usize> = (stale..uploads.len()).chain(0..stale).collect();
        let finished = join_all(
            order
                .iter()
                .map(|index| self.upload_bounded(&uploads[*index])),
        )
        .await;
        let mut outcomes = vec![Outcome::Retry; uploads.len()];
        for (index, outcome) in order.into_iter().zip(finished) {
            outcomes[index] = outcome;
        }
        let retries = self.uploader.retries.load(Ordering::Relaxed);
        if retries > 0 {
            tracing::warn!(
                "s3 logging: {retries} in-call retries across {} uploads this flush",
                uploads.len()
            );
        }
        let delivered = outcomes
            .iter()
            .filter(|outcome| **outcome == Outcome::Delivered)
            .count();
        let failed: Vec<(&Element, Outcome)> = uploads
            .iter()
            .zip(outcomes)
            .filter(|(_, outcome)| *outcome != Outcome::Delivered)
            .collect();
        let now = Instant::now();
        let max_age = self.settings().max_retry_age;
        let requeued: Vec<Element> = if delivered == 0 {
            failed
                .iter()
                .map(|(element, _)| (*element).clone())
                .collect()
        } else {
            failed
                .iter()
                .filter(|(element, outcome)| {
                    *outcome != Outcome::Dropped
                        && !matches!(
                            (max_age, element.retrying_since),
                            (Some(max_age), Some(since)) if now.duration_since(since) > max_age
                        )
                })
                .map(|(element, _)| {
                    let mut element = (*element).clone();
                    if element.retrying_since.is_none() && max_age.is_some() {
                        element.retrying_since = Some(now);
                    }
                    element
                })
                .collect()
        };
        let dropped = failed.len() - requeued.len();
        if dropped > 0 {
            let terminal = failed
                .iter()
                .filter(|(_, outcome)| *outcome == Outcome::Dropped)
                .count();
            tracing::warn!(
                "s3 logging: {dropped} uploads dropped this flush: {terminal} rejected as terminal by S3, {} retrying longer than s3_max_retry_age_seconds={:?}",
                dropped - terminal,
                max_age.map(|age| age.as_secs())
            );
        }
        let mut queue = self.queue();
        if requeued.is_empty() {
            queue.requeued = 0;
            return true;
        }
        let max_queue_size = self.settings().max_queue_size;
        let split = batch.len().min(queue.items.len());
        let arrivals = queue.items.split_off(split);
        let overflow = (requeued.len() + arrivals.len()).saturating_sub(max_queue_size);
        if overflow > 0 {
            tracing::warn!(
                "s3 logging: queue exceeded max_queue_size={max_queue_size} after a failed flush, dropped {overflow} oldest events"
            );
        }
        let requeued_length = requeued.len();
        queue.items = requeued
            .into_iter()
            .chain(arrivals)
            .skip(overflow)
            .collect();
        queue.requeued = requeued_length.saturating_sub(overflow);
        tracing::error!(
            "{} of {} S3 uploads failed; transient failures kept in queue for the next flush",
            failed.len(),
            uploads.len()
        );
        false
    }
}

impl Callback for S3Log {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            if input.hook == Hook::AttemptEnded
                && let Some(value) = input.fields.raw(Field::StandardLoggingPayload)?
            {
                self.submit(value);
            }
            Ok(None)
        })
    }

    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(S3Log::flush(self))
    }
}

pub fn system_clock() -> SystemTime {
    SystemTime::now()
}
