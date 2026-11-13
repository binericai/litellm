use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use futures_util::future::BoxFuture;
use litellm_callbacks_s3_log::{Constants, Host, S3Log, S3LogParts, Settings};
use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{method, path_regex},
};

#[derive(Default)]
struct Recorder {
    failures: AtomicUsize,
    next: AtomicU64,
    payloads: Mutex<HashMap<u64, Option<String>>>,
    region: Mutex<Option<Result<Option<String>, String>>>,
    region_panics: AtomicBool,
}

const PANICS: &str = "panics while rendering";

impl Recorder {
    fn stash(&self, body: Option<&str>) -> u64 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.payloads
            .lock()
            .unwrap()
            .insert(id, body.map(str::to_owned));
        id
    }

    fn rewrite(&self, id: u64, body: &str) {
        self.payloads
            .lock()
            .unwrap()
            .insert(id, Some(body.to_owned()));
    }

    fn held(&self) -> usize {
        self.payloads.lock().unwrap().len()
    }

    fn failures(&self) -> usize {
        self.failures.load(Ordering::Relaxed)
    }
}

impl Host for Recorder {
    fn callback_failed(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    fn cold_storage_is_s3_v2(&self) -> bool {
        false
    }

    fn render(&self, payloads: Vec<u64>) -> BoxFuture<'static, Result<Vec<Bytes>, String>> {
        let bodies: Vec<Option<String>> = payloads
            .iter()
            .map(|payload| {
                self.payloads
                    .lock()
                    .unwrap()
                    .get(payload)
                    .cloned()
                    .flatten()
            })
            .collect();
        let rendered = bodies
            .into_iter()
            .map(|body| match body {
                Some(body) if body == PANICS => panic!("dictionary changed size during iteration"),
                Some(body) => Ok(Bytes::from(body)),
                None => Err("the payload cannot be encoded".to_owned()),
            })
            .collect();
        Box::pin(std::future::ready(rendered))
    }

    fn release(&self, payload: u64) {
        self.payloads.lock().unwrap().remove(&payload);
    }

    fn region_from_environment(&self) -> Result<Option<String>, String> {
        assert!(
            !self.region_panics.load(Ordering::Relaxed),
            "the host panicked"
        );
        self.region.lock().unwrap().clone().unwrap_or(Ok(None))
    }

    fn web_identity_token(&self, reference: &str) -> Result<String, String> {
        Ok(reference.to_owned())
    }
}

fn fixed_clock() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_763_024_707)
}

fn no_env(_: &str) -> Option<String> {
    None
}

fn sink(server: &MockServer, extra: Value) -> (Arc<S3Log>, Arc<Recorder>) {
    sink_with(server, extra, Constants::default())
}

fn sink_with(
    server: &MockServer,
    extra: Value,
    constants: Constants,
) -> (Arc<S3Log>, Arc<Recorder>) {
    let mut params = json!({
        "s3_bucket_name": "logs",
        "s3_region_name": "us-east-1",
        "s3_endpoint_url": server.uri(),
        "s3_aws_access_key_id": "AKIDEXAMPLE",
        "s3_aws_secret_access_key": "secret",
        "s3_path": "base",
    });
    params
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    params
        .as_object_mut()
        .unwrap()
        .retain(|_, value| !value.is_null());
    let host = Arc::new(Recorder::default());
    let sink = S3Log::new(S3LogParts {
        settings: Settings::from_params(params.as_object().unwrap(), &constants),
        client: litellm_http::Client::for_test(),
        download_client: litellm_http::Client::for_test(),
        environment: Arc::new(no_env),
        clock: fixed_clock,
        backoff: Arc::new(|_| Duration::ZERO),
        host: Arc::clone(&host) as Arc<dyn Host>,
    });
    (sink, host)
}

fn event(host: &Recorder, id: &str, body: &str) -> Value {
    event_with(host.stash(Some(body)), id)
}

fn event_with(payload: u64, id: &str) -> Value {
    json!({
        "body": payload,
        "start": [2026, 11, 13, 9, 5, 7, 42],
        "id": id,
        "metadata": "mapping",
        "team_alias": "team-a",
        "key_alias": null,
        "cold_storage_object_key": null,
        "cold_storage_s3_v2": false,
    })
}

fn header(request: &Request, name: &str) -> Option<String> {
    request
        .headers
        .get(name)
        .map(|value| value.to_str().unwrap().to_owned())
}

#[rstest]
#[tokio::test]
async fn a_flush_puts_one_object_per_event_with_the_python_headers() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(
        &server,
        json!({"s3_use_team_prefix": true, "s3_sse_kms_key_id": "kms-1"}),
    );
    sink.submit(&event(&host, "chatcmpl-1", "{\"id\": \"chatcmpl-1\"}"));
    sink.flush().await;

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let put = &requests[0];
    assert_eq!(
        put.url.path(),
        "/logs/base/team-a/2026-11-13/time-09-05-07-000042_chatcmpl-1.json"
    );
    assert_eq!(put.body, b"{\"id\": \"chatcmpl-1\"}");
    assert_eq!(
        header(put, "content-type").as_deref(),
        Some("application/json")
    );
    assert_eq!(
        header(put, "content-md5").as_deref(),
        Some("Ma6yKrRUR7fU10HikmP0wA==")
    );
    assert_eq!(
        header(put, "content-disposition").as_deref(),
        Some("inline; filename=\"time-2026-11-13T09-05-07-000042_chatcmpl-1.json\"")
    );
    assert_eq!(
        header(put, "x-amz-server-side-encryption").as_deref(),
        Some("aws:kms")
    );
    assert_eq!(
        header(put, "x-amz-date").as_deref(),
        Some("20251113T090507Z")
    );
    assert!(
        header(put, "authorization").unwrap().starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20251113/us-east-1/s3/aws4_request"
        )
    );
    assert!(sink.queued_keys().is_empty());
    assert_eq!(host.failures.load(Ordering::Relaxed), 0);
}

#[rstest]
#[tokio::test]
async fn a_bucket_that_fails_every_upload_keeps_the_queue() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({}));
    sink.submit(&event(&host, "a", "{}"));
    sink.submit(&event(&host, "b", "{}"));
    sink.flush().await;

    assert_eq!(server.received_requests().await.unwrap().len(), 6);
    assert_eq!(sink.queued_keys().len(), 2);
    assert_eq!(host.failures.load(Ordering::Relaxed), 2);
}

#[rstest]
#[tokio::test]
async fn a_terminal_rejection_is_dropped_when_another_upload_lands() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path_regex("_bad\\.json$"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_string("<Error><Code>EntityTooLarge</Code></Error>"),
        )
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .and(path_regex("_slow\\.json$"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({}));
    for id in ["ok", "bad", "slow"] {
        sink.submit(&event(&host, id, "{}"));
    }
    sink.flush().await;

    assert_eq!(
        sink.queued_keys(),
        ["base/2026-11-13/time-09-05-07-000042_slow.json"]
    );
    assert_eq!(host.held(), 1);
}

#[rstest]
#[tokio::test]
async fn batch_files_group_one_flush_by_folder() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({"s3_batch_file_upload": "yes"}));
    sink.submit(&event(&host, "a", "{\"n\": 1}"));
    sink.submit(&event(&host, "b", "{\"n\": 2}"));
    sink.flush().await;

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0]
            .url
            .path()
            .starts_with("/logs/base/2026-11-13/batch_09-05-07_")
    );
    assert_eq!(requests[0].body, b"{\"n\": 1}\n{\"n\": 2}");
    assert_eq!(
        header(&requests[0], "content-type").as_deref(),
        Some("application/x-ndjson")
    );
}

#[rstest]
#[tokio::test]
async fn an_event_python_would_reject_is_counted_and_skipped() {
    let server = MockServer::start().await;
    let (sink, host) = sink(&server, json!({"s3_use_key_prefix": true}));
    let mut bad = event(&host, "a", "{}");
    bad["key_alias"] = json!(true);
    sink.submit(&bad);
    sink.submit(&Value::Null);

    assert!(sink.queued_keys().is_empty());
    assert_eq!(host.failures(), 1);
    assert_eq!(host.held(), 0);
}

#[rstest]
#[tokio::test]
async fn the_body_is_encoded_at_flush_and_released_once_delivered() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({}));
    let payload = host.stash(Some("{\"n\": 1}"));
    sink.submit(&event_with(payload, "a"));
    host.rewrite(payload, "{\"n\": 2}");
    sink.flush().await;

    assert_eq!(
        server.received_requests().await.unwrap()[0].body,
        b"{\"n\": 2}"
    );
    assert_eq!(host.held(), 0);
}

#[rstest]
#[tokio::test]
async fn a_body_that_cannot_be_encoded_stays_queued_and_counts_a_failure() {
    let server = MockServer::start().await;
    let (sink, host) = sink(&server, json!({}));
    sink.submit(&event_with(host.stash(None), "a"));
    sink.flush().await;

    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(sink.queued_keys().len(), 1);
    assert_eq!(host.failures(), 1);
}

#[rstest]
#[case::environment(Ok(Some("ap-south-1".to_owned())), Some("/ap-south-1/"))]
#[case::default(Ok(None), Some("/us-west-2/"))]
#[case::invalid(Ok(Some("Ap_South".to_owned())), None)]
#[case::secret_failure(Err("secret manager down".to_owned()), None)]
#[tokio::test]
async fn without_a_region_param_the_host_resolves_the_signing_region(
    #[case] region: Result<Option<String>, String>,
    #[case] scope: Option<&str>,
) {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({"s3_region_name": null}));
    *host.region.lock().unwrap() = Some(region);
    sink.submit(&event(&host, "a", "{}"));
    sink.flush().await;

    let requests = server.received_requests().await.unwrap();
    let signed = requests
        .first()
        .and_then(|put| header(put, "authorization"))
        .map(|authorization| scope.is_some_and(|scope| authorization.contains(scope)));
    assert_eq!(signed, scope.map(|_| true));
    assert_eq!(sink.queued_keys().len(), usize::from(scope.is_none()));
}

#[rstest]
#[tokio::test]
async fn a_reply_slower_than_the_attempt_timeout_is_retried_then_kept() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(600)))
        .mount(&server)
        .await;
    let (sink, host) = sink_with(
        &server,
        json!({}),
        Constants {
            upload_attempt_timeout_seconds: 0.1,
            ..Constants::default()
        },
    );
    sink.submit(&event(&host, "a", "{}"));
    sink.flush().await;

    assert_eq!(server.received_requests().await.unwrap().len(), 3);
    assert_eq!(sink.queued_keys().len(), 1);
    assert_eq!(host.failures(), 1);
}

#[rstest]
#[tokio::test]
async fn a_panic_while_rendering_keeps_that_object_and_the_flush_goes_on() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({}));
    let racing = host.stash(Some(PANICS));
    sink.submit(&event_with(racing, "a"));
    sink.submit(&event(&host, "b", "{}"));
    sink.flush().await;

    let paths: Vec<String> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|put| put.url.path().to_owned())
        .collect();
    assert_eq!(paths, ["/logs/base/2026-11-13/time-09-05-07-000042_b.json"]);
    assert_eq!(
        sink.queued_keys(),
        ["base/2026-11-13/time-09-05-07-000042_a.json"]
    );
    assert_eq!(host.failures(), 1);

    host.rewrite(racing, "{}");
    sink.flush().await;
    assert!(sink.queued_keys().is_empty());
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[rstest]
#[tokio::test]
async fn a_panic_while_rendering_a_batch_file_keeps_the_whole_queue() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({"s3_batch_file_upload": true}));
    let racing = host.stash(Some(PANICS));
    sink.submit(&event_with(racing, "a"));
    sink.submit(&event(&host, "b", "{}"));
    sink.flush().await;

    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(sink.queued_keys().len(), 2);

    host.rewrite(racing, "{}");
    sink.flush().await;
    assert!(sink.queued_keys().is_empty());
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_panic_in_a_host_call_keeps_the_object_and_later_flushes_deliver_it() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, host) = sink(&server, json!({"s3_region_name": null}));
    host.region_panics.store(true, Ordering::Relaxed);
    sink.submit(&event(&host, "a", "{}"));
    sink.flush().await;

    assert!(server.received_requests().await.unwrap().is_empty());
    assert_eq!(sink.queued_keys().len(), 1);
    assert_eq!(host.failures(), 1);

    host.region_panics.store(false, Ordering::Relaxed);
    sink.flush().await;
    assert!(sink.queued_keys().is_empty());
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}
