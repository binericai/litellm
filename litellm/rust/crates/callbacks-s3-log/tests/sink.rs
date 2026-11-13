use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

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
}

impl Host for Recorder {
    fn callback_failed(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    fn cold_storage_is_s3_v2(&self) -> bool {
        false
    }
}

fn fixed_clock() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_763_024_707)
}

fn no_env(_: &str) -> Option<String> {
    None
}

fn sink(server: &MockServer, extra: Value) -> (Arc<S3Log>, Arc<Recorder>) {
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
    let host = Arc::new(Recorder::default());
    let sink = S3Log::new(S3LogParts {
        settings: Settings::from_params(params.as_object().unwrap(), &Constants::default()),
        client: litellm_http::Client::for_test(),
        environment: Arc::new(no_env),
        clock: fixed_clock,
        backoff: Arc::new(|_| Duration::ZERO),
        host: Arc::clone(&host) as Arc<dyn Host>,
    });
    (sink, host)
}

fn event(id: &str, body: &str) -> Value {
    json!({
        "body": body,
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
    sink.submit(&event("chatcmpl-1", "{\"id\": \"chatcmpl-1\"}"));
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
    sink.submit(&event("a", "{}"));
    sink.submit(&event("b", "{}"));
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
    let (sink, _) = sink(&server, json!({}));
    for id in ["ok", "bad", "slow"] {
        sink.submit(&event(id, "{}"));
    }
    sink.flush().await;

    assert_eq!(
        sink.queued_keys(),
        ["base/2026-11-13/time-09-05-07-000042_slow.json"]
    );
}

#[rstest]
#[tokio::test]
async fn batch_files_group_one_flush_by_folder() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let (sink, _) = sink(&server, json!({"s3_batch_file_upload": "yes"}));
    sink.submit(&event("a", "{\"n\": 1}"));
    sink.submit(&event("b", "{\"n\": 2}"));
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
    let mut bad = event("a", "{}");
    bad["key_alias"] = json!(true);
    sink.submit(&bad);
    sink.submit(&Value::Null);

    assert!(sink.queued_keys().is_empty());
    assert_eq!(host.failures.load(Ordering::Relaxed), 1);
}
