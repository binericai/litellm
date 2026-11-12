use std::time::SystemTime;

use aws_credential_types::Credentials;
use reqwest::{Request, header::HeaderMap};

use crate::{
    AwsAuthService, AwsCredentialSource, Error, SigningScope, aws_signature_headers,
    constants::AWS4_AUTHORIZATION_PREFIX, is_sigv4_computed_header, sign_headers,
};
use litellm_http::outbound::RequestSigner;

#[derive(Clone, Debug)]
pub struct SigV4Signer {
    region: String,
    service: &'static str,
    credentials: Credentials,
    clock: fn() -> SystemTime,
}

impl SigV4Signer {
    pub fn new(region: String, service: &'static str, credentials: Credentials) -> Self {
        Self {
            region,
            service,
            credentials,
            clock: SystemTime::now,
        }
    }

    pub fn with_clock(self, clock: fn() -> SystemTime) -> Self {
        Self { clock, ..self }
    }

    pub async fn resolve(
        auth: &AwsAuthService,
        region: String,
        service: &'static str,
        credentials: AwsCredentialSource,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Self, Error> {
        let credentials = credentials
            .resolve(auth, &region, service, env_lookup)
            .await?;
        Ok(Self::new(region, service, credentials))
    }
}

const CALLER_HEADERS_THE_SIGNER_REPLACES: &[&str] = &["x-amz-date", "x-amz-security-token", "date"];

fn signable(headers: &HeaderMap) -> HeaderMap {
    aws_signature_headers(headers)
        .iter()
        .filter(|(name, _)| !is_sigv4_computed_header(name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn overrides_the_signature(headers: &HeaderMap) -> bool {
    headers
        .get(reqwest::header::AUTHORIZATION)
        .is_some_and(|value| {
            !value
                .as_bytes()
                .starts_with(AWS4_AUTHORIZATION_PREFIX.as_bytes())
        })
}

impl RequestSigner for SigV4Signer {
    fn owned_headers(&self) -> &'static [&'static str] {
        CALLER_HEADERS_THE_SIGNER_REPLACES
    }

    fn sign(&self, request: &Request) -> Result<HeaderMap, litellm_http::Error> {
        let body = match request.body() {
            None => &[],
            Some(body) => body
                .as_bytes()
                .ok_or_else(litellm_http::Error::unsignable_body)?,
        };
        let signed = sign_headers(
            request.method().as_str(),
            request.url().as_str(),
            body,
            &signable(request.headers()),
            SigningScope {
                region: &self.region,
                service: self.service,
                credentials: &self.credentials,
                time: (self.clock)(),
            },
        )
        .map_err(|error| match error {
            Error::NonAsciiHeader { .. } => litellm_http::Error::unsignable_header(error),
            Error::ExpiredCredentials
            | Error::Credentials { .. }
            | Error::WebIdentityTokenReference
            | Error::MissingWebIdentityToken
            | Error::WebIdentityToken(_)
            | Error::MissingWebIdentityCredentials
            | Error::WebIdentityExpiration(_)
            | Error::Signing { .. } => litellm_http::Error::signature(error),
        })?;
        Ok(match overrides_the_signature(request.headers()) {
            true => signed
                .iter()
                .filter(|(name, _)| *name != reqwest::header::AUTHORIZATION)
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            false => signed,
        })
    }
}

#[cfg(test)]
mod tests {
    use reqwest::{Body, Method, Url};
    use serde_json::{Value, json as json_value};
    use std::time::{Duration, UNIX_EPOCH};

    use litellm_http::outbound::{json, signed};

    use super::*;

    const URL: &str = "https://textract.us-east-1.amazonaws.com/";

    fn fixed_clock() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    fn signer(service: &'static str) -> SigV4Signer {
        SigV4Signer::new(
            "us-east-1".into(),
            service,
            Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
        )
        .with_clock(fixed_clock)
    }

    fn target() -> (String, String) {
        ("X-Amz-Target".into(), "Textract.DetectDocumentText".into())
    }

    fn authorization(body: &Value, service: &'static str) -> String {
        signed(
            json(URL, &[target()], body, None).unwrap(),
            &signer(service),
        )
        .unwrap()
        .headers()["authorization"]
            .to_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn the_signature_verifies_against_the_bytes_that_are_sent() {
        let sent = signed(
            json(
                URL,
                &[target()],
                &json_value!({"Document": {"Bytes": "aGk="}}),
                None,
            )
            .unwrap(),
            &signer("textract"),
        )
        .unwrap();
        let unsigned: HeaderMap = sent
            .headers()
            .iter()
            .filter(|(name, _)| !is_sigv4_computed_header(name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let recomputed = sign_headers(
            "POST",
            sent.url().as_str(),
            sent.body().unwrap().as_bytes().unwrap(),
            &aws_signature_headers(&unsigned),
            SigningScope {
                region: "us-east-1",
                service: "textract",
                credentials: &Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
                time: fixed_clock(),
            },
        )
        .unwrap();

        assert_eq!(sent.headers()["authorization"], recomputed["authorization"]);
    }

    #[test]
    fn the_signature_depends_on_the_body_and_the_service() {
        let original = authorization(&json_value!({"text": "card 4111"}), "textract");

        assert_ne!(
            original,
            authorization(&json_value!({"text": "card [REDACTED]"}), "textract")
        );
        assert_ne!(
            original,
            authorization(&json_value!({"text": "card 4111"}), "bedrock")
        );
        assert!(original.contains("/us-east-1/textract/aws4_request"));
    }

    fn sent(headers: &[(&str, &str)], credentials: Credentials) -> Request {
        signed(
            json(
                URL,
                &headers
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect::<Vec<_>>(),
                &json_value!({}),
                None,
            )
            .unwrap(),
            &SigV4Signer::new("us-east-1".into(), "bedrock", credentials).with_clock(fixed_clock),
        )
        .unwrap()
    }

    fn header<'a>(request: &'a Request, name: &str) -> Vec<&'a str> {
        request
            .headers()
            .get_all(name)
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect()
    }

    #[rstest::rstest]
    #[case::caller_date("date", None)]
    #[case::caller_amz_date("x-amz-date", Some("20231114T221320Z"))]
    #[case::caller_security_token_without_a_session("x-amz-security-token", None)]
    fn a_header_the_signer_computes_is_dropped_from_the_callers_copy(
        #[case] name: &str,
        #[case] expected: Option<&str>,
    ) {
        let request = sent(
            &[(name, "caller-value")],
            Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
        );

        assert_eq!(
            header(&request, name),
            expected.into_iter().collect::<Vec<_>>()
        );
        assert!(header(&request, "authorization")[0].starts_with("AWS4-HMAC-SHA256 "));
    }

    #[test]
    fn a_session_token_replaces_the_callers_security_token() {
        let request = sent(
            &[("X-Amz-Security-Token", "caller-value")],
            Credentials::new(
                "AKIDEXAMPLE",
                "secret",
                Some("session".into()),
                None,
                "test",
            ),
        );

        assert_eq!(header(&request, "x-amz-security-token"), ["session"]);
    }

    #[rstest::rstest]
    #[case::bearer_overrides_the_signature("Bearer gateway", "Bearer gateway")]
    #[case::a_stale_signature_is_replaced(
        "AWS4-HMAC-SHA256 Credential=stale",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20231114/us-east-1/bedrock/aws4_request, SignedHeaders=content-type;host;x-amz-date"
    )]
    fn a_callers_authorization_overrides_the_signature_unless_it_is_sigv4(
        #[case] caller: &str,
        #[case] expected: &str,
    ) {
        let request = sent(
            &[("Authorization", caller)],
            Credentials::new("AKIDEXAMPLE", "secret", None, None, "test"),
        );
        let authorization = header(&request, "authorization");

        assert_eq!(authorization.len(), 1);
        assert_eq!(
            authorization[0].split(", Signature=").next(),
            Some(expected)
        );
        assert_eq!(header(&request, "x-amz-date"), ["20231114T221320Z"]);
    }

    #[test]
    fn a_forwarded_header_that_cannot_be_signed_is_the_callers_fault() {
        let error = signed(
            json(
                URL,
                &[("x-amz-meta-note".into(), "café".into())],
                &json_value!({}),
                None,
            )
            .unwrap(),
            &signer("textract"),
        )
        .unwrap_err();

        let report = litellm_error::Report::new(&error).to_string();
        assert!(report.contains("x-amz-meta-note"), "{report}");
        assert!(error.is_request());
        assert!(!error.dispatched());
    }

    #[test]
    fn a_streaming_body_cannot_be_signed() {
        let mut request = Request::new(Method::POST, Url::parse(URL).unwrap());
        *request.body_mut() = Some(Body::wrap_stream(futures_util::stream::once(async {
            Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"{}"))
        })));

        assert!(matches!(
            signed(request, &signer("textract")),
            Err(error) if !error.dispatched() && !error.is_request()
        ));
    }
}
