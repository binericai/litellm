use std::time::{Duration, SystemTime};

use bytes::Bytes;
use litellm_auth_aws::{Credentials, SigningScope, sign_s3_headers};
use reqwest::Method;

pub struct SignedRequest<'a> {
    pub method: Method,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub body: Bytes,
    pub region: &'a str,
    pub credentials: &'a Credentials,
    pub time: SystemTime,
    pub timeout: Option<Duration>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    pub status: u16,
    pub body: Bytes,
}

impl Reply {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SendError {
    #[error("the request cannot be signed: {0}")]
    Signing(#[from] litellm_auth_aws::Error),
    #[error("the request was not sent: {0}")]
    Request(String),
    #[error("{kind}: {message}")]
    Transport { kind: &'static str, message: String },
}

impl SendError {
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::Transport { .. })
    }
}

pub fn signature_headers(request: &SignedRequest<'_>) -> Result<Vec<(String, String)>, SendError> {
    Ok(sign_s3_headers(
        request.method.as_str(),
        request.url,
        &request.body,
        request.headers,
        SigningScope {
            region: request.region,
            service: "s3",
            credentials: request.credentials,
            time: request.time,
        },
    )?)
}

pub async fn send(
    client: &litellm_http::Client,
    request: SignedRequest<'_>,
) -> Result<Reply, SendError> {
    let signature = signature_headers(&request)?;
    let mut builder = client.request(request.method.clone(), request.url);
    if let Some(timeout) = request.timeout {
        builder = builder.timeout(timeout);
    }
    for (name, value) in request.headers.iter().chain(&signature) {
        builder = builder.header(name, value);
    }
    if !request.body.is_empty() || request.method == Method::PUT {
        builder = builder.body(request.body.clone());
    }
    let response = builder.send().await.map_err(transport)?;
    let status = response.status().as_u16();
    let body = response.bytes().await.map_err(transport)?;
    Ok(Reply { status, body })
}

fn transport(error: reqwest::Error) -> SendError {
    let message = error.to_string();
    if error.is_builder() || error.is_redirect() {
        return SendError::Request(message);
    }
    let kind = if error.is_timeout() {
        "Timeout"
    } else if error.is_connect() {
        "ConnectError"
    } else {
        "TransportError"
    };
    SendError::Transport { kind, message }
}
