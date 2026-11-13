use std::cell::RefCell;

use litellm_callbacks::CallbackError;
use serde_json::{Map, Value};

use crate::{
    emit::{Emitter, failed},
    labels::{LabelValues, combined_custom_metadata, custom_labels_from_metadata, is_py_dict},
    pyfmt::py_str,
    success::{
        AttemptFacts, DeploymentFailure, LogLine, Success, exception_text, get, required,
        set_deployment_state, skips_invalid_key,
    },
};

pub struct Failure<'a> {
    pub emit: Emitter<'a>,
    pub facts: &'a AttemptFacts,
    pub logs: &'a RefCell<Vec<LogLine>>,
}

impl Failure<'_> {
    pub fn run(&self) -> Result<(), CallbackError> {
        let Some(payload) = is_py_dict(&self.facts.payload) else {
            return Err(failed(format!(
                "'{}' object is not subscriptable",
                py_str(&self.facts.payload)
            )));
        };
        if skips_invalid_key(self.facts.exception.as_ref()) {
            return Ok(());
        }
        let metadata = required(payload, "metadata")?;
        let metadata =
            is_py_dict(metadata).ok_or_else(|| failed("metadata is not subscriptable"))?;
        let user_id = required(metadata, "user_api_key_user_id")?;
        let hashed_api_key = required(metadata, "user_api_key_hash")?;
        let api_key_alias = required(metadata, "user_api_key_alias")?;
        let team = required(metadata, "user_api_key_team_id")?;
        let team_alias = required(metadata, "user_api_key_team_alias")?;
        let block = (|| -> Result<(), CallbackError> {
            let combined = combined_custom_metadata(payload);
            let values = LabelValues {
                custom_metadata_labels: custom_labels_from_metadata(
                    &combined,
                    &self.facts.settings.custom_metadata_labels,
                ),
                ..LabelValues::default()
            }
            .with("end_user", self.facts.end_user_id.clone())
            .with("hashed_api_key", hashed_api_key.clone())
            .with("api_key_alias", api_key_alias.clone())
            .with("model", self.facts.model.clone())
            .with("team", team.clone())
            .with("team_alias", team_alias.clone())
            .with("user", user_id.clone())
            .with(
                "model_id",
                payload.get("model_id").cloned().unwrap_or(Value::from("")),
            )
            .with("requested_model", get(payload, "model_group").clone())
            .with("api_provider", get(payload, "custom_llm_provider").clone());
            let context = values.context(&self.facts.settings);
            self.emit
                .inc("litellm_llm_api_failed_requests_metric", &context, 1.0)?;
            let success = Success {
                emit: Emitter {
                    store: self.emit.store,
                    settings: self.emit.settings,
                },
                facts: self.facts,
                owns_latency: false,
                logs: self.logs,
            };
            success.zero_cost(payload, &values)?;
            if let Some(deployment) = &self.facts.deployment_failure
                && let Err(error) = self.deployment(deployment)
            {
                success.log(LogLine::new(
                    "debug",
                    "Prometheus Error: set_llm_deployment_failure_metrics. Exception occured - %s",
                    vec![exception_text(&error)],
                ));
            }
            success.deferred()
        })();
        if let Err(error) = block {
            self.logs.borrow_mut().push(LogLine::new(
                "exception",
                "prometheus Layer Error(): Exception occured - %s",
                vec![exception_text(&error)],
            ));
        }
        Ok(())
    }

    fn deployment(&self, deployment: &DeploymentFailure) -> Result<(), CallbackError> {
        if deployment.selected {
            set_deployment_state(
                &self.emit,
                1.0,
                &deployment.litellm_model_name,
                &deployment.model_id,
                &deployment.api_base,
                &deployment.provider,
            )?;
        }
        let values = is_py_dict(&deployment.values)
            .map_or_else(LabelValues::default, |map: &Map<String, Value>| {
                LabelValues::from_enum(map)
            });
        let context = values.context(&self.facts.settings);
        if deployment.has_exception {
            self.emit
                .inc("litellm_deployment_failure_responses", &context, 1.0)?;
        }
        self.emit
            .inc("litellm_deployment_total_requests", &context, 1.0)
    }
}
