use std::fmt::Write;

use crate::{
    pyfmt::float_to_go_string,
    store::{FamilySnapshot, Kind},
};

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub name: String,
    pub labels: Vec<(String, String)>,
    pub value: f64,
}

fn labelled(names: &[String], values: &[String]) -> Vec<(String, String)> {
    names.iter().cloned().zip(values.iter().cloned()).collect()
}

pub fn samples(family: &FamilySnapshot, created: bool) -> Vec<Sample> {
    let name = &family.name;
    family
        .series
        .iter()
        .flat_map(|(values, series)| {
            let labels = labelled(&family.labelnames, values);
            let sample = |suffix: &str, labels: Vec<(String, String)>, value: f64| Sample {
                name: format!("{name}{suffix}"),
                labels,
                value,
            };
            let mut out = Vec::new();
            match family.kind {
                Kind::Counter => out.push(sample("_total", labels.clone(), series.value)),
                Kind::Gauge => out.push(sample("", labels.clone(), series.value)),
                Kind::Histogram => {
                    let mut acc = 0.0;
                    for (bound, count) in family.buckets.iter().zip(&series.buckets) {
                        acc += count;
                        let mut bucket = labels.clone();
                        bucket.push(("le".into(), float_to_go_string(*bound)));
                        out.push(sample("_bucket", bucket, acc));
                    }
                    out.push(sample("_count", labels.clone(), acc));
                    if family.buckets.first().is_some_and(|first| *first >= 0.0) {
                        out.push(sample("_sum", labels.clone(), series.sum));
                    }
                }
            }
            if created && family.kind != Kind::Gauge {
                out.push(sample("_created", labels, series.created));
            }
            out
        })
        .collect()
}

fn escape_value(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('\n', r"\n")
        .replace('"', r#"\""#)
}

fn escape_help(help: &str) -> String {
    help.replace('\\', r"\\").replace('\n', r"\n")
}

fn sample_line(out: &mut String, sample: &Sample) {
    out.push_str(&sample.name);
    if !sample.labels.is_empty() {
        let mut labels = sample.labels.clone();
        labels.sort_by(|left, right| left.0.cmp(&right.0));
        out.push('{');
        let rendered: Vec<String> = labels
            .iter()
            .map(|(key, value)| format!("{key}=\"{}\"", escape_value(value)))
            .collect();
        out.push_str(&rendered.join(","));
        out.push('}');
    }
    let _ = writeln!(out, " {}", float_to_go_string(sample.value));
}

pub fn encode_text(families: &[FamilySnapshot], created: bool) -> String {
    let mut out = String::new();
    for family in families {
        let exposed = match family.kind {
            Kind::Counter => format!("{}_total", family.name),
            Kind::Gauge | Kind::Histogram => family.name.clone(),
        };
        let help = escape_help(&family.documentation);
        let _ = writeln!(out, "# HELP {exposed} {help}");
        let _ = writeln!(out, "# TYPE {exposed} {}", family.kind.as_str());
        let created_name = format!("{}_created", family.name);
        let (late, early): (Vec<Sample>, Vec<Sample>) = samples(family, created)
            .into_iter()
            .partition(|sample| sample.name == created_name);
        for sample in &early {
            sample_line(&mut out, sample);
        }
        if !late.is_empty() {
            let _ = writeln!(out, "# HELP {created_name} {help}");
            let _ = writeln!(out, "# TYPE {created_name} gauge");
            for sample in &late {
                sample_line(&mut out, sample);
            }
        }
    }
    out
}
