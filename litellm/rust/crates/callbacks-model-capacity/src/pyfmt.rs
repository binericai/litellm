use serde_json::{Number, Value};

pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let formatted = format!("{value:e}");
        let (mantissa, exponent) = formatted.split_once('e').unwrap_or((&formatted, "0"));
        let (sign, digits) = exponent
            .strip_prefix('-')
            .map_or(("+", exponent), |digits| ("-", digits));
        return format!("{mantissa}e{sign}{digits:0>2}");
    }
    let formatted = format!("{value}");
    if formatted.contains('.') {
        formatted
    } else {
        format!("{formatted}.0")
    }
}

pub fn number(value: &Number) -> String {
    if value.is_i64() || value.is_u64() {
        return value.to_string();
    }
    value.as_f64().map_or_else(|| value.to_string(), float_repr)
}

pub fn value(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".into(),
        Some(Value::Bool(true)) => "True".into(),
        Some(Value::Bool(false)) => "False".into(),
        Some(Value::Number(number)) => self::number(number),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

pub fn optional_int(value: Option<i64>) -> String {
    value.map_or_else(|| "None".into(), |value| value.to_string())
}

pub fn percent(value: f64, places: usize) -> String {
    let scaled = value * 100.0;
    if scaled.is_nan() {
        return "nan%".into();
    }
    if scaled.is_infinite() {
        return if scaled > 0.0 { "inf%" } else { "-inf%" }.into();
    }
    format!("{scaled:.places$}%")
}

pub fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|number| number != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(map)) => !map.is_empty(),
        Some(Value::Bool(true)) => true,
    }
}

pub fn is_latin1(text: &str) -> bool {
    text.chars().all(|character| (character as u32) < 256)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(2.0, "2.0")]
    #[case(1.5, "1.5")]
    #[case(1_000_000.0, "1000000.0")]
    #[case(0.1, "0.1")]
    #[case(1e-5, "1e-05")]
    #[case(1e16, "1e+16")]
    #[case(-3.25, "-3.25")]
    fn floats_render_like_python_repr(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(float_repr(value), expected);
    }

    #[rstest]
    #[case(0.8, 1, "80.0%")]
    #[case(0.123456, 2, "12.35%")]
    #[case(1.0 / 3.0, 1, "33.3%")]
    #[case(0.0, 2, "0.00%")]
    #[case(1.25, 1, "125.0%")]
    fn percents_render_like_python_format(
        #[case] value: f64,
        #[case] places: usize,
        #[case] expected: &str,
    ) {
        assert_eq!(percent(value, places), expected);
    }
}
