//! Facade re-export proofs: everything reachable through `ai_sdk::*` must
//! resolve, construct offline, and behave. No engine, network, or provider
//! credentials are involved — these are the SDK's public surface guarantees.

use std::time::Duration;

use ai_sdk::prelude::*;
use ai_sdk::*;

#[test]
fn cosine_similarity_reexport_matches_unit_vectors() {
    let identical = cosine_similarity(&[0.5, 0.5], &[1.0, 1.0]).expect("non-zero vectors");
    assert!((identical - 1.0).abs() < 1e-6);

    let orthogonal = cosine_similarity(&[1.0, 0.0], &[0.0, 2.0]).expect("non-zero vectors");
    assert!(orthogonal.abs() < 1e-6);

    // Degenerate inputs are rejected with None rather than NaN.
    assert!(cosine_similarity(&[], &[1.0]).is_none());
    assert!(cosine_similarity(&[0.0], &[1.0]).is_none());
}

#[test]
fn ttl_cache_reexport_round_trips() {
    let cache = TtlCache::new(Duration::from_secs(60), 8);
    assert_eq!(cache.get("missing"), None);
    cache.set("k", serde_json::json!({ "v": 42 }));
    assert_eq!(cache.get("k"), Some(serde_json::json!({ "v": 42 })));
}

#[test]
fn reciprocal_rank_fusion_reexport_scores_known_rankings() {
    let mut ranked = reciprocal_rank_fusion(
        &[vec!["a".into(), "b".into()], vec!["b".into(), "c".into()]],
        60.0,
    );
    ranked.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    let ids: Vec<&str> = ranked.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, vec!["b", "a", "c"]);
    let score_of = |id: &str| ranked.iter().find(|(k, _)| k == id).unwrap().1;
    assert!((score_of("b") - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-6);
}

#[tokio::test]
async fn collect_text_reexport_joins_text_deltas_and_surfaces_errors() {
    let ok_events: Vec<Result<StreamEvent, AiError>> = vec![
        Ok(StreamEvent::TextDelta {
            delta: "hel".into(),
        }),
        Ok(StreamEvent::TextDelta { delta: "lo".into() }),
    ];
    let text = collect_text(Box::pin(futures::stream::iter(ok_events)))
        .await
        .expect("text deltas collect");
    assert_eq!(text, "hello");

    let err_events: Vec<Result<StreamEvent, AiError>> = vec![Err(AiError::Serialization(
        ai_errors::SerializationError::new("boom".to_string()),
    ))];
    assert!(
        collect_text(Box::pin(futures::stream::iter(err_events)))
            .await
            .is_err()
    );
}

#[test]
fn security_reexports_redact_detect_and_gate_urls() {
    let redactor = Redactor::new(vec!["hunter2-secret-value".into()]);
    let redacted = redactor.redact("password is hunter2-secret-value ok");
    assert!(!redacted.contains("hunter2-secret-value"));

    let pii = PiiDetector::new();
    assert!(
        !pii.find("reach me at person@example.com please").is_empty(),
        "email must be detected"
    );

    // Default policy allows public http(s) and rejects private-range hosts.
    assert!(matches!(
        UrlPolicy::new().validate("https://example.com/x"),
        ai_security::UrlVerdict::Allowed
    ));
    assert!(matches!(
        UrlPolicy::new().validate("http://127.0.0.1/x"),
        ai_security::UrlVerdict::Rejected(_)
    ));
}

#[tokio::test]
async fn default_tools_registry_serves_the_calculator_end_to_end() {
    let registry = default_tools();
    assert!(
        registry.len() >= 4,
        "http/time/calculator/uuid ship by default"
    );
    let math = registry
        .require("calculator")
        .expect("calculator registered");
    let context = ToolContext {
        permissions: Permissions::new(),
        execution_id: Some("facade-proof".into()),
        deadline: None,
        max_response_bytes: None,
    };
    let output = run_tool(
        math.as_ref(),
        serde_json::json!({ "expression": "6 * 7" }),
        &context,
    )
    .await
    .expect("calculator evaluates");
    assert!(format!("{output:?}").contains("42"));
}

#[test]
fn model_registry_default_catalog_is_populated() {
    let catalog = default_catalog();
    assert!(!catalog.is_empty(), "default catalog ships models");
    assert!(
        !catalog.for_provider("openai").is_empty(),
        "openai entries present in the default catalog"
    );
}

#[test]
fn computer_transport_reexport_constructs_without_dialing() {
    // Construction only: proves the re-export resolves and stays offline.
    let client = JsonRpcHttpClient::new("http://127.0.0.1:9/rpc", None);
    drop(client);
}

#[test]
fn edge_runtime_detection_reexport_reports_a_runtime() {
    let text = format!("{:?}", detect_runtime());
    assert!(!text.is_empty());
}

#[test]
fn prelude_imports_compile_and_roles_exist() {
    let _agent_builder_ready: Option<AgentBuilder> = None;
    assert_ne!(Role::User, Role::Assistant);
}
