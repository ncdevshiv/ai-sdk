//! Temporary diagnostic: dump the raw wire response for one provider's
//! /models call so transport-level rejections can be root-caused.

use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let base = args.next().expect("base url");
    let key = args.next().expect("key");
    let t = ai_discovery::probe::Transport::with_policy(
        base.clone(),
        key,
        Duration::from_secs(30),
        ai_discovery::probe::TransportPolicy::none(),
    )?;
    println!("transport: {t:?}");
    let raw = t.get_once("models").await;
    println!("status          : {}", raw.status);
    println!("transport_error : {:?}", raw.transport_error);
    println!("retry_after     : {:?}", raw.retry_after);
    println!("body (first 800): {}", &raw.body[..raw.body.len().min(800)]);
    Ok(())
}
