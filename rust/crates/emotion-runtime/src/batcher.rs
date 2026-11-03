//! 请求级批调度：满足 max_batch 或 max_wait_ms 即打包。
//!
//! 服务端形态下由 axum handler 往 `Batcher` 投递请求；
//! Python 直连时走同步 `logits()`，批调度即同步分块。

use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

pub struct BatchRequest {
    pub texts: Vec<String>,
    pub langs: Vec<String>,
    pub resp: Sender<Vec<Vec<f32>>>,
}

pub struct Batcher {
    tx: Sender<BatchRequest>,
}

impl Batcher {
    pub fn new(max_batch: usize, max_wait: Duration) -> (Self, Receiver<Vec<BatchRequest>>) {
        let (req_tx, req_rx) = channel::<BatchRequest>();
        let (batch_tx, batch_rx) = channel::<Vec<BatchRequest>>();

        std::thread::spawn(move || {
            let mut pending: Vec<BatchRequest> = Vec::new();
            let mut deadline: Option<Instant> = None;
            loop {
                let timeout = deadline
                    .map(|d| d.saturating_duration_since(Instant::now()))
                    .unwrap_or(Duration::from_secs(3600));

                match req_rx.recv_timeout(timeout) {
                    Ok(req) => {
                        pending.push(req);
                        if deadline.is_none() {
                            deadline = Some(Instant::now() + max_wait);
                        }
                    }
                    Err(_) => {
                        if pending.is_empty() {
                            deadline = None;
                            continue;
                        }
                    }
                }

                let total: usize = pending.iter().map(|r| r.texts.len()).sum();
                let expired = deadline
                    .map(|d| Instant::now() >= d)
                    .unwrap_or(false);
                if total >= max_batch || expired || (!pending.is_empty() && req_rx.is_empty() && expired) {
                    let batch: Vec<BatchRequest> = pending.drain(..).collect();
                    if batch_tx.send(batch).is_err() {
                        break;
                    }
                    deadline = None;
                }
            }
        });

        (Self { tx: req_tx }, batch_rx)
    }

    pub fn submit(&self, texts: Vec<String>, langs: Vec<String>) -> Receiver<Vec<Vec<f32>>> {
        let (resp_tx, resp_rx) = channel();
        let _ = self.tx.send(BatchRequest { texts, langs, resp: resp_tx });
        resp_rx
    }
}
