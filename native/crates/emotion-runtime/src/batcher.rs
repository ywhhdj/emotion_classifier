//! 请求级批调度：满足 max_batch 或 max_wait_ms 即打包。
//!
//! v0.2.0 修复：
//!   * `std::sync::mpsc::Receiver` 没有 `is_empty()`（v0.1 无法编译）；
//!   * 超时后不再重复发送空批、不再把 deadline 悬空；
//!   * 发送端全部关闭时线程正常退出，不泄漏后台线程。

use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
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
        let max_batch = max_batch.max(1);

        std::thread::spawn(move || {
            let mut pending: Vec<BatchRequest> = Vec::new();
            let mut deadline: Option<Instant> = None;
            loop {
                let wait = deadline
                    .map(|d| d.saturating_duration_since(Instant::now()))
                    .unwrap_or_else(|| Duration::from_secs(3600));

                match req_rx.recv_timeout(wait) {
                    Ok(req) => {
                        pending.push(req);
                        if deadline.is_none() {
                            deadline = Some(Instant::now() + max_wait);
                        }
                        // 一次性收干已到达的请求，减少 syscall
                        while let Ok(extra) = req_rx.try_recv() {
                            pending.push(extra);
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        if !pending.is_empty() {
                            let _ = batch_tx.send(pending.drain(..).collect());
                        }
                        break;
                    }
                }

                if pending.is_empty() {
                    deadline = None;
                    continue;
                }
                let total: usize = pending.iter().map(|r| r.texts.len()).sum();
                let expired = deadline
                    .map(|d| Instant::now() >= d)
                    .unwrap_or(false);
                if total >= max_batch || expired {
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
