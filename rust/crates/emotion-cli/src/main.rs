//! 内核自检 CLI：不依赖 Python，直接验证权重、分词与前向。
//!
//!   emotion-cli check  --model-dir ./models          # 加载 + warmup + 自检
//!   emotion-cli bench  --model-dir ./models -n 10000 # 延迟 p50/p99 与吞吐
//!   emotion-cli run    --model-dir ./models --text "今天心情真好"

use anyhow::Result;
use emotion_core::{detect_lang, EngineConfig};
use emotion_postprocess::{probs, top_k_thresholded};
use emotion_runtime::Engine;
use std::time::Instant;

fn load_cfg(model_dir: &str) -> EngineConfig {
    let mut cfg = EngineConfig::default();
    cfg.model_dir = std::path::PathBuf::from(model_dir);
    cfg
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("check");
    let model_dir = arg(&args, "--model-dir").unwrap_or_else(|| "models".to_string());

    let engine = Engine::load(load_cfg(&model_dir))?;
    println!("[cli] 模型就绪: labels={} hidden={}",
             engine.num_labels(), engine.hidden_size());

    match cmd {
        "check" => {
            let samples = vec![
                "脸颊泛红，偷偷瞄了你一眼".to_string(),
                "i'm so excited i can hardly breathe".to_string(),
                "怖くて震えが止まらない".to_string(),
            ];
            let langs: Vec<String> = samples.iter().map(|t| detect_lang(t).to_string()).collect();
            let logits = engine.logits(&samples, &langs)?;
            for (t, row) in samples.iter().zip(logits.iter()) {
                let p = probs(row, None);
                let top = top_k_thresholded(&p, 3, None);
                println!("[check] 「{t}」 -> {:?}", top);
            }
        }
        "bench" => {
            let n: usize = arg(&args, "-n")
                .or_else(|| arg(&args, "--n"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(10_000);
            let text = "今天心情真好，非常开心".to_string();
            let texts: Vec<String> = (0..n).map(|_| text.clone()).collect();
            let langs: Vec<String> = (0..n).map(|_| "zh".to_string()).collect();

            // warmup
            let _ = engine.logits(&texts[..32.min(n)], &langs[..32.min(n)])?;

            let t0 = Instant::now();
            let _ = engine.logits(&texts, &langs)?;
            let elapsed = t0.elapsed();
            let per = elapsed.as_secs_f64() * 1000.0 / n as f64;
            println!("[bench] n={n} total={elapsed:?} 单句均值={per:.3} ms 吞吐={:.1} 句/秒",
                     n as f64 / elapsed.as_secs_f64());
        }
        "run" => {
            let text = arg(&args, "--text").unwrap_or_else(|| "今天的天气真好".to_string());
            let texts = vec![text.clone()];
            let langs = vec![detect_lang(&text).to_string()];
            let logits = engine.logits(&texts, &langs)?;
            let p = probs(&logits[0], None);
            for (id, s) in top_k_thresholded(&p, 5, None) {
                println!("[run] #{id} {s:.4}");
            }
        }
        other => anyhow::bail!("未知命令: {other}（可用: check / bench / run）"),
    }
    Ok(())
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}
