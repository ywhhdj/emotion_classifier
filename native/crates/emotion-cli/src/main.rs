//! 内核自检 CLI：不依赖 Python，直接验证权重、分词与前向。
//!
//!   emotion-cli check  --model-dir ./models --precision int8
//!   emotion-cli bench  --model-dir ./models -n 10000
//!   emotion-cli run    --model-dir ./models --text "今天心情真好"

use anyhow::Result;
use emotion_core::{detect_lang, BackendKind, EngineConfig, Precision};
use emotion_postprocess::{probs, top_k_thresholded};
use emotion_runtime::global_engine;
use std::sync::Arc;
use std::time::Instant;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

fn parse_precision(s: Option<String>) -> Precision {
    match s.as_deref().unwrap_or("fp32").to_lowercase().as_str() {
        "fp16" | "float16" => Precision::Fp16,
        "int8" | "quant" => Precision::Int8,
        _ => Precision::Fp32,
    }
}

fn load_cfg(args: &[String]) -> EngineConfig {
    let model_dir = arg(args, "--model-dir").unwrap_or_else(|| "models".to_string());
    let precision = parse_precision(arg(args, "--precision"));
    let backend = if arg(args, "--backend").as_deref() == Some("candle") {
        BackendKind::Candle
    } else {
        BackendKind::Onnx
    };
    let max_seq_len = arg(args, "--max-seq-len")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let use_lang_token = arg(args, "--lang-token").is_some();
    EngineConfig {
        model_dir: std::path::PathBuf::from(model_dir),
        precision,
        backend,
        max_seq_len,
        use_lang_token,
        model_path: arg(args, "--model-path").map(std::path::PathBuf::from),
        tokenizer_dir: arg(args, "--tokenizer-dir").map(std::path::PathBuf::from),
        ..EngineConfig::default()
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("check");
    let cfg = load_cfg(&args);
    println!("[cli] v{} 精度={:?} 后端={:?} max_seq_len={}",
             emotion_core::VERSION, cfg.precision, cfg.backend, cfg.max_seq_len);

    // 进程内唯一实例：重复初始化会被拒绝，避免多份权重常驻
    let engine: Arc<_> = global_engine(cfg)?;
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
            let warm = 32usize.min(n);
            let _ = engine.logits(&texts[..warm], &langs[..warm])?;

            let t0 = Instant::now();
            let _ = engine.logits(&texts, &langs)?;
            let elapsed = t0.elapsed();
            let per = elapsed.as_secs_f64() * 1000.0 / n.max(1) as f64;
            println!("[bench] n={n} total={elapsed:?} 单句均值={per:.3} ms 吞吐={:.1} 句/秒",
                     n as f64 / elapsed.as_secs_f64().max(1e-9));
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
