//! 后处理：温度缩放 -> softmax -> per-class 阈值 -> top-k。
//!
//! 放在 Rust 侧完成，避免把 N×18 的 float 矩阵搬回 Python 再做向量运算。

use emotion_core::{Calibration, Scored};

pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in logits.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// 温度缩放 + softmax，返回概率向量。
pub fn probs(logits: &[f32], calib: Option<&Calibration>) -> Vec<f32> {
    let t = calib.map(|c| c.temperature()).unwrap_or(1.0);
    let mut v: Vec<f32> = if (t - 1.0).abs() > 1e-6 {
        logits.iter().map(|x| x / t).collect()
    } else {
        logits.to_vec()
    };
    softmax(&mut v);
    v
}

/// top-k（按分数降序）。
pub fn top_k(p: &[f32], k: usize) -> Vec<Scored> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter()
        .take(k.max(1).min(p.len()))
        .map(|i| (i as u16, p[i]))
        .collect()
}

/// 阈值过滤后取 top-k（per-class 阈值来自校准）。
pub fn top_k_thresholded(p: &[f32], k: usize, thresholds: Option<&[f32]>) -> Vec<Scored> {
    let mut scored: Vec<Scored> = match thresholds {
        Some(th) => p
            .iter()
            .enumerate()
            .filter(|(i, _)| th.get(*i).copied().unwrap_or(0.5) <= p[*i])
            .map(|(i, &s)| (i as u16, s))
            .collect(),
        None => p.iter().enumerate().map(|(i, &s)| (i as u16, s)).collect(),
    };
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k.max(1));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, 3.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn topk_ordering() {
        let p = vec![0.1f32, 0.7, 0.2];
        let t = top_k(&p, 2);
        assert_eq!(t[0].0, 1);
        assert_eq!(t[1].0, 2);
    }
}
