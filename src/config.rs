use serde::Deserialize;

// ==========================================
// 配置结构体定义
// ==========================================
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub camera: CameraConfig,
    pub window: WindowConfig,
    pub algorithm: AlgorithmConfigs,
    pub performance: PerformanceConfig,

    // 【新增】资源配置
    pub assets: AssetsConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CameraConfig {
    pub index: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WindowConfig {
    pub title: String,
    pub scale: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PerformanceConfig {
    pub active_fps: i32,
    pub idle_fps: i32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AlgorithmConfigs {
    pub hand: AlgorithmParams,
    pub face: AlgorithmParams,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AlgorithmParams {
    pub stability: f32,
    pub threshold: f32,
    pub overlap: Option<f32>,
}

// 【新增】资源路径配置
#[derive(Debug, Deserialize, Clone)]
pub struct AssetsConfig {
    pub avatar: String,
    pub scale: f32,
}
