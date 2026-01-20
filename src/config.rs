// src/config.rs
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
    pub assets: AssetsConfig,
    pub mouse: MouseConfig,
    pub shortcuts: ShortcutsConfig,
    pub gesture: GestureConfig,
    pub debug: DebugConfig,
    pub inference: InferenceConfig,
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
    pub enabled: Option<bool>,
    pub stability: f32,
    pub threshold: f32,
    pub overlap: Option<f32>,
}

// 【新增】锚点位置枚举
#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
pub enum AnchorPoint {
    LU, // 左上 (Left-Up)
    LD, // 左下 (Left-Down)
    U,  // 上 (Up)
    D,  // 下 (Down)
    L,  // 左 (Left)
    R,  // 右 (Right)
    C,  // 居中 (Center)
    RU, // 右上 (Right-Up)
    RD, // 右下 (Right-Down)
}

// 为枚举提供默认值
impl Default for AnchorPoint {
    fn default() -> Self {
        Self::C
    }
}

// 资源路径配置
#[derive(Debug, Deserialize, Clone)]
pub struct AssetsConfig {
    pub avatar: String,
    pub scale: f32,
    pub cursor_normal: String,
    pub cursor_scale_normal: f32,
    pub cursor_scroll: String,
    pub cursor_scale_scroll: f32,
    pub cursor_text: String,
    pub cursor_scale_text: f32,
    pub rotation_speed: f32,

    // 【新增】配置锚点，默认为居中
    #[serde(default)]
    pub anchor: AnchorPoint,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MouseTrackMode {
    Head,
    Hand,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MouseConfig {
    pub enabled: bool,
    pub track_mode: MouseTrackMode,
    #[serde(default)]
    pub virtual_cursor_mode: bool,
    pub auto_screen_size: bool,
    pub manual_width: f64,
    pub manual_height: f64,
    pub sensitivity_x: f32,
    pub sensitivity_y: f32,
    pub smoothing: f32,
    pub dead_zone: f32,
    pub invert_x: bool,
    pub invert_y: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ShortcutsConfig {
    pub reset_center: Vec<String>,
    pub toggle_mouse: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GestureConfig {
    pub pinch_threshold_on: f32,
    pub pinch_threshold_off: f32,
    pub rotation_threshold_degrees: f32,
    pub rotation_cooldown_ms: u128,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DebugConfig {
    #[allow(dead_code)]
    pub show_debug_window: bool,
    pub window_always_on_top: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct InferenceConfig {
    pub device: String,
    pub cpu_threads: usize,
}
