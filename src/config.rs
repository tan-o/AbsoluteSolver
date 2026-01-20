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

    pub mouse: MouseConfig,
    // 【新增】快捷键配置
    pub shortcuts: ShortcutsConfig,
    // 【新增】手势配置
    pub gesture: GestureConfig,
    // 【新增】调试配置
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
    pub enabled: Option<bool>, // 【新增】是否启用此检测
    pub stability: f32,
    pub threshold: f32,
    pub overlap: Option<f32>,
}

// 【新增】资源路径配置
#[derive(Debug, Deserialize, Clone)]
pub struct AssetsConfig {
    pub avatar: String,
    pub scale: f32,
    // 【新增】鼠标普通状态
    pub cursor_normal: String,
    pub cursor_scale_normal: f32,
    // 【新增】鼠标滚动状态
    pub cursor_scroll: String,
    pub cursor_scale_scroll: f32,
    // 【新增】文本编辑状态
    pub cursor_text: String,
    pub cursor_scale_text: f32,
    // 【新增】旋转速度
    pub rotation_speed: f32,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MouseTrackMode {
    Head,
    Hand,
}

// 【新增】
#[derive(Debug, Deserialize, Clone)]
pub struct MouseConfig {
    pub enabled: bool,
    pub track_mode: MouseTrackMode, // 【新增】追踪模式：head 或 hand
    pub virtual_cursor_mode: bool,
    pub auto_screen_size: bool, // 是否自动检测分辨率
    pub manual_width: f64,      // 手动备选宽度
    pub manual_height: f64,     // 手动备选高度
    pub sensitivity_x: f32,     // X轴灵敏度 (建议 1.5 ~ 3.0)
    pub sensitivity_y: f32,     // Y轴灵敏度
    pub smoothing: f32,         // 平滑系数 (Beta值，越小越平滑但延迟越高)
    pub dead_zone: f32,         // 中心死区 (像素)，防止微小抖动
    // 【新增】
    pub invert_x: bool,
    pub invert_y: bool,
}

// 【新增】快捷键结构体
#[derive(Debug, Deserialize, Clone)]
pub struct ShortcutsConfig {
    // 使用 Vec<String> 允许用户定义组合键，例如 ["LControl", "LAlt", "R"]
    pub reset_center: Vec<String>,
    pub toggle_mouse: Vec<String>,
}
// ==========================================
// 【新增】手势识别配置
// ==========================================
#[derive(Debug, Deserialize, Clone)]
pub struct GestureConfig {
    // 捏合检测阈值
    pub pinch_threshold_on: f32,  // 按下阈值（越小越灵敏）
    pub pinch_threshold_off: f32, // 松开阈值（需要 > on_threshold）

    // 旋转检测参数
    pub rotation_threshold_degrees: f32, // 旋转触发阈值（度数，推荐30~60）
    pub rotation_cooldown_ms: u128,      // 旋转冷却时间（毫秒）
}

// ==========================================
// 【新增】调试配置
// ==========================================
#[derive(Debug, Deserialize, Clone)]
pub struct DebugConfig {
    #[allow(dead_code)] // 【注】保留配置选项以便用户设置，即使不在代码中使用
    pub show_debug_window: bool, // 是否显示调试窗口
    pub window_always_on_top: bool, // 摄像头窗口是否置顶
}

#[derive(Debug, Deserialize, Clone)]
pub struct InferenceConfig {
    pub device: String,
    pub cpu_threads: usize,
}
