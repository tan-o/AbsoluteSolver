use crate::config::ShortcutsConfig;
use anyhow::Result;
use device_query::{DeviceQuery, DeviceState, Keycode};

// 【关键修复】确保这里只有一个定义，并且加上了 PartialEq
#[derive(PartialEq)]
pub enum AppAction {
    None,
    ToggleMouse,
    ResetAnchor,
}

pub struct InputManager {
    device_state: DeviceState,

    // 缓存解析好的按键组合
    toggle_keys: Vec<Keycode>,
    reset_keys: Vec<Keycode>,

    // 按键状态记录（用于去抖动）
    toggle_pressed: bool,
    reset_pressed: bool,
}

impl InputManager {
    pub fn new(config: &ShortcutsConfig) -> Result<Self> {
        Ok(Self {
            device_state: DeviceState::new(),
            toggle_keys: parse_keycodes(&config.toggle_mouse),
            reset_keys: parse_keycodes(&config.reset_center),
            toggle_pressed: false,
            reset_pressed: false,
        })
    }

    /// 轮询当前按键状态，返回触发的动作
    pub fn check_action(&mut self) -> AppAction {
        let keys: Vec<Keycode> = self.device_state.get_keys();
        let mut action = AppAction::None;

        // 1. 检查 Toggle
        let is_toggle_active =
            !self.toggle_keys.is_empty() && self.toggle_keys.iter().all(|k| keys.contains(k));

        if is_toggle_active {
            if !self.toggle_pressed {
                action = AppAction::ToggleMouse;
                self.toggle_pressed = true;
            }
        } else {
            self.toggle_pressed = false;
        }

        if action != AppAction::None {
            return action;
        }

        // 2. 检查 Reset
        let is_reset_active =
            !self.reset_keys.is_empty() && self.reset_keys.iter().all(|k| keys.contains(k));

        if is_reset_active {
            if !self.reset_pressed {
                action = AppAction::ResetAnchor;
                self.reset_pressed = true;
            }
        } else {
            self.reset_pressed = false;
        }

        action
    }
}

// 辅助函数：将配置文件的字符串解析为 Keycode
fn parse_keycodes(keys: &Vec<String>) -> Vec<Keycode> {
    keys.iter()
        .filter_map(|k| {
            match k.to_uppercase().as_str() {
                "LCONTROL" | "CTRL" => Some(Keycode::LControl),
                "RCONTROL" => Some(Keycode::RControl),
                "LALT" | "ALT" => Some(Keycode::LAlt),
                "RALT" => Some(Keycode::RAlt),
                "LSHIFT" | "SHIFT" => Some(Keycode::LShift),
                "SPACE" => Some(Keycode::Space),
                "ENTER" => Some(Keycode::Enter),
                "ESCAPE" | "ESC" => Some(Keycode::Escape),
                "TAB" => Some(Keycode::Tab),

                // 字母
                "A" => Some(Keycode::A),
                "B" => Some(Keycode::B),
                "C" => Some(Keycode::C),
                "D" => Some(Keycode::D),
                "E" => Some(Keycode::E),
                "F" => Some(Keycode::F),
                "G" => Some(Keycode::G),
                "H" => Some(Keycode::H),
                "I" => Some(Keycode::I),
                "J" => Some(Keycode::J),
                "K" => Some(Keycode::K),
                "L" => Some(Keycode::L),
                "M" => Some(Keycode::M),
                "N" => Some(Keycode::N),
                "O" => Some(Keycode::O),
                "P" => Some(Keycode::P),
                "Q" => Some(Keycode::Q),
                "R" => Some(Keycode::R),
                "S" => Some(Keycode::S),
                "T" => Some(Keycode::T),
                "U" => Some(Keycode::U),
                "V" => Some(Keycode::V),
                "W" => Some(Keycode::W),
                "X" => Some(Keycode::X),
                "Y" => Some(Keycode::Y),
                "Z" => Some(Keycode::Z),

                // 数字
                "0" => Some(Keycode::Key0),
                "1" => Some(Keycode::Key1),
                "2" => Some(Keycode::Key2),
                "3" => Some(Keycode::Key3),
                "4" => Some(Keycode::Key4),
                "5" => Some(Keycode::Key5),
                "6" => Some(Keycode::Key6),
                "7" => Some(Keycode::Key7),
                "8" => Some(Keycode::Key8),
                "9" => Some(Keycode::Key9),

                // 功能键
                "F1" => Some(Keycode::F1),
                "F2" => Some(Keycode::F2),
                "F3" => Some(Keycode::F3),
                "F4" => Some(Keycode::F4),
                "F5" => Some(Keycode::F5),
                "F6" => Some(Keycode::F6),
                "F7" => Some(Keycode::F7),
                "F8" => Some(Keycode::F8),
                "F9" => Some(Keycode::F9),
                "F10" => Some(Keycode::F10),
                "F11" => Some(Keycode::F11),
                "F12" => Some(Keycode::F12),

                _ => {
                    println!(">> [Config] 警告：未知按键 '{}'", k);
                    None
                }
            }
        })
        .collect()
}
