// src/overlay.rs

use anyhow::Result;
use opencv::core::{Mat, Point2f, Rect, Scalar, Size as CvSize};
use opencv::imgproc;
use opencv::prelude::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::config::AnchorPoint; // 引入锚点枚举
use crate::interaction::SharedCursorCoords;

use windows::core::w;
use windows::Win32::Foundation::{COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, POINT, SIZE, WPARAM};
use windows::Win32::Graphics::Gdi::{
    CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, GetDC, ReleaseDC, SelectObject,
    AC_SRC_ALPHA, AC_SRC_OVER, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, BLENDFUNCTION, DIB_RGB_COLORS,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetCursorInfo, GetCursorPos,
    GetForegroundWindow, GetGUIThreadInfo, GetWindowThreadProcessId, LoadCursorW, PeekMessageW,
    RegisterClassW, SetWindowPos, ShowWindow, UpdateLayeredWindow, CS_HREDRAW, CS_VREDRAW,
    CURSORINFO, GUITHREADINFO, GUI_CARETBLINKING, HCURSOR, HICON, HWND_TOPMOST, IDC_IBEAM, MSG,
    PM_REMOVE, SWP_NOACTIVATE, SWP_NOSIZE, SW_HIDE, SW_SHOWNA, ULW_ALPHA, WNDCLASSW, WS_EX_LAYERED,
    WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP, WS_VISIBLE,
};

// ==========================================
// 状态定义
// ==========================================
pub const STATE_HIDDEN: i32 = 0;
pub const STATE_NORMAL: i32 = 1;
pub const STATE_CLICK_HOLD: i32 = 2;
pub const STATE_SCROLL_CW: i32 = 3;
pub const STATE_SCROLL_CCW: i32 = 4;

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

fn load_and_scale(path: &str, scale: f32) -> Result<Mat> {
    let raw = crate::algorithms::load_and_prepare_mask(path)?;
    let mut scaled = Mat::default();
    let new_w = (raw.cols() as f32 * scale) as i32;
    let new_h = (raw.rows() as f32 * scale) as i32;

    let interpolation = if scale > 1.0 {
        imgproc::INTER_LINEAR
    } else {
        imgproc::INTER_AREA
    };

    imgproc::resize(
        &raw,
        &mut scaled,
        CvSize::new(new_w, new_h),
        0.0,
        0.0,
        interpolation,
    )?;
    Ok(scaled)
}

pub fn spawn_mouse_overlay(
    config: &crate::config::AssetsConfig,
    mouse_state: Arc<AtomicI32>,
    shared_coords: Arc<SharedCursorCoords>,
    is_virtual_mode: bool,
) -> Result<()> {
    // 1. 预加载图片
    let img_normal = load_and_scale(&config.cursor_normal, config.cursor_scale_normal)?;
    let img_scroll = load_and_scale(&config.cursor_scroll, config.cursor_scale_scroll)?;
    let img_text = load_and_scale(&config.cursor_text, config.cursor_scale_text)?;

    // 获取锚点配置
    let anchor_mode = config.anchor;

    // 2. 计算最大窗口
    let get_diag = |m: &Mat| ((m.cols().pow(2) + m.rows().pow(2)) as f64).sqrt();
    let max_diag = get_diag(&img_normal)
        .max(get_diag(&img_scroll))
        .max(get_diag(&img_text));

    let win_size = max_diag.ceil() as i32;
    let win_center = win_size as f32 / 2.0;
    let base_speed = config.rotation_speed;

    thread::spawn(move || {
        unsafe {
            let instance = HINSTANCE(GetModuleHandleW(None).unwrap().0);
            let class_name = w!("RustMouseOverlay");

            let system_ibeam_cursor =
                LoadCursorW(None, IDC_IBEAM).unwrap_or(HCURSOR(std::ptr::null_mut()));

            let wnd_class = WNDCLASSW {
                lpfnWndProc: Some(wnd_proc),
                hInstance: instance,
                lpszClassName: class_name,
                style: CS_HREDRAW | CS_VREDRAW,
                hCursor: HCURSOR(std::ptr::null_mut()),
                hIcon: HICON(std::ptr::null_mut()),
                ..Default::default()
            };
            RegisterClassW(&wnd_class);

            let hwnd = CreateWindowExW(
                WS_EX_LAYERED
                    | WS_EX_TRANSPARENT
                    | WS_EX_TOPMOST
                    | WS_EX_TOOLWINDOW
                    | WS_EX_NOACTIVATE,
                class_name,
                w!("Overlay"),
                WS_POPUP | WS_VISIBLE,
                0,
                0,
                win_size,
                win_size,
                None,
                None,
                Some(instance),
                None,
            )
            .expect("Failed to create window");

            let mut msg = MSG::default();
            let mut current_angle = 0.0f32;
            let mut last_time = Instant::now();

            let mut canvas = Mat::new_rows_cols_with_default(
                win_size,
                win_size,
                opencv::core::CV_8UC4,
                Scalar::all(0.0),
            )
            .unwrap();

            let mut ci = CURSORINFO {
                cbSize: std::mem::size_of::<CURSORINFO>() as u32,
                ..Default::default()
            };

            loop {
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    DispatchMessageW(&msg);
                }

                let app_state = mouse_state.load(Ordering::Relaxed);

                if app_state == STATE_HIDDEN {
                    let _ = ShowWindow(hwnd, SW_HIDE);
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }

                let _ = ShowWindow(hwnd, SW_SHOWNA);

                let mut is_text_mode = false;
                if GetCursorInfo(&mut ci).is_ok() {
                    if !system_ibeam_cursor.0.is_null() && ci.hCursor == system_ibeam_cursor {
                        is_text_mode = true;
                    }
                }
                if !is_text_mode {
                    let foreground = GetForegroundWindow();
                    if !foreground.0.is_null() {
                        let thread_id = GetWindowThreadProcessId(foreground, None);
                        let mut gui_info = GUITHREADINFO {
                            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
                            ..Default::default()
                        };
                        if GetGUIThreadInfo(thread_id, &mut gui_info).is_ok() {
                            if (gui_info.flags & GUI_CARETBLINKING).0 != 0 {
                                is_text_mode = true;
                            }
                        }
                    }
                }

                let current_src_img;
                let should_rotate;
                let mut current_speed = base_speed;

                if is_text_mode {
                    current_src_img = &img_text;
                    should_rotate = true;
                } else {
                    match app_state {
                        STATE_SCROLL_CW => {
                            current_src_img = &img_scroll;
                            should_rotate = true;
                            current_speed = base_speed;
                        }
                        STATE_SCROLL_CCW => {
                            current_src_img = &img_scroll;
                            should_rotate = true;
                            current_speed = -base_speed;
                        }
                        STATE_CLICK_HOLD => {
                            current_src_img = &img_normal;
                            should_rotate = false;
                        }
                        _ => {
                            current_src_img = &img_normal;
                            should_rotate = true;
                            current_speed = base_speed;
                        }
                    }
                }

                canvas.set_to(&Scalar::all(0.0), &Mat::default()).unwrap();

                let now = Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32();
                last_time = now;

                if should_rotate && current_speed.abs() > 0.001 {
                    current_angle += current_speed * dt;
                    current_angle %= 360.0;
                }

                let target_x = (win_size - current_src_img.cols()) / 2;
                let target_y = (win_size - current_src_img.rows()) / 2;
                let rect = Rect::new(
                    target_x,
                    target_y,
                    current_src_img.cols(),
                    current_src_img.rows(),
                );

                let mut roi_mat = Mat::roi_mut(&mut canvas, rect).unwrap();
                current_src_img.copy_to(&mut roi_mat).unwrap();

                let final_display_mat: Mat;

                if current_angle.abs() > 0.1 {
                    let rot_mat = imgproc::get_rotation_matrix_2d(
                        Point2f::new(win_center, win_center),
                        current_angle as f64,
                        1.0,
                    )
                    .unwrap();

                    let mut rotated_canvas = Mat::default();
                    imgproc::warp_affine(
                        &canvas,
                        &mut rotated_canvas,
                        &rot_mat,
                        CvSize::new(win_size, win_size),
                        imgproc::INTER_LINEAR,
                        opencv::core::BORDER_CONSTANT,
                        Scalar::default(),
                    )
                    .unwrap();

                    final_display_mat = rotated_canvas;
                } else {
                    final_display_mat = canvas.clone();
                }

                let continuous_mat = final_display_mat.clone();
                if let Ok(data) = continuous_mat.data_bytes() {
                    update_layered_window_raw(hwnd, data, win_size, win_size);
                }

                // ==========================================
                // 决定窗口位置 (加入 Anchor 逻辑)
                // ==========================================
                let (mouse_x, mouse_y) = if is_virtual_mode {
                    let vx = shared_coords.x.load(Ordering::Relaxed);
                    let vy = shared_coords.y.load(Ordering::Relaxed);
                    (vx, vy)
                } else {
                    let mut p = POINT::default();
                    let _ = GetCursorPos(&mut p);
                    (p.x, p.y)
                };

                // 计算偏移量
                // win_size 是正方形窗口边长
                let offset_x;
                let offset_y;

                match anchor_mode {
                    AnchorPoint::LU => {
                        offset_x = 0;
                        offset_y = 0;
                    }
                    AnchorPoint::LD => {
                        offset_x = 0;
                        offset_y = win_size;
                    }
                    AnchorPoint::RU => {
                        offset_x = win_size;
                        offset_y = 0;
                    }
                    AnchorPoint::RD => {
                        offset_x = win_size;
                        offset_y = win_size;
                    }
                    AnchorPoint::L => {
                        offset_x = 0;
                        offset_y = win_size / 2;
                    }
                    AnchorPoint::R => {
                        offset_x = win_size;
                        offset_y = win_size / 2;
                    }
                    AnchorPoint::U => {
                        offset_x = win_size / 2;
                        offset_y = 0;
                    }
                    AnchorPoint::D => {
                        offset_x = win_size / 2;
                        offset_y = win_size;
                    }
                    AnchorPoint::C => {
                        offset_x = win_size / 2;
                        offset_y = win_size / 2;
                    }
                }

                // 最终窗口坐标 = 鼠标坐标 - 偏移量
                let final_win_x = mouse_x - offset_x;
                let final_win_y = mouse_y - offset_y;

                let _ = SetWindowPos(
                    hwnd,
                    Some(HWND_TOPMOST),
                    final_win_x,
                    final_win_y,
                    0,
                    0,
                    SWP_NOSIZE | SWP_NOACTIVATE,
                );

                thread::sleep(Duration::from_millis(16));
            }
        }
    });
    Ok(())
}

unsafe fn update_layered_window_raw(hwnd: HWND, data: &[u8], w: i32, h: i32) {
    let screen_dc = GetDC(None);
    let mem_dc = CreateCompatibleDC(Some(screen_dc));

    let bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: w,
            biHeight: -h,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            biSizeImage: (w * h * 4) as u32,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut p_bits: *mut c_void = std::ptr::null_mut();
    let hbitmap = CreateDIBSection(
        Some(mem_dc),
        &bmi as *const _ as *const _,
        DIB_RGB_COLORS,
        &mut p_bits,
        None,
        0,
    )
    .unwrap();

    if !p_bits.is_null() {
        let src_slice = data;
        let dst_slice = std::slice::from_raw_parts_mut(p_bits as *mut u8, src_slice.len());

        for (i, chunk) in src_slice.chunks_exact(4).enumerate() {
            let b = chunk[0] as u32;
            let g = chunk[1] as u32;
            let r = chunk[2] as u32;
            let a = chunk[3] as u32;

            if a > 0 {
                let dst_idx = i * 4;
                dst_slice[dst_idx + 0] = ((b * a) / 255) as u8;
                dst_slice[dst_idx + 1] = ((g * a) / 255) as u8;
                dst_slice[dst_idx + 2] = ((r * a) / 255) as u8;
                dst_slice[dst_idx + 3] = a as u8;
            } else {
                let dst_idx = i * 4;
                dst_slice[dst_idx + 0] = 0;
                dst_slice[dst_idx + 1] = 0;
                dst_slice[dst_idx + 2] = 0;
                dst_slice[dst_idx + 3] = 0;
            }
        }
    }

    let old_bitmap = SelectObject(mem_dc, hbitmap.into());

    let size = SIZE { cx: w, cy: h };
    let point = POINT { x: 0, y: 0 };
    let blend = BLENDFUNCTION {
        BlendOp: AC_SRC_OVER as u8,
        BlendFlags: 0,
        SourceConstantAlpha: 255,
        AlphaFormat: AC_SRC_ALPHA as u8,
    };

    let _ = UpdateLayeredWindow(
        hwnd,
        Some(screen_dc),
        None,
        Some(&size),
        Some(mem_dc),
        Some(&point),
        COLORREF(0),
        Some(&blend),
        ULW_ALPHA,
    );

    SelectObject(mem_dc, old_bitmap);
    let _ = DeleteObject(hbitmap.into());
    let _ = DeleteDC(mem_dc);
    let _ = ReleaseDC(None, screen_dc);
}
