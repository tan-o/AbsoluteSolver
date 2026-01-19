use anyhow::Result;
use opencv::core::{Mat, Point2f, Rect, Scalar, Size as CvSize};
use opencv::imgproc;
use opencv::prelude::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

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
// 【新增】区分滚轮方向
pub const STATE_SCROLL_CW: i32 = 3; // 顺时针 (向下滚)
pub const STATE_SCROLL_CCW: i32 = 4; // 逆时针 (向上滚)

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

// src/overlay.rs

fn load_and_scale(path: &str, scale: f32) -> Result<Mat> {
    let raw = crate::algorithms::load_and_prepare_mask(path)?;
    let mut scaled = Mat::default();
    let new_w = (raw.cols() as f32 * scale) as i32;
    let new_h = (raw.rows() as f32 * scale) as i32;

    // 【关键修复】判断是放大还是缩小
    // 如果是放大 (scale > 1.0)，必须使用 INTER_LINEAR (双线性) 或 INTER_CUBIC (双三次)
    // INTER_AREA 在放大时会产生严重的锯齿（马赛克）
    let interpolation = if scale > 1.0 {
        imgproc::INTER_LINEAR // 推荐：平滑且快。如果想要更平滑可以使用 INTER_CUBIC
    } else {
        imgproc::INTER_AREA // 缩小保持用 AREA，效果最好
    };

    imgproc::resize(
        &raw,
        &mut scaled,
        CvSize::new(new_w, new_h),
        0.0,
        0.0,
        interpolation, // 使用正确的插值算法
    )?;
    Ok(scaled)
}

pub fn spawn_mouse_overlay(
    config: &crate::config::AssetsConfig,
    mouse_state: Arc<AtomicI32>,
) -> Result<()> {
    // 1. 预加载图片
    let img_normal = load_and_scale(&config.cursor_normal, config.cursor_scale_normal)?;
    let img_scroll = load_and_scale(&config.cursor_scroll, config.cursor_scale_scroll)?;
    let img_text = load_and_scale(&config.cursor_text, config.cursor_scale_text)?;

    // 2. 计算最大窗口
    let get_diag = |m: &Mat| ((m.cols().pow(2) + m.rows().pow(2)) as f64).sqrt();
    let max_diag = get_diag(&img_normal)
        .max(get_diag(&img_scroll))
        .max(get_diag(&img_text));

    let win_size = max_diag.ceil() as i32;
    let win_center = win_size as f32 / 2.0;

    // 基础旋转速度 (绝对值)
    let base_speed = config.rotation_speed;

    thread::spawn(move || {
        unsafe {
            let instance = HINSTANCE(GetModuleHandleW(None).unwrap().0);
            let class_name = w!("RustMouseOverlay");

            // 系统 I-Beam 句柄
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

            // 主画布
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

                // ==========================================
                // 1. 检测文本模式 (VS Code 兼容增强版)
                // ==========================================
                let mut is_text_mode = false;

                // 检测 A: 句柄比对 (原生应用)
                if GetCursorInfo(&mut ci).is_ok() {
                    if !system_ibeam_cursor.0.is_null() && ci.hCursor == system_ibeam_cursor {
                        is_text_mode = true;
                    }
                }

                // 检测 B: 插入符检测 (VS Code / Electron)
                if !is_text_mode {
                    let foreground = GetForegroundWindow();
                    if !foreground.0.is_null() {
                        let thread_id = GetWindowThreadProcessId(foreground, None);
                        let mut gui_info = GUITHREADINFO {
                            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
                            ..Default::default()
                        };

                        if GetGUIThreadInfo(thread_id, &mut gui_info).is_ok() {
                            // 只要有输入光标在闪烁，或者光标句柄存在，就认为是编辑状态
                            if (gui_info.flags & GUI_CARETBLINKING).0 != 0 {
                                is_text_mode = true;
                            }
                        }
                    }
                }

                // ==========================================
                // 2. 决定图片、旋转开关、旋转方向
                // ==========================================
                let current_src_img;
                let should_rotate;
                let mut current_speed = base_speed; // 默认顺时针

                if is_text_mode {
                    current_src_img = &img_text;
                    should_rotate = true; // 文本模式通常不旋转
                } else {
                    match app_state {
                        STATE_SCROLL_CW => {
                            current_src_img = &img_scroll;
                            should_rotate = true;
                            current_speed = base_speed; // 正数：顺时针
                        }
                        STATE_SCROLL_CCW => {
                            current_src_img = &img_scroll;
                            should_rotate = true;
                            current_speed = -base_speed; // 负数：逆时针
                        }
                        STATE_CLICK_HOLD => {
                            current_src_img = &img_normal;
                            should_rotate = false;
                        }
                        _ => {
                            current_src_img = &img_normal;
                            should_rotate = true;
                            current_speed = base_speed; // 普通状态顺时针
                        }
                    }
                }

                // ==========================================
                // 3. 绘制逻辑 (修复变花问题)
                // ==========================================
                canvas.set_to(&Scalar::all(0.0), &Mat::default()).unwrap();

                let now = Instant::now();
                let dt = now.duration_since(last_time).as_secs_f32();
                last_time = now;

                if should_rotate && current_speed.abs() > 0.001 {
                    current_angle += current_speed * dt;
                    current_angle %= 360.0;
                }

                // 复制到中心
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

                // 最终要显示的 Mat
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

                // 【核心修复】强制 Clone 以保证内存连续性 (Contiguous)
                // Windows GDI 对内存步长(Stride)非常敏感，OpenCV 的 ROI 或 Warp 结果
                // 有时会产生非连续内存或 padding，直接传给 CreateBitmap 就会变花。
                // continuous_mat 保证了数据是紧凑排列的 BGRA。
                let continuous_mat = final_display_mat.clone();

                if let Ok(data) = continuous_mat.data_bytes() {
                    update_layered_window_raw(hwnd, data, win_size, win_size);
                }

                // 移动窗口
                let mut p = POINT::default();
                let _ = GetCursorPos(&mut p);
                let x = p.x - (win_size / 2);
                let y = p.y - (win_size / 2);
                let _ = SetWindowPos(
                    hwnd,
                    Some(HWND_TOPMOST),
                    x,
                    y,
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

// src/overlay.rs

// src/overlay.rs

unsafe fn update_layered_window_raw(hwnd: HWND, data: &[u8], w: i32, h: i32) {
    let screen_dc = GetDC(None);
    let mem_dc = CreateCompatibleDC(Some(screen_dc));

    // 1. 设置位图头 (保持你上次修改的 Top-Down 负数高度)
    let bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: w,
            biHeight: -h, // Top-down
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
        // ============================================================
        // 【关键修复】执行 Premultiplied Alpha (预乘 Alpha)
        // ============================================================
        // 我们不能直接 copy，必须逐像素处理。
        // 为了性能，我们把 u8 数组转换成可变的切片进行操作。
        // 注意：data 是 OpenCV 的数据，p_bits 是我们要写入 Windows 的内存。

        let src_slice = data;
        let dst_slice = std::slice::from_raw_parts_mut(p_bits as *mut u8, src_slice.len());

        // 使用 chunks_exact(4) 每次处理一个像素 (B, G, R, A)
        // 这样比手动索引快，且利用 CPU 缓存
        for (i, chunk) in src_slice.chunks_exact(4).enumerate() {
            let b = chunk[0] as u32;
            let g = chunk[1] as u32;
            let r = chunk[2] as u32;
            let a = chunk[3] as u32;

            // 只有当 Alpha > 0 时才需要计算，否则直接全 0
            if a > 0 {
                // 核心公式: Color_new = Color_old * Alpha / 255
                // 这里的 dst_idx 就是 i * 4
                let dst_idx = i * 4;
                dst_slice[dst_idx + 0] = ((b * a) / 255) as u8; // B
                dst_slice[dst_idx + 1] = ((g * a) / 255) as u8; // G
                dst_slice[dst_idx + 2] = ((r * a) / 255) as u8; // R
                dst_slice[dst_idx + 3] = a as u8; // A (保持不变)
            } else {
                // Alpha 为 0，全清零
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
        AlphaFormat: AC_SRC_ALPHA as u8, // 这里告诉 Windows 我们已经预乘过了
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
