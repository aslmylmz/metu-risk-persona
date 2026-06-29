//! Tauri v2 shell for the offline BART instrument (SPEC §10).
//!
//! Kept deliberately thin: it opens the window that loads the Vite SPA and
//! registers the minimal plugins (native dialogs + scoped file access for
//! study.json). The sidecar lifecycle, port handoff, and commands land in
//! issue 11; the study.json load/save + kiosk toggle in issue 12.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .run(tauri::generate_context!())
        .expect("error while running the BART desktop shell");
}
